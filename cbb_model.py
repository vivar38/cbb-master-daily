"""CBB Master v1.0 | historical team-stat foundation, not an approved wagering model.

Reproducible, day-grouped pregame feature engineering and true forward-season holdout.
No current-season results, end-of-season ratings, market odds, or future roster data
enter a game's prediction features. Input: Kaggle MRegularSeasonDetailedResults.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, brier_score_loss

MARGIN_COLS = [
    'venue_a', 'elo_diff', 'net_quality_diff', 'matchup_o_diff',
    'shoot_diff', 'tov_diff', 'orb_diff', 'ftr_diff', 'rest_diff',
    'games_min', 'season_day'
]
TOTAL_COLS = [
    'venue_a', 'mean_pace', 'off_match_sum', 'off_quality_sum',
    'def_quality_sum', 'shoot_sum', 'tov_sum', 'orb_sum', 'ftr_sum',
    'games_min', 'season_day'
]
REQUIRED = [
    'Season','DayNum','WTeamID','WScore','LTeamID','LScore','WLoc','NumOT',
    'WFGM','WFGA','WFGM3','WFGA3','WFTM','WFTA','WOR','WDR','WTO',
    'LFGM','LFGA','LFGM3','LFGA3','LFTM','LFTA','LOR','LDR','LTO'
]
PRIOR = {'off': 101., 'def': 101., 'pace': 69., 'efg': .5,
         'tov': .18, 'orb': .30, 'ftr': .33}
PRIOR_GAMES = 5.


def validate_input(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing historical input: {path}. See README.md for Kaggle source.')
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if frame.empty:
        raise ValueError('Historical input is empty')
    if frame['WLoc'].isin(['H','A','N']).sum() != len(frame):
        raise ValueError('Unrecognized WLoc (expected H/A/N)')
    numeric = [c for c in REQUIRED if c != 'WLoc']
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='raise')
    if frame[numeric].isna().any().any():
        raise ValueError('Missing required numeric fields')
    if (frame.WTeamID == frame.LTeamID).any() or (frame.WScore <= frame.LScore).any():
        raise ValueError('Invalid winner/loser rows')
    return frame.sort_values(['Season', 'DayNum', 'WTeamID', 'LTeamID']).reset_index(drop=True)


def normalize_games(raw: pd.DataFrame) -> pd.DataFrame:
    # Deterministic ID ordering prevents winner/loser identity from leaking into features.
    a_win = raw.WTeamID.to_numpy() < raw.LTeamID.to_numpy()
    a = np.where(a_win, raw.WTeamID, raw.LTeamID)
    b = np.where(a_win, raw.LTeamID, raw.WTeamID)
    points_a = np.where(a_win, raw.WScore, raw.LScore)
    points_b = np.where(a_win, raw.LScore, raw.WScore)
    loc_w = raw.WLoc.to_numpy()
    venue = np.where(loc_w == 'N', 0, np.where(loc_w == 'H', 1, -1)) * np.where(a_win, 1, -1)
    g = pd.DataFrame({'season':raw.Season.astype(int), 'day_num':raw.DayNum.astype(int),
                      'team_a':a.astype(int), 'team_b':b.astype(int), 'venue_a':venue.astype(int),
                      'score_a':points_a.astype(float), 'score_b':points_b.astype(float),
                      'num_ot':raw.NumOT.astype(int)})
    for prefix, arr in [('a', a_win), ('b', ~a_win)]:
        for field_name, short in [('FGM','fgm'), ('FGA','fga'), ('FGM3','fgm3'),('FGA3','fga3'),
                                  ('FTM','ftm'), ('FTA','fta'), ('OR','or'), ('DR','dr'),('TO','to')]:
            g[f'{prefix}_{short}'] = np.where(arr, raw[f'W{field_name}'], raw[f'L{field_name}']).astype(float)
    g['margin'] = g.score_a - g.score_b
    g['total'] = g.score_a + g.score_b
    g['poss_a'] = g.a_fga - g.a_or + g.a_to + .475*g.a_fta
    g['poss_b'] = g.b_fga - g.b_or + g.b_to + .475*g.b_fta
    g['poss'] = (g.poss_a + g.poss_b)/2.
    if (g.poss < 35).any():
        raise ValueError('Implausible estimated possessions: check detailed-result fields')
    return g.sort_values(['season','day_num','team_a','team_b']).reset_index(drop=True)


@dataclass
class TeamState:
    n: int = 0
    elo: float = 1500.0
    sums: dict = field(default_factory=lambda: {k: 0. for k in PRIOR})
    last_day: int = -99

    def get(self, key):
        return (self.sums[key] + PRIOR_GAMES * PRIOR[key])/(self.n + PRIOR_GAMES)


def get_feat(game, states):
    aa = states.setdefault(int(game.team_a), TeamState())
    bb = states.setdefault(int(game.team_b), TeamState())
    v = int(game.venue_a)
    aoff, boff = aa.get('off'), bb.get('off')
    adef, bdef = aa.get('def'), bb.get('def')
    # A higher defensive rating means worse defense; interpolate weak-sample opponent adjustment.
    oa = aoff + .5*(bdef - 101.)
    ob = boff + .5*(adef - 101.)
    out = {
        'venue_a':float(v), 'elo_diff':(aa.elo - bb.elo)/100.,
        'net_quality_diff':(aoff-adef)-(boff-bdef), 'matchup_o_diff':oa-ob,
        'mean_pace':(aa.get('pace') + bb.get('pace'))/2.,
        'off_match_sum':oa+ob,
        'off_quality_sum':aoff+boff, 'def_quality_sum':adef+bdef,
        'games_min':float(min(aa.n,bb.n)), 'games_a':aa.n, 'games_b':bb.n,
        'season_day':float(game.day_num),
        'rest_diff':float(np.clip((game.day_num - aa.last_day)-(game.day_num - bb.last_day),-7,7)) if aa.n and bb.n else 0.,
    }
    for key, prefix in [('efg','shoot'),('tov','tov'),('orb','orb'),('ftr','ftr')]:
        out[prefix+'_diff'] = aa.get(key) - bb.get(key)
        out[prefix+'_sum'] = aa.get(key) + bb.get(key)
    return out


def update_states(game, snapshot, states):
    a, b = states[int(game.team_a)], states[int(game.team_b)]
    # Snapshot states from before today's games, not updated by intra-day matches.
    apre, bpre = snapshot
    poss = float(game.poss)
    pace = poss*40/(40 + 5*int(game.num_ot))
    observed = []
    for prefix, our, their, opp in [('a', game.score_a, game.score_b, bpre),
                                   ('b', game.score_b, game.score_a, apre)]:
        fga = float(game[prefix+'_fga']); fg3 = float(game[prefix+'_fgm3'])
        fgm = float(game[prefix+'_fgm']); fta = float(game[prefix+'_fta'])
        ora = float(game[prefix+'_or']); tov = float(game[prefix+'_to'])
        opposing = 'b' if prefix=='a' else 'a'
        oppdr = float(game[opposing+'_dr'])
        sample = {
            'off':float(our)/poss*100 - .50*(opp['def']-101.),
            'def':float(their)/poss*100 - .50*(opp['off']-101.),
            'pace':pace, 'efg':(fgm + .5*fg3)/max(fga,1),
            'tov':tov/max(fga + .475*fta - ora + tov,1),
            'orb':ora/max(ora+oppdr,1), 'ftr':fta/max(fga,1),
        }
        observed.append(sample)
    for state, obs in [(a,observed[0]),(b,observed[1])]:
        for key in PRIOR: state.sums[key] += obs[key]
        state.n += 1
        state.last_day = int(game.day_num)
    home_adv_elo = 55. * int(game.venue_a)
    expect = 1/(1+10**(-((apre['elo']-bpre['elo'])+home_adv_elo)/400))
    target = 1. if game.margin>0 else 0.
    # Do not tune Elo with future results; fixed preregistered K value.
    delta = 22. * (target-expect) * (1.+ min(math.log1p(abs(float(game.margin)))/4., .65))
    a.elo += delta
    b.elo -= delta


def snapshot_state(s):
    return {key:s.get(key) for key in PRIOR} | {'elo':s.elo}


def build_pregame_features(games):
    rows=[]
    for season, season_games in games.groupby('season',sort=True):
        # Reinitialize teams each season. No false assumption of roster continuity.
        states={}
        for day, todays_games in season_games.groupby('day_num',sort=True):
            pending=[]
            for row in todays_games.itertuples(index=False):
                d=row._asdict()
                # Use a named pd.Series to preserve arbitrary statistics keys.
                series=pd.Series(d)
                feats=get_feat(series,states)
                rows.append({'season':int(season),'day_num':int(day),
                             'team_a':int(series.team_a),'team_b':int(series.team_b),
                             'score_a':float(series.score_a),'score_b':float(series.score_b),
                             'margin':float(series.margin),'total':float(series.total),
                             'num_ot':int(series.num_ot),**feats})
                pending.append((series, (snapshot_state(states[int(series.team_a)]),
                                         snapshot_state(states[int(series.team_b)]))))
            for game,snap in pending:
                update_states(game,snap,states)
    frame=pd.DataFrame(rows)
    if frame.duplicated(['season','day_num','team_a','team_b']).any():
        raise ValueError('Duplicate game key')
    return frame


def estimator(alpha=35.0):
    return Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))])


def model_report(y, pred, kind, std=None):
    out={'count':int(len(y)),'mae':round(float(mean_absolute_error(y,pred)),4),
         'bias':round(float(np.mean(pred-y)),4),
         'rmse':round(float(np.sqrt(np.mean((pred-y)**2))),4)}
    if kind=='margin':
        actual_win = np.asarray(y)>0
        p=ndtr(np.asarray(pred)/max(float(std),1))
        out['winner_accuracy']=round(float(np.mean((np.asarray(pred)>0)==actual_win)),4)
        out['brier']=round(float(brier_score_loss(actual_win,p)),4)
    return out


def run_backtest(raw_file:Path, output:Path, train_end:int=2024, valid:int=2025, test:int=2026, train_start:int=2018):
    output.mkdir(parents=True,exist_ok=True)
    raw=validate_input(raw_file)
    games=normalize_games(raw)
    feats=build_pregame_features(games)
    tr=feats[feats.season.between(train_start,train_end)]
    if tr.season.nunique()<3: raise ValueError('Need at least 3 historical seasons of training.')
    if not (train_start<=train_end<valid<test): raise ValueError('Require train_start <= train_end < validation < test season')
    val=feats[feats.season==valid]
    tst=feats[feats.season==test]
    if val.empty or tst.empty: raise ValueError('Validation and test season must be present and nonempty')
    if (feats.season>test).any():
        # Extra future seasons can exist, but will never enter training or evaluation.
        pass
    margin=estimator(); total=estimator()
    margin.fit(tr[MARGIN_COLS],tr['margin'])
    total.fit(tr[TOTAL_COLS],tr['total'])
    train_m=margin.predict(tr[MARGIN_COLS]); train_t=total.predict(tr[TOTAL_COLS])
    # Train residuals only, no target-data leakage into probability scaling.
    std_m=float(np.std(tr.margin - train_m,ddof=1))
    std_t=float(np.std(tr.total - train_t,ddof=1))
    base_m=estimator()
    base_m.fit(tr[['elo_diff','venue_a']],tr.margin)
    base_total=float(tr.total.mean())
    coverage=feats.groupby('season').agg(game_rows=('total','size'),teams_a=('team_a','nunique'),teams_b=('team_b','nunique'),last_day=('day_num','max')).reset_index()
    coverage.to_csv(output/'season_coverage.csv',index=False)
    test_day_max=int(feats.loc[feats.season==test,'day_num'].max())
    pred=[]; report={'model_id':'CBB-MASTER-v1.0-TEAM-BASELINE-RESEARCH',
                      'model_status':'RESEARCH_ONLY_NO_PREGAME_LOCK',
                      'data_source':str(raw_file),'sha256':hashlib.sha256(raw_file.read_bytes()).hexdigest(),
                      'data_quality_warnings':(['TEST_SEASON_MAY_BE_INCOMPLETE: last DayNum < 125'] if test_day_max<125 else []),
                      'data_coverage':{'min_season':int(feats.season.min()), 'max_season':int(feats.season.max()),
                                       'all_rows':int(len(feats)), 'train_rows':int(len(tr)),
                                       'validation_rows':int(len(val)), 'test_rows':int(len(tst))},
                      'splits':{'train_from':train_start,'train_through':train_end,'validation':valid,'test':test},
                      'probability_assumption':'Normal CDF of predicted margin / train residual std; not separately calibrated',
                      'margin_train_residual_sd':std_m,'total_train_residual_sd':std_t,
                      'odds_or_betting_ev_tested':False,'roster_and_transfer_adjustments_tested':False}
    for name,data in [('validation',val),('test',tst)]:
        pm=margin.predict(data[MARGIN_COLS]); pt=total.predict(data[TOTAL_COLS])
        eb=base_m.predict(data[['elo_diff','venue_a']]); bt=np.full(len(data),base_total)
        report[name]={'margin':model_report(data.margin,pm,'margin',std_m),
                      'total':model_report(data.total,pt,'total',std_t),
                      'baseline_elo_margin':model_report(data.margin,eb,'margin',std_m),
                      'baseline_league_total':model_report(data.total,bt,'total',std_t)}
        for bucket,cond in [('games_0_4',data.games_min<=4),
                            ('games_5_10',data.games_min.between(5,10)),
                            ('games_11_plus',data.games_min>=11)]:
            if cond.any():
                report[name][bucket]={'margin':model_report(data.loc[cond,'margin'],pm[cond.to_numpy()],'margin',std_m),
                                      'total':model_report(data.loc[cond,'total'],pt[cond.to_numpy()],'total',std_t)}
        output_batch=data[['season','day_num','team_a','team_b','venue_a','games_a','games_b','score_a','score_b','margin','total']].copy()
        output_batch['split']=name
        output_batch['pred_margin']=pm
        output_batch['pred_total']=pt
        output_batch['pred_score_a']=(pt+pm)/2
        output_batch['pred_score_b']=(pt-pm)/2
        output_batch['pred_prob_a_win']=ndtr(pm/max(std_m,1))
        output_batch['baseline_pred_margin']=eb
        output_batch['baseline_pred_total']=bt
        pred.append(output_batch)
    predictions=pd.concat(pred,ignore_index=True)
    predictions.to_csv(output/'holdout_predictions.csv',index=False)
    feats.to_csv(output/'historical_pregame_features.csv',index=False)
    import joblib
    joblib.dump({'margin_model':margin,'total_model':total,
                 'margin_cols':MARGIN_COLS,'total_cols':TOTAL_COLS,
                 'margin_residual_sd':std_m,'total_residual_sd':std_t,
                 'fit_through_season':train_end,'approved_for_betting':False},
                output/'holdout_model.joblib')
    with open(output/'metrics.json','w') as f: json.dump(report,f,indent=2)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games',type=Path,default=Path('data/raw/MRegularSeasonDetailedResults.csv'))
    ap.add_argument('--out',type=Path,default=Path('outputs'))
    ap.add_argument('--train-start',type=int,default=2018)
    ap.add_argument('--train-end',type=int,default=2024)
    ap.add_argument('--valid',type=int,default=2025)
    ap.add_argument('--test',type=int,default=2026)
    args=ap.parse_args()
    result=run_backtest(args.games,args.out,args.train_end,args.valid,args.test,args.train_start)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
