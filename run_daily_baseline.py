"""CBB Master v1.0: reproduce the fitted *historical statistical baseline* for an ESPN season day.

This CLI does NOT perform the 16-point lineup/scouting adjustment or betting audit.
Historical fitted coefficients are fixed to 2018-24; team states use only earlier 2026-27 games.
The source CSVs are SportsDataverse's ESPN men’s basketball team box and schedule releases.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr

from cbb_model import (TeamState, get_feat, normalize_games, snapshot_state,
                       update_states, MARGIN_COLS, TOTAL_COLS)
from margin_v11 import build as build_v11, features as features_v11, State as StateV11
from fetch_sportsdataverse import fetch_public_asset, convert_one_year, NAMES

# Verified against the 2027 schedule: ACC, Big East, Big Ten, Big 12, SEC.
POWER_CONFERENCE_IDS = {2, 4, 7, 8, 23}


def states_before_day(converted: pd.DataFrame, season: int, cutoff_day: int):
    if converted.empty:
        return {}
    old=converted.loc[(converted.Season.eq(season)) & (converted.DayNum.lt(cutoff_day))]
    if old.empty:
        return {}
    games=normalize_games(old)
    states={}
    for _, group in games.groupby('day_num',sort=True):
        pending=[]
        for row in group.itertuples(index=False):
            game=pd.Series(row._asdict())
            get_feat(game,states)
            pending.append((game,(snapshot_state(states[int(game.team_a)]),
                                 snapshot_state(states[int(game.team_b)]))))
        for game,snapshot in pending:
            update_states(game,snapshot,states)
    return states


def forecast(converter_games:pd.DataFrame, schedule:pd.DataFrame, model:dict,
             season:int, day:date):
    expected={'game_id','season','season_type','game_date','home_id','away_id',
              'neutral_site','status_type_completed'}
    missing=expected-set(schedule.columns)
    if missing:
        raise ValueError(f'2027 schedule schema changed; missing: {sorted(missing)}')
    season_day=(day-date(season-1,10,1)).days
    if not 1 <= season_day <= 240:
        raise ValueError('Target date is not within season')
    s=schedule.copy()
    s['parsed_date']=pd.to_datetime(s['game_date'],errors='coerce').dt.date
    s=s.loc[pd.to_numeric(s['season'],errors='coerce').eq(season)
            & pd.to_numeric(s['season_type'],errors='coerce').eq(2)
            & s.parsed_date.eq(day)].copy()
    if s.empty:
        raise ValueError(f'No ESPN schedule rows for {day.isoformat()}')
    # Require an ESPN team identity and venue designation for each fixture.
    s=s.sort_values('game_id').drop_duplicates('game_id')
    games=converter_games.loc[converter_games.DayNum.lt(season_day)].copy()
    states=states_before_day(games,season,season_day)
    enh=model["enhanced_margin"]
    enhanced_states=build_v11(normalize_games(games),enh["prior"],emit=False) if not games.empty else {}
    rows=[]; skipped=[]
    for _, g in s.iterrows():
        try:
            home=int(g['home_id']);away=int(g['away_id']);gid=int(g['game_id'])
            if home == away or pd.isna(g['neutral_site']):
                raise ValueError('invalid team IDs or unknown venue')
            flag=str(g['neutral_site']).strip().lower()
            if flag not in ('true','false','0','1'):
                raise ValueError('unknown neutral-site boolean')
            neutral=flag in ('true','1')
            a=min(home,away);b=max(home,away)
            venue=0 if neutral else (1 if a==home else -1)
            game=pd.Series({'team_a':a,'team_b':b,'venue_a':venue,'day_num':season_day})
            f=get_feat(game,states)
            x_m=pd.DataFrame([f])[model['margin_cols']]
            x_t=pd.DataFrame([f])[model['total_cols']]
            original_margin=float(model['margin_model'].predict(x_m)[0]);t=float(model['total_model'].predict(x_t)[0])
            ff=features_v11(enhanced_states.get(a,StateV11()),enhanced_states.get(b,StateV11()),venue,season_day,enh['prior'])
            m=float(enh['model'].predict(pd.DataFrame([ff])[enh['cols']])[0])*enh['weight']+original_margin*(1-enh['weight'])
            bucket=f"{int(ff['n']<5)}:{int(abs(m)>=15)}"
            scale=enh['scales'][bucket]
            if abs(m)>=t: raise ValueError('Incoherent score projection')
            a_score=(t+m)/2;b_score=(t-m)/2
            home_s=a_score if a==home else b_score
            away_s=b_score if a==home else a_score
            hmargin=home_s-away_s
            hprob=float(ndtr(hmargin/max(float(scale['sd']),1)))
            rows.append({'season':season,'game_date':day.isoformat(),'game_id':gid,
                         'home_id':home,'away_id':away,
                         'home_name':g.get('home_display_name',str(home)),
                         'away_name':g.get('away_display_name',str(away)),
                         'neutral_site':neutral,'home_games_before':states.get(home,TeamState()).n,
                         'away_games_before':states.get(away,TeamState()).n,
                         'home_pred_score':round(home_s,2),'away_pred_score':round(away_s,2),
                         'home_pred_margin':round(hmargin,2),'pred_total':round(t,2),
                         'home_win_prob_uncalibrated':round(hprob,4),
                         'source_status':'V1.1_FITTED_BASELINE_ONLY_NO_16PT_AUDIT',
                         'model_version':'v1.1','margin_error_scale':round(scale['sd'],4),'total_error_scale':round(scale['total_rmse'],4),'uncertainty_bucket':bucket,
                         'data_cutoff_exclusive':day.isoformat()})
        except (ValueError,TypeError,KeyError) as exc:
            skipped.append({'game_id':str(g.get('game_id','')),'reason':str(exc)})
    return pd.DataFrame(rows),skipped,season_day,len(games)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date',required=True,help='ESPN Eastern-calendar game date, e.g. 2026-11-24')
    p.add_argument('--season',type=int,default=2027)
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--model',type=Path,default=Path('outputs/holdout_model.joblib'))
    p.add_argument('--out',type=Path,default=Path('outputs/daily'))
    p.add_argument('--refresh',action='store_true',help='Fetch current public 2027 season CSVs directly; requires network')
    args=p.parse_args()
    target=date.fromisoformat(args.date)
    if args.season!=target.year+(target.month>=7):
        # Season 2027 spans November 2026-April 2027; this check is a convenience warning only.
        expected=target.year+int(target.month>=7)
        if expected!=args.season:raise ValueError('Date does not match season year')
    source=args.data_dir/'sportsdataverse'
    if args.refresh:
        for typ in ('box','schedule'):
            path=source/NAMES[typ].format(year=args.season)
            # Refresh even an existing cached file to avoid stale daily results.
            path.unlink(missing_ok=True)
            fetch_public_asset(typ,args.season,source)
    boxpath=source/NAMES['box'].format(year=args.season)
    schedulepath=source/NAMES['schedule'].format(year=args.season)
    if not boxpath.exists() or not schedulepath.exists():
        raise FileNotFoundError('Current-year public CSVs are not locally available; retry with --refresh in a network-enabled environment')
    box=pd.read_csv(boxpath,low_memory=False)
    schedule=pd.read_csv(schedulepath,low_memory=False)
    converted,quality=convert_one_year(box,schedule,args.season,source)
    # Reject any games on or after the forecast date as input features.
    fitted=joblib.load(args.model)
    fitted["enhanced_margin"]=joblib.load(Path(__file__).with_name("margin_v11.joblib"))
    if fitted.get('fit_through_season')!=2024:
        raise ValueError('Unexpected trained model version; refusing to silently change coefficients')
    if fitted.get('margin_cols')!=MARGIN_COLS or fitted.get('total_cols')!=TOTAL_COLS:
        raise ValueError('Unexpected feature definitions; refusing to change engine')
    result,skipped,dnum,used=forecast(converted,schedule,fitted,args.season,target)
    args.out.mkdir(parents=True,exist_ok=True)
    name=f'CBB_baseline_{args.date}'
    result.to_csv(args.out/f'{name}.csv',index=False)
    conference_cols = ['game_id','home_conference_id','away_conference_id','game_date_time','status_type_name','time_valid']
    if not set(conference_cols).issubset(schedule.columns):
        raise ValueError('Schedule missing fields for routine power-conference selector')
    enriched = result.merge(schedule[conference_cols], on='game_id', validate='one_to_one')
    enriched['tip_CT'] = pd.to_datetime(enriched.game_date_time, utc=True, errors='coerce').dt.tz_convert('America/Chicago').astype(str)
    power = enriched.loc[enriched.home_conference_id.isin(POWER_CONFERENCE_IDS) | enriched.away_conference_id.isin(POWER_CONFERENCE_IDS)].sort_values(['tip_CT','game_id'])
    power.to_csv(args.out/f'CBB_{args.date}_power_conference_baseline.csv',index=False)
    schedule_date = pd.to_datetime(schedule.game_date, errors='coerce').dt.date
    complete = schedule.loc[(schedule_date < target) & schedule.status_type_completed.astype(str).str.lower().eq('true') & schedule.season.eq(args.season) & schedule.season_type.eq(2)]
    missing = complete.loc[~complete.game_id.isin(converted.SourceGameID), ['game_id','game_date','away_display_name','home_display_name']]
    report={'status':'V1.1_FITTED_BASELINE_ONLY_NOT_FULL_MASTER','model_version':'v1.1','season':args.season,
            'asof_date_exclusive':args.date,'season_day':dnum,'season_games_used':used,
            'fixtures_emitted':len(result),'fixtures_skipped':skipped,
            'power_conference_fixtures':len(power),
            'scheduled_completed_games_before_cutoff':len(complete),
            'missing_completed_games':missing.to_dict('records'),
            'latest_box_date':str(box.game_date.max()),
            'source_freshness_warning':'A local CSV snapshot is not a live feed. Refresh before each subsequent slate; verify provider timestamp and coverage.',
            'public_source':'SportsDataverse ESPN team boxes + schedules','data_quality':quality,
            'historical_training_ends':2024,
            'warning':'No 16-point personnel audit, current line audit or fitted ATS/total probabilities'}
    (args.out/f'{name}.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:report[k] for k in ['status','asof_date_exclusive','season_games_used','fixtures_emitted','fixtures_skipped']},indent=2))
    if result.empty:
        raise RuntimeError('No valid fixtures scored. Check schedule IDs and current-year source completeness.')

if __name__=='__main__':main()
