"""Validate fresh source inputs and publish immutable dated baseline plus readable latest report."""
from pathlib import Path
from datetime import datetime, date, timezone
import argparse, json, hashlib, shutil
import pandas as pd

POWER={2,4,7,8,23}

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--date',required=True); p.add_argument('--season',type=int,default=2027); p.add_argument('--out',type=Path,default=Path('outputs/daily')); p.add_argument('--source',type=Path,default=Path('data/raw/sportsdataverse')); p.add_argument('--publish',type=Path,default=Path('forecasts')); p.add_argument('--generated-at',help='Test replay only; not used by cloud workflow'); a=p.parse_args()
    d=date.fromisoformat(a.date); now=datetime.fromisoformat(a.generated_at) if a.generated_at else datetime.now(timezone.utc)
    if now.tzinfo is None: raise ValueError('Timestamp must be timezone-aware')
    base=f'CBB_baseline_{d}'; report=json.loads((a.out/f'{base}.json').read_text()); power=pd.read_csv(a.out/f'CBB_{d}_power_conference_baseline.csv'); schedule_path=a.source/f'mbb_schedule_{a.season}.csv'; box_path=a.source/f'team_box_{a.season}.csv'; s=pd.read_csv(schedule_path,low_memory=False); b=pd.read_csv(box_path,low_memory=False)
    if s.game_id.duplicated().any() or b.duplicated(['game_id','team_id']).any(): raise ValueError('Duplicate source keys; review upstream')
    if report['asof_date_exclusive']!=str(d) or report['season']!=a.season: raise ValueError('Wrong forecast cutoff')
    if report['fixtures_skipped']: raise ValueError('Schedule fixtures skipped: '+str(report['fixtures_skipped']))
    if not power.empty and power[['home_pred_score','away_pred_score','home_pred_margin','pred_total']].isna().any().any(): raise ValueError('Missing forecasts')
    prior=s[(s.season==a.season)&(s.season_type==2)&(s.game_date.astype(str)<str(d))].copy()
    final=prior.status_type_completed.astype(str).str.lower().eq('true'); exc=prior.status_type_name.isin(['STATUS_POSTPONED','STATUS_CANCELED','STATUS_CANCELLED'])
    pending=prior[~final & ~exc]
    if len(pending): raise ValueError(f'{len(pending)} prior scheduled games unresolved; source may be stale')
    count=report['scheduled_completed_games_before_cutoff']; used=report['season_games_used']; missing=report['missing_completed_games']; coverage=used/count if count else 0
    if not count or count!=int(final.sum()) or used>count or coverage<.99: raise ValueError(f'Completed-game coverage insufficient: {used}/{count}')
    relevant=set(power.home_id)|set(power.away_id); missing_ids={x['game_id'] for x in missing}; m=prior[prior.game_id.isin(missing_ids)]
    if set(m.home_id).intersection(relevant) or set(m.away_id).intersection(relevant): raise ValueError('A missing completed game involves a team on today\'s power-conference slate')
    if power.empty: print('No power-conference games scheduled; publishing an empty slate.')
    # Per-game flags; a late rerun cannot be called a prospective forecast.
    tip=pd.to_datetime(power.game_date_time,utc=True,errors='coerce'); scheduled=power.status_type_name.eq('STATUS_SCHEDULED'); valid_time=power.time_valid.astype(str).str.lower().eq('true')
    power['prospective_at_generation']=(tip>now)&scheduled&valid_time
    power['generated_at_utc']=now.astimezone(timezone.utc).isoformat()
    report.update({'generated_at_utc':now.astimezone(timezone.utc).isoformat(),'coverage':coverage,'prior_unresolved':len(pending),'missing_games_involving_today_power_teams':0,'prospective_power_games':int(power.prospective_at_generation.sum()),'source_sha256':{'team_box':digest(box_path),'schedule':digest(schedule_path),'fitted_model':digest(Path('holdout_model.joblib')),'engine':digest(Path('cbb_model.py'))},'publication_status':'VALIDATED_BASELINE_ONLY','validation_policy':'At least 99% completed schedule coverage; zero unresolved prior fixtures, duplicate keys or missing boxes involving today power teams. Does not independently verify completeness against all external scoreboards.'})
    dest=a.publish/str(d); dest.mkdir(parents=True,exist_ok=False)
    power.to_csv(dest/'power_conference_baseline.csv',index=False); shutil.copyfile(a.out/f'{base}.csv',dest/'all_baseline.csv'); (dest/'validation.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    lines=[f'# CBB Master v1.0 fitted baseline — {d}', '',f'Generated (UTC): {report["generated_at_utc"]}',f'Prior completed games used: {used}/{count} ({coverage:.2%}).',f'Power-conference fixtures: {len(power)}; prospective at generation: {report["prospective_power_games"]}.','Status: VALIDATED FITTED BASELINE ONLY — not a full 16-point Master lock or betting advice.','Current-season state, fixed 2018–24 fitted coefficients, five-game generic prior.','', '| Tip (CT) | Away | Home | Away pts | Home pts | Home margin | Total | Pregame at generation |','|---|---|---|---:|---:|---:|---:|---|']
    for r in power.itertuples():
        lines.append(f'| {r.tip_CT} | {r.away_name} | {r.home_name} | {r.away_pred_score:.2f} | {r.home_pred_score:.2f} | {r.home_pred_margin:+.2f} | {r.pred_total:.2f} | {"Yes" if r.prospective_at_generation else "No/unknown"} |')
    lines+=['','Missing completed game boxes (if any):']+[f'- {m["game_date"]}: {m["away_display_name"]} at {m["home_display_name"]} (ID {m["game_id"]})' for m in missing]
    if not missing: lines.append('- None')
    lines+=['','Source: SportsDataverse ESPN men’s college basketball public team box scores and schedules.','Full validation and CSVs: [dated directory](./forecasts/'+str(d)+').','No injury adjustments, market lines, or ATS/total probabilities have been applied.','Check this report date and latest workflow status before use; a failed refresh leaves the last valid report unchanged.']
    body='\n'.join(lines)+'\n'; (dest/'report.md').write_text(body); Path('LATEST.md').write_text(body); print(f'Validated {len(power)} scoped forecasts for {d}; coverage {coverage:.2%}')
if __name__=='__main__': main()
