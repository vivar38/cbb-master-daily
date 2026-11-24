"""No-login NCAA Division I men's historical data loader and baseline adapter.

Public release assets (CSV) from sportsdataverse/sportsdataverse-data.
Exports a file with Kaggle *column names* for compatibility with cbb_model.py,
but this is independently sourced ESPN data, NOT a copy of Kaggle data.

Usage:
 python fetch_sportsdataverse.py --start 2018 --end 2026 --roster-years 2025 2026 2027
 python cbb_model.py --games data/raw/MRegularSeasonDetailedResults.csv --out outputs

No account, API key, or registration is required. Network access required.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
TAGS = {
    'box': 'espn_mens_college_basketball_team_boxscores',
    'schedule': 'espn_mens_college_basketball_schedules',
    'roster': 'espn_mens_college_basketball_rosters',
}
NAMES = {'box': 'team_box_{year}.csv', 'schedule': 'mbb_schedule_{year}.csv', 'roster': 'rosters_{year}.csv'}
STATS = {
    'FGM':'field_goals_made', 'FGA':'field_goals_attempted',
    'FGM3':'three_point_field_goals_made', 'FGA3':'three_point_field_goals_attempted',
    'FTM':'free_throws_made', 'FTA':'free_throws_attempted',
    'OR':'offensive_rebounds', 'DR':'defensive_rebounds', 'TO':'turnovers',
}
REQUIRED_BOX = {'game_id','season','season_type','game_date','team_id','team_score',
                'team_home_away', *STATS.values()}
REQUIRED_SCHEDULE = {'game_id','neutral_site','status_period','status_type_completed'}
LOG = logging.getLogger('sportsdataverse')


def source_url(kind: str, year: int) -> str:
    if kind not in TAGS or not (2000 <= year <= 2050):
        raise ValueError('Invalid dataset or year')
    return f'{ROOT}/{TAGS[kind]}/{NAMES[kind].format(year=year)}'


def fetch_public_asset(kind: str, year: int, folder: Path, attempts=3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / NAMES[kind].format(year=year)
    if dest.exists() and dest.stat().st_size > 100:
        LOG.info('Already downloaded: %s (%s bytes)', dest.name, dest.stat().st_size)
        return dest
    url = source_url(kind, year)
    LOG.info('Downloading %s from public release', dest.name)
    error = None
    for attempt in range(attempts):
        try:
            req = Request(url, headers={'User-Agent':'CBB-Research-Data-Client/1.0',
                                        'Accept':'text/csv,application/octet-stream'})
            with urlopen(req, timeout=90) as r, dest.with_suffix('.download').open('wb') as out:
                import shutil
                shutil.copyfileobj(r, out, length=1024 * 1024)
            temp = dest.with_suffix('.download')
            if temp.stat().st_size <= 100:
                raise ValueError(f'Unexpectedly small file from {url}')
            with temp.open('rb') as f:
                first = f.read(250)
            if first.lstrip().startswith((b'<html',b'<!DOCTYPE',b'{"message"')) or b'game_id' not in first and kind != 'roster':
                # CSV headers for schedules/box generally contain game_id early, but allow >250-byte headers.
                with temp.open('rb') as f: header = f.readline()[:6000]
                if b'game_id' not in header:
                    raise ValueError(f'Download did not look like expected CSV: {url}')
            temp.replace(dest)
            return dest
        except (OSError, HTTPError, URLError, ValueError) as ex:
            error = ex
            dest.with_suffix('.download').unlink(missing_ok=True)
            if attempt < attempts-1:
                time.sleep(2**attempt)
    raise RuntimeError(f'Could not download {url}: {error}')


def _bool_series(s: pd.Series) -> pd.Series:
    # Strict conversion: a missing/unknown 'neutral' flag must never silently mean 'not neutral'.
    d = {'true':True,'false':False,'1':True,'0':False,'yes':True,'no':False}
    result = s.astype('string').str.strip().str.lower().map(d)
    return result.astype('boolean')


def resolve_same_day_collisions(out: pd.DataFrame, year: int, audit_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Deduplicate or quarantine ambiguous ESPN game IDs without hiding the data issue.

    A single logical NCAA matchup should have one winner/loser box score per
    recorded calendar date. Multiple ESPN event IDs for a matchup may represent
    duplicate records or (rarely) a genuine doubleheader. We cannot safely
    distinguish two *conflicting* records with available fields, so exclude all
    conflicting versions pending manual review; never arbitrarily select one.

    Truly identical score/box/venue/OT rows are collapsed to one match, keeping
    the lowest source game ID. Every collision is exported for audit.
    """
    if out.empty:
        return out, {'duplicate_fixture_groups': 0, 'identical_duplicates_removed': 0,
                     'conflicting_games_quarantined': 0}
    out = out.copy()
    out['_team_low'] = out[['WTeamID','LTeamID']].min(axis=1)
    out['_team_high'] = out[['WTeamID','LTeamID']].max(axis=1)
    key = ['Season', 'DayNum', '_team_low', '_team_high']
    suspect = out.duplicated(key, keep=False)
    collisions = out.loc[suspect].sort_values(key + ['SourceGameID']).copy()
    keep_idx = set(out.index[~suspect].tolist())
    groups = 0
    identical_removed = 0
    ambiguous_removed = 0
    audit_frames = []
    compare_cols = [c for c in out.columns if c not in {'SourceGameID', '_team_low', '_team_high'}]
    for _, g in collisions.groupby(key, dropna=False, sort=True):
        groups += 1
        fingerprint_count = len(g[compare_cols].drop_duplicates())
        if fingerprint_count == 1:
            keep = int(g.index[0])
            keep_idx.add(keep)
            identical_removed += len(g) - 1
            labels = ['kept_identical_source_game'] + ['removed_identical_duplicate'] * (len(g) - 1)
        else:
            ambiguous_removed += len(g)
            labels = ['quarantined_conflicting_source_game'] * len(g)
        audit = g.copy()
        audit['audit_resolution'] = labels
        audit_frames.append(audit)
    if audit_frames and audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f'duplicate_candidates_{year}.csv'
        pd.concat(audit_frames, ignore_index=True).to_csv(audit_path, index=False)
        LOG.warning('Season %s: %s same-day matchup collision group(s); '
                    '%s identical entries collapsed, %s conflicting entries quarantined. '
                    'Audit: %s', year, groups, identical_removed, ambiguous_removed, audit_path)
    elif groups:
        LOG.warning('Season %s: %s duplicated matchup group(s); '
                    '%s identical entries collapsed, %s conflicting entries quarantined.',
                    year, groups, identical_removed, ambiguous_removed)
    result = out.loc[sorted(keep_idx)].drop(columns=['_team_low','_team_high']).reset_index(drop=True)
    if result.empty:
        raise ValueError(f'Season {year} has no unambiguous games after fixture collision review')
    if result.assign(team_low=result[['WTeamID','LTeamID']].min(axis=1),
                     team_high=result[['WTeamID','LTeamID']].max(axis=1)) \
             .duplicated(['Season','DayNum','team_low','team_high']).any():
        raise AssertionError(f'Unresolved fixture collision in {year}')
    return result, {'duplicate_fixture_groups': groups,
                    'identical_duplicates_removed': int(identical_removed),
                    'conflicting_games_quarantined': int(ambiguous_removed)}


def convert_one_year(box: pd.DataFrame, schedules: pd.DataFrame, year: int,
                     audit_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    missing_b = sorted(REQUIRED_BOX-set(box.columns))
    missing_s = sorted(REQUIRED_SCHEDULE-set(schedules.columns))
    if missing_b or missing_s:
        raise ValueError(f'Changed upstream schema; missing box={missing_b}, schedules={missing_s}')
    b=box.loc[pd.to_numeric(box['season'],errors='coerce').eq(year)].copy()
    b=b.loc[pd.to_numeric(b['season_type'],errors='coerce').eq(2)].copy()  # regular season
    s=schedules.copy()
    if 'season' in s:
        s=s.loc[pd.to_numeric(s['season'],errors='coerce').eq(year)].copy()
    if 'season_type' in s:
        s=s.loc[pd.to_numeric(s['season_type'],errors='coerce').eq(2)].copy()
    if not len(b) or not len(s):
        raise ValueError(f'No regular-season data for {year}; verify source and season_type=2')
    for col in ['game_id','team_id','team_score',*STATS.values()]:
        b[col]=pd.to_numeric(b[col],errors='coerce')
    s['game_id']=pd.to_numeric(s['game_id'],errors='coerce')
    s['neutral_clean']=_bool_series(s['neutral_site'])
    s['complete_clean']=_bool_series(s['status_type_completed'])
    s['period_clean']=pd.to_numeric(s['status_period'],errors='coerce')
    s = s.sort_values('game_id').drop_duplicates('game_id')
    # Only completed games with known venue classification are allowed.
    s=s.loc[s['complete_clean'].eq(True) & s['neutral_clean'].notna()]
    merged=b.merge(s[['game_id','neutral_clean','period_clean']],on='game_id',how='inner',validate='many_to_one')
    merged['game_date']=pd.to_datetime(merged['game_date'],errors='coerce')
    numeric=['team_id','team_score',*STATS.values()]
    merged=merged.dropna(subset=numeric+['game_date'])
    merged['home_away_clean']=merged['team_home_away'].astype(str).str.lower().str.strip()
    merged=merged.loc[merged.home_away_clean.isin(['home','away'])]
    counts=merged.groupby('game_id').size()
    valid=counts.loc[counts.eq(2)].index
    merged=merged.loc[merged.game_id.isin(valid)].sort_values(['game_id','team_id']).copy()
    # Do not synthesize or average away malformed scores/box results.
    rows=[]; invalid=0; missing_period=0
    eligible_box_game_count = int(b['game_id'].nunique())
    paired_game_count = int(merged['game_id'].nunique())
    for gid, pair in merged.groupby('game_id',sort=True):
        a,b=pair.iloc[0],pair.iloc[1]
        if int(a.team_id)==int(b.team_id) or a.team_score==b.team_score or a.game_date.date()!=b.game_date.date():
            invalid+=1;continue
        if a.home_away_clean == b.home_away_clean:
            invalid+=1;continue
        winner, loser=(a,b) if a.team_score>b.team_score else (b,a)
        neutral=bool(winner.neutral_clean)
        wloc='N' if neutral else ('H' if winner.home_away_clean=='home' else 'A')
        period=a.period_clean
        if pd.isna(period):
            missing_period+=1
            # Do not quietly claim zero overtime when we don't know it.
            invalid+=1;continue
        num_ot=max(int(period)-2,0)  # NCAA men's regulation: two 20-minute halves.
        day_num=(a.game_date.date()-pd.Timestamp(year=year-1,month=10,day=1).date()).days
        if day_num < 0 or day_num>240:
            invalid+=1;continue
        obj={'Season':year,'DayNum':day_num,'SourceGameID':int(gid),'WTeamID':int(winner.team_id),
             'WScore':int(winner.team_score),'LTeamID':int(loser.team_id),
             'LScore':int(loser.team_score),'WLoc':wloc,'NumOT':num_ot}
        for short,longname in STATS.items():
            obj[f'W{short}']=int(winner[longname]);obj[f'L{short}']=int(loser[longname])
        rows.append(obj)
    out=pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f'No clean games converted for {year}')
    out, collision_audit = resolve_same_day_collisions(out, year, audit_dir)
    meta={'season':year,'team_box_rows':int(len(box)),'schedule_rows':int(len(schedules)),
          'completed_games_converted':int(len(out)),'unpaired_or_missing_data_games':int(eligible_box_game_count-paired_game_count),
          'malformed_or_unknown_period_games':int(invalid),'unknown_period_games':int(missing_period),
          'min_day_num':int(out.DayNum.min()),'max_day_num':int(out.DayNum.max()),
          'teams':int(pd.concat([out.WTeamID,out.LTeamID]).nunique()),
          **collision_audit}
    return out,meta


def stage_seasons(raw_folder:Path, start:int=2018, end:int=2026, roster_years=(2025,2026,2027), download=True):
    if not (2005<=start<=end<=2027):
        raise ValueError('Valid team box score data starts in earnest in 2005; use 2005–2027.')
    source=raw_folder/'sportsdataverse'
    parts=[];meta=[]
    for year in range(start,end+1):
        boxpath=(fetch_public_asset('box',year,source) if download else source/NAMES['box'].format(year=year))
        schedpath=(fetch_public_asset('schedule',year,source) if download else source/NAMES['schedule'].format(year=year))
        part, info=convert_one_year(pd.read_csv(boxpath,low_memory=False),pd.read_csv(schedpath,low_memory=False),
                                    year, audit_dir=source)
        parts.append(part);meta.append(info)
        LOG.info('Season %d: %d usable completed games',year,len(part))
    for year in roster_years:
        (fetch_public_asset('roster',year,source) if download else source/NAMES['roster'].format(year=year))
    allgames=pd.concat(parts,ignore_index=True).sort_values(['Season','DayNum','WTeamID','LTeamID'])
    target=raw_folder/'MRegularSeasonDetailedResults.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    allgames.to_csv(target,index=False)
    manifest={'source':'SportsDataverse ESPN team boxscores + schedules, NOT Kaggle',
              'source_release':'https://github.com/sportsdataverse/sportsdataverse-data/releases',
              'source_policy':'Public direct-download GitHub release assets; no account required',
              'generated_file':str(target),'seasons':meta,'total_games':len(allgames),
              'notes':['Regular season season_type=2; may omit postseason tournament games',
                       'DayNum relative to October 1 previous year, not Kaggle DayZero',
                       'ESPN IDs must NOT be joined to Kaggle IDs without explicit crosswalk',
                       'No historical betting odds or direct proof of betting profitability',
                       'Games with missing box figures, unknown neutral site, or period are excluded',
                       'Same-day matchup collisions are audited: identical records deduplicated, conflicting records quarantined',
                       'Duplicate candidates per season are saved as sportsdataverse/duplicate_candidates_{year}.csv']}
    (raw_folder/'sportsdataverse_provenance.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=2018)
    parser.add_argument('--end',type=int,default=2026)
    parser.add_argument('--dest',type=Path,default=Path('data/raw'))
    parser.add_argument('--roster-years',type=int,nargs='*',default=[2025,2026,2027])
    parser.add_argument('--offline',action='store_true',help='Convert existing downloaded files; do not fetch')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    try:
        result=stage_seasons(args.dest,args.start,args.end,args.roster_years,not args.offline)
    except Exception as e:
        parser.exit(1,f'Failed: {e}\n')
    print(json.dumps(result,indent=2))
    print('Next: python cbb_model.py --games data/raw/MRegularSeasonDetailedResults.csv --out outputs')

if __name__=='__main__':main()
