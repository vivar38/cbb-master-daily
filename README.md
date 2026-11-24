# CBB Master v1.0 — automated fitted baseline

This is the recovered fixed Ridge/Elo engine. It updates team states using current-season completed games, not the model coefficients. It downloads only public SportsDataverse ESPN team box-score and schedule data. No credentials, personal records, betting ledger, or prior scouting reports are needed. The fitted model file is included; only use this trusted copy.

**Output:** [LATEST.md](LATEST.md) (once a workflow succeeds) and dated files under `forecasts/`. The outputs contain the statistical baseline, not the full 16-point matchup audit or market-confidence rankings. Dates follow ESPN's Eastern calendar, tip times Central. A new dated directory is published once; failed refreshes never replace a valid report. No daily upload is required once cloud execution and ChatGPT retrieval are verified.

**Data source:** https://github.com/sportsdataverse/sportsdataverse-data/releases . This project does not redistribute bulk source files. Original fixed training 2018–24; historical holdout 2026; generic five-game shrinkage prior retained. Power-conference scope: ACC, Big East, Big Ten, Big 12, SEC, including Illinois schools.

**Operations:** Actions → CBB daily fitted baseline → Run workflow for manual recovery. The scheduled attempts occur at 12:17 and 14:17 UTC (6:17/8:17 AM CST, 7:17/9:17 AM CDT); the second attempts only if a dated report is absent. Delays and source failures can occur. Verify report date and validation before using it. Workflow currently targets 2027 season (through April 2027); update season configuration next year. GitHub-hosted runner, no personal access token. The workflow uses its built-in GITHUB_TOKEN with repository content write access only.
