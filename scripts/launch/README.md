# Launch Scripts

Official launcher entrypoints are centralized here.

- `run_matrix.sh` / `run_matrix.ps1`: generate matrix configs and execute benchmark runs.
- `run_smoke.sh`: run a small smoke subset.
- `run_tests.sh`: run core benchmark tests (`--full` for full suite).

Run launchers from the repository root, for example:

- `bash scripts/launch/run_smoke.sh cuda:0`
- `bash scripts/launch/run_matrix.sh`
- `bash scripts/launch/run_tests.sh`
