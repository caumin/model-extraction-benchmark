# Launch Scripts

Official launcher entrypoints are centralized here.

- `run_matrix.sh` / `run_matrix.ps1`: generate matrix configs and execute benchmark runs.
- `run_smoke.sh`: run a small smoke subset.
- `run_tests.sh`: run config-driven per-attack execution checks.

Root-level scripts are compatibility wrappers that delegate to this directory.

Note:

- Run launchers from the repository root, or use root-level wrappers (`run_*.sh`, `run_*.ps1`) which normalize execution context.
