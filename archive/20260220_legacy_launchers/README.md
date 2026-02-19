# 20260220_legacy_launchers

Legacy launcher scripts moved from repository root during cleanup.

Rationale:
- Root contained many overlapping run scripts.
- Official entrypoints were consolidated under `scripts/launch/`.
- Historical/variant launchers are preserved here for reproducibility.

Backward compatibility:
- Root-level wrapper scripts with the original names were kept and now delegate to these archived scripts.
