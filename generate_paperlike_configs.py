from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from generate_configs import generate_paperlike_configs


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper-like reproduction configs")
    parser.add_argument("--out", type=str, default="configs/paperlike")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing *.yaml in output dir before generation",
    )
    parser.add_argument(
        "--pool-num-workers",
        type=int,
        default=8,
        help="Default workers for loaders used by attack/pool configs",
    )
    parser.add_argument(
        "--substitute-num-workers",
        type=int,
        default=4,
        help="Default substitute DataLoader workers (substitute.num_workers)",
    )
    parser.add_argument(
        "--substitute-train-num-workers",
        type=int,
        default=None,
        help="Override substitute train loader workers (defaults to --substitute-num-workers)",
    )
    parser.add_argument(
        "--substitute-val-num-workers",
        type=int,
        default=None,
        help="Override substitute val loader workers (defaults to --substitute-num-workers)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out)
    count = generate_paperlike_configs(
        out_dir=out_dir,
        device=str(args.device),
        seeds=list(args.seeds),
        clean=(not args.no_clean),
        pool_num_workers=int(args.pool_num_workers),
        substitute_num_workers=int(args.substitute_num_workers),
        substitute_train_num_workers=(
            int(args.substitute_train_num_workers)
            if args.substitute_train_num_workers is not None
            else None
        ),
        substitute_val_num_workers=(
            int(args.substitute_val_num_workers)
            if args.substitute_val_num_workers is not None
            else None
        ),
    )
    print(f"Generated {count} paper-like configs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
