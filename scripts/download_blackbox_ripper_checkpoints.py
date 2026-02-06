import argparse
import sys
from pathlib import Path

# Add project root to path for consistent relative paths.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


GOOGLE_DRIVE_FILE_ID = "1S_tI80dECZ1UC5FEktpBOGwinL4-0fRU"  # models.zip in upstream script


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download official Black-Box Ripper pretrained checkpoints (from upstream Google Drive)."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory (default: checkpoints/blackbox_ripper/official). "
            "Note: repository .gitignore ignores *.pth/*.pt by default."
        ),
    )
    parser.add_argument(
        "--file-id",
        type=str,
        default=GOOGLE_DRIVE_FILE_ID,
        help="Google Drive file id (default matches temp_ripper/download_checkpoints.sh)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (project_root / "checkpoints" / "blackbox_ripper" / "official")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "models.zip"

    try:
        import gdown  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency: gdown\n\n"
            "Install it with:\n"
            "  python -m pip install gdown\n\n"
            "Then re-run:\n"
            f"  python {Path(__file__).name}\n"
        )

    url = f"https://drive.google.com/uc?id={args.file_id}"
    print(f"Downloading {url} -> {zip_path}")
    gdown.download(url, str(zip_path), quiet=False)

    # Unzip into out_dir.
    import zipfile

    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(out_dir))

    print("Done.")
    print("Example config usage:")
    print(
        "  attack:\n"
        "    name: blackbox_ripper\n"
        "    output_mode: soft_prob\n"
        "    generator_name: cifar_sngan\n"
        f"    generator_checkpoint: {out_dir.as_posix()}/<pick-a-weights-file>.pth\n"
    )


if __name__ == "__main__":
    main()
