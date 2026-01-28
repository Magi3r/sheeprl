#!/usr/bin/env python3
"""Remove log folders whose checkpoints are missing or below a minimum step.

Usage:
  python remove_test_logs.py --min-step 10000 --dry-run

Scans `./logs/runs/dem/*/*/version_*/checkpoint` and removes the run folder
(`DATE_NAME` directory) when no checkpoint is found or the largest step
number found in the checkpoint files is less than `--min-step`.
"""
import argparse
import re
import shutil
from pathlib import Path
import sys


def extract_step_from_filename(name: str):
    """Return the largest integer found in the filename, or None."""
    nums = re.findall(r"(\d+)", name)
    if not nums:
        return None
    return max(int(n) for n in nums)


def parse_tf_checkpoint_file(path: Path):
    """Parse a TF 'checkpoint' text file to extract a step number if present."""
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    # look for patterns like model_checkpoint_path: "...ckpt-12345" or ckpt-12345
    m = re.search(r"ckpt[-_]?(\d+)", text)
    if m:
        return int(m.group(1))
    # fallback: any integer in file
    nums = re.findall(r"(\d+)", text)
    if nums:
        return max(int(n) for n in nums)
    return None


def find_max_step_in_checkpoint_dir(ckpt_dir: Path):
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None
    max_step = None
    for p in ckpt_dir.iterdir():
        if p.is_file():
            # if there's a TF 'checkpoint' file, parse it specially
            if p.name == "checkpoint":
                s = parse_tf_checkpoint_file(p)
                if s is not None:
                    max_step = s if max_step is None else max(max_step, s)
                continue
            s = extract_step_from_filename(p.name)
            if s is not None:
                max_step = s if max_step is None else max(max_step, s)
    return max_step


def scan_and_clean(base: Path, min_step: int, dry_run: bool = True, verbose: bool = False):
    """Scan the `base` directory for runs and remove folders where checkpoint missing or < min_step.

    base expected layout: base / GAME_NAME / DATE_NAME / version_x / checkpoint
    We remove the DATE_NAME directory when condition met.
    """
    base = base.resolve()
    if not base.exists():
        print(f"Base path {base} does not exist", file=sys.stderr)
        return

    for game_dir in sorted(base.iterdir()):
        if not game_dir.is_dir():
            continue
        for date_dir in sorted(game_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            # find version_* dirs inside date_dir
            version_dirs = [d for d in date_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]
            # if no version dirs, consider deleting
            if not version_dirs:
                reason = "no version_* directories"
                target = date_dir
                if verbose or dry_run:
                    print(f"Would remove {target} ({reason})")
                if not dry_run:
                    shutil.rmtree(target)
                continue

            # examine each version dir's checkpoint subdir
            max_step_for_run = None
            for v in version_dirs:
                ckpt_dir = v / "checkpoint"
                s = find_max_step_in_checkpoint_dir(ckpt_dir)
                if s is not None:
                    max_step_for_run = s if max_step_for_run is None else max(max_step_for_run, s)

            if max_step_for_run is None:
                reason = "no checkpoint files found"
                target = date_dir
                if verbose or dry_run:
                    print(f"Would remove {target} ({reason})")
                if not dry_run:
                    shutil.rmtree(target)
                continue

            if max_step_for_run < min_step:
                reason = f"max step {max_step_for_run} < {min_step}"
                target = date_dir
                if verbose or dry_run:
                    print(f"Would remove {target} ({reason})")
                if not dry_run:
                    shutil.rmtree(target)
            else:
                if verbose:
                    print(f"Keeping {date_dir} (max step {max_step_for_run})")


def main():
    p = argparse.ArgumentParser(description="Remove logs with missing or small checkpoints")
    p.add_argument("--base", default="logs/runs/dem", help="Base logs path to scan (default: logs/runs/dem)")
    p.add_argument("--min-step", type=int, default=10000, help="Minimum step to keep a run (default: 10000)")
    p.add_argument("--dry-run", action="store_true", default=False, help="Don't actually delete, just print actions")
    p.add_argument("--verbose", action="store_true", default=False, help="Verbose output")
    args = p.parse_args()

    base = Path(args.base)
    scan_and_clean(base=base, min_step=args.min_step, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
