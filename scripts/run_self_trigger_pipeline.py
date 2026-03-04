#!/usr/bin/env python3
"""
Self-trigger processing pipeline.

  Step 1 (calibrate) : calibrate_hits.py          -> calibrated_hits/
  Step 2 (dq)        : self_trigger_dq_flags.py   -> dq_flags/

Use --steps to control which steps are run (default: all).
Requesting calibrate automatically includes dq.
"""

import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_STEPS = {"calibrate", "dq"}


def expand_steps(requested):
    """Expand requested steps to include required downstream steps."""
    steps = set(requested)
    if "calibrate" in steps:
        steps.add("dq")
    return steps


def run_self_pipeline(input_files, run_number, output_base,
                      steps_to_run=None, debug=False, not_official_const=False):
    """
    Run the self-trigger pipeline for a single run.

    Parameters
    ----------
    input_files        : list of str  – raw WCTEReadoutWindows ROOT files
    run_number         : str          – run number
    output_base        : str          – base output directory; <run_number>/ subdir created automatically
    steps_to_run       : set or None  – steps to execute; None means all (calibrate, dq)
    debug              : bool
    not_official_const : bool
    """
    if steps_to_run is None:
        steps_to_run = set(ALL_STEPS)
    else:
        steps_to_run = expand_steps(steps_to_run)

    run_dir = os.path.join(output_base, str(run_number))
    cal_dir = os.path.join(run_dir, "calibrated_hits")
    dq_dir  = os.path.join(run_dir, "dq_flags")
    for d in [cal_dir, dq_dir]:
        os.makedirs(d, exist_ok=True)

    extra   = ["--debug"] if debug else []
    not_off = ["--not_official_const"] if not_official_const else []

    for f in input_files:
        if f"R{run_number}" not in os.path.basename(f):
            print(f"[ERROR] '{f}' does not match run number R{run_number}")
            sys.exit(1)

    failed = []

    for input_file in input_files:
        base     = os.path.splitext(os.path.basename(input_file))[0]
        cal_file = os.path.join(cal_dir, f"{base}_calibrated_hits.root")

        print(f"\n{'#'*60}\n  {os.path.basename(input_file)}\n{'#'*60}")

        # ── Step 1: Calibrate hits ────────────────────────────────────────
        if "calibrate" in steps_to_run:
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "calibrate_hits.py"),
                 "-i", input_file, "-r", str(run_number), "-o", cal_dir] + not_off + extra
            )
            if result.returncode != 0:
                print("[ERROR] Calibration failed — skipping remaining steps for this file")
                failed.append(input_file)
                continue
        else:
            if not os.path.exists(cal_file):
                print(f"[ERROR] Calibrated output not found (required for DQ):\n  {cal_file}")
                failed.append(input_file)
                continue
            print(f"[SKIP] calibrate — using {os.path.basename(cal_file)}")

        # ── Step 2: Data quality flags ────────────────────────────────────
        if "dq" not in steps_to_run:
            print("[SKIP] dq")
            continue

        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "self_trigger_dq_flags.py"),
             "-i", input_file,
             "-c", cal_dir,
             "-r", str(run_number),
             "-o", dq_dir] + extra
        )
        if result.returncode != 0:
            print("[ERROR] DQ flags failed for this file")
            failed.append(input_file)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total  = len(input_files)
    n_failed = len(failed)
    print(f"\n{'='*60}")
    print(f"  {n_total - n_failed}/{n_total} files completed successfully")
    if failed:
        for f in failed:
            print(f"  ✗  {os.path.basename(f)}")
        return False
    print("*** Self trigger pipeline complete ***")
    return True


def main():
    parser = argparse.ArgumentParser(description="Self-trigger pipeline: calibrate -> dq")
    parser.add_argument("-i", "--input_files", required=True, nargs="+",
                        help="Raw WCTEReadoutWindows ROOT input files")
    parser.add_argument("-r", "--run_number", required=True)
    parser.add_argument("-o", "--output_base", required=True,
                        help="Base output directory; <run_number>/ subdir is created automatically")
    parser.add_argument("--steps", nargs="*", choices=list(ALL_STEPS),
                        default=None, metavar="STEP",
                        help=f"Steps to run (default: all). Choices: {', '.join(sorted(ALL_STEPS))}. "
                             "calibrate automatically includes dq.")
    parser.add_argument("--not_official_const", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    steps = expand_steps(args.steps) if args.steps is not None else set(ALL_STEPS)

    success = run_self_pipeline(
        input_files=args.input_files,
        run_number=args.run_number,
        output_base=args.output_base,
        steps_to_run=steps,
        debug=args.debug,
        not_official_const=args.not_official_const,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
