#!/usr/bin/env python3
"""Beam particle identification workflow using the WCTE beam monitoring detectors.

This script provides a template for performing particle identification (PID) using
the beam monitors. It is intended as an example for building optimized analyses.

The script uses the refactored beam_monitors_pid package which provides:
    - BeamAnalysis: Main class orchestrating the PID workflow
    - Constants: Detector configurations and physical parameters
    - Utilities: File I/O, flag management, curve fitting, etc.

For more details on the modular package, see:
    analysis_tools/beam_monitors_pid/README.md
    analysis_tools/beam_monitors_pid/MIGRATION.md
"""

import numpy as np
import sys
import argparse
import os

# Path to analysis_tools - change as needed for your environment
sys.path.append("../")

# Import main components from the refactored beam_monitors_pid package
from analysis_tools.beam_monitors_pid import BeamAnalysis
# Optional: import utilities for reference or custom analysis
# from analysis_tools.beam_monitors_pid import constants, file_utils, flag_utils, detector_utils, fitting

# Import general analysis tools
from analysis_tools import ReadBeamRunInfo

# Number of events to read in debug mode
DEBUG_N_EVENTS = 5000


def parse_args():
    """Parse command-line arguments for beam analysis configuration."""
    parser = argparse.ArgumentParser(
        description="WCTE beam particle identification analysis"
    )

    parser.add_argument(
        "-r", "--run_number", required=True, type=int,
        help="Run number to analyse"
    )
    parser.add_argument(
        "-i", "--input_files", required=True, nargs='+',
        help="Path(s) to WCTEReadoutWindows ROOT file(s)"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory to write output ROOT file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=f"Enable debug mode: limits to {DEBUG_N_EVENTS} events"
    )

    return parser.parse_args()


def main():
    """Execute the beam analysis workflow."""
    
    # Parse command-line arguments
    args = parse_args()

    # Read run information from the JSON database
    run_info = ReadBeamRunInfo()
    run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5, beam_config = \
        run_info.get_info_run_number(args.run_number)
    run_info.print_run_summary(there_is_ACT5)

    # Set number of events to read (use DEBUG_N_EVENTS if debug mode enabled)
    n_events = -1  # -1 means read all events
    if args.debug and n_events == -1:
        print(f"Debug mode: limiting to {DEBUG_N_EVENTS} events")
        n_events = DEBUG_N_EVENTS

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each input file
    for input_file in args.input_files:

        # Validate that input file matches run number
        if f"R{args.run_number}" not in os.path.basename(input_file):
            print(f"[ERROR] '{input_file}' does not match run number R{args.run_number}")
            sys.exit(1)

        # Generate output filenames
        base = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = os.path.join(args.output_dir, f"{base}_beam_analysis.root")
        pdf_name = f"{base}_PID.pdf"

        print(f"\n{'#'*60}\n  {os.path.basename(input_file)}\n{'#'*60}")

        # ===== BEAM ANALYSIS WORKFLOW =====
        
        # Initialize the BeamAnalysis object with run configuration
        ana = BeamAnalysis(
            run_number, 
            run_momentum, 
            n_eveto_group, 
            n_tagger_group, 
            there_is_ACT5, 
            args.output_dir, 
            pdf_name
        )

        # Load data: set require_t5=False if particles need not reach T5
        ana.open_file(n_events, require_t5=True, input_file=input_file)

        # Step 1: Calibrate 1-photoelectron response
        ana.adjust_1pe_calibration()

        # Step 2: Tag protons using T0-T1 Time-of-Flight
        # NOTE: Protons must be tagged first to avoid double-counting
        ana.tag_protons_TOF()
        # TODO: identify protons that produce knock-on electrons

        # Step 3: Tag electrons using ACT0-2 charge measurement
        ana.tag_electrons_ACT02()

        # Step 4: Visual validation of electron/proton removal in ACT3-5
        ana.plot_ACT35_left_vs_right()

        # Step 5: Separate muons and pions using ACT3-5
        # Uses muon tagger when ≥0.5% of muons/pions are above cut line
        ana.tag_muons_pions_ACT35()

        # Step 6: Correct Time-of-Flight offset (cable length, etc.)
        # Essential for accurate momentum estimation
        ana.measure_particle_TOF()

        # Step 7: Estimate particle momentum for each type and trigger
        # Error on TOF is estimated from TS0-TS1 resolution (electron TOF Gaussian fit)
        ana.estimate_particle_momentum()

        # Step 8: Calculate number of particles per POT
        ana.plot_number_particles_per_POT()

        # Step 9: Finalize analysis and prepare for output
        ana.end_analysis()

        # Output results to ROOT file
        ana.output_to_root(output_filename)
        print(f"Output written to: {output_filename}")


if __name__ == "__main__":
    main()
