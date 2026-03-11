"""
WCTE Beam Monitors PID (Particle Identification) Package

A modularized Python package for particle identification using the WCTE beam
monitoring detectors. This package provides utilities for reading detector data,
calibrating responses, identifying particles using Time-of-Flight and ACT charge
measurements, and estimating particle momenta.

Modules:
    constants: Detector configuration constants and physical parameters
    file_utils: File I/O and staging utilities for accessing data from EOS
    flag_utils: Event quality bitmask encoding/decoding utilities
    detector_utils: TDC hit processing and detector requirement checking
    fitting: Curve fitting functions for detector response analysis
    beam_analysis: Main BeamAnalysis class for particle identification

Example:
    >>> from beam_monitors_pid import BeamAnalysis
    >>> from beam_monitors_pid.constants import particle_masses
    >>> 
    >>> analysis = BeamAnalysis(
    ...     run_number=12345,
    ...     run_momentum=1000,
    ...     n_eveto=1.03,
    ...     n_tagger=1.06,
    ...     there_is_ACT5=True,
    ...     output_dir="./output"
    ... )
    >>> analysis.open_file(input_file="data.root")
    >>> analysis.adjust_1pe_calibration()
    >>> analysis.tag_electrons_ACT02()
    >>> analysis.tag_protons_TOF()
    >>> analysis.end_analysis()
"""

__version__ = "0.1.0"
__author__ = "WCTE Collaboration"
__all__ = [
    "BeamAnalysis",
    "constants",
    "file_utils",
    "flag_utils",
    "detector_utils",
    "fitting",
]

# Import main class and submodules
try:
    from beam_monitors_pid.beam_analysis import BeamAnalysis
except ImportError:
    # Graceful fallback if BeamAnalysis not yet created
    BeamAnalysis = None

# Import utility modules
from beam_monitors_pid import constants
from beam_monitors_pid import file_utils
from beam_monitors_pid import flag_utils
from beam_monitors_pid import detector_utils
from beam_monitors_pid import fitting
