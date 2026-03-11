"""
Detector configuration constants and particle properties for WCTE beam analysis.

This module contains all detector-related constants, channel/group definitions,
detector distances, and particle properties used in beam monitoring and PID.
"""

# ============================================================================
# Physical Constants
# ============================================================================
c = 0.299792458  # Speed of light in m/ns (do not change units - used throughout)

# ============================================================================
# Detector Distances (in meters)
# ============================================================================
L = 444.03  # Total distance from T0 to T1
L_t0t4 = 305.68  # Distance from T0 to T4
L_t4t1 = 143.38  # Distance from T4 to T1

# ============================================================================
# Particle Properties
# ============================================================================
# Particle masses in GeV/c^2
particle_masses = {
    "Electrons": 0.000511,
    "Muons": 0.105658,
    "Pions": 0.13957,
    "Protons": 0.938272,
}

# TOF (Time of Flight) cuts in nanoseconds
# These are replaced with more accurate estimates using the travel time of all particles
helium3_tof_cut = 30  # ad-hoc
tritium_tof_cut = 80  # ad-hoc
lithium6_tof_cut = 90  # ad-hoc

# ============================================================================
# Detector Channel Assignments and Groups
# ============================================================================

# TDC Reference channels
reference_ids = (31, 46)  # (TDC ref for IDs <31, ref for IDs >31)

# T0 detector channels
t0_group = [0, 1, 2, 3]  # must all be present

# T1 detector channels
t1_group = [4, 5, 6, 7]  # must all be present

# T4 detector channels and QDC cuts
t4_group = [42, 43]  # must all be present
t4_qdc_cut = 200  # Only hits above this value in QDC

# ============================================================================
# ACT (Anti-Coincidence Trigger) Groups
# ============================================================================

# ACT detector left/right channel pairs
ACT0_group = (12, 13)
ACT1_group = (14, 15)
ACT2_group = (16, 17)
ACT3_group = (18, 19)
ACT4_group = (20, 21)
ACT5_group = (22, 23)

# ACT eveto channels (veto channels for identifying electrons)
act_eveto_group = [12, 13, 14, 15, 16, 17]

# ACT tagger channels (for tagging)
act_tagger_group = [18, 19, 20, 21, 22, 23]

# ============================================================================
# HC (Hadronic Calorimeter) Group
# ============================================================================
hc_group = [9, 10]
hc_charge_cut = 150  # Charge threshold for HC hit

# ============================================================================
# T5 (TOF) Detector Groups
# ============================================================================
# T5 is the TOF detector with 8 bars, each with a pair of SiPMs
# Both SiPMs need to be above threshold for at least one bar for event to be kept

t5_b0_group = [48, 56]  # Bar 0 (SiPMs on either side)
t5_b1_group = [49, 57]  # Bar 1
t5_b2_group = [50, 58]  # Bar 2
t5_b3_group = [51, 59]  # Bar 3
t5_b4_group = [52, 60]  # Bar 4
t5_b5_group = [53, 61]  # Bar 5
t5_b6_group = [54, 62]  # Bar 6
t5_b7_group = [55, 63]  # Bar 7

t5_total_group = [
    t5_b0_group,
    t5_b1_group,
    t5_b2_group,
    t5_b3_group,
    t5_b4_group,
    t5_b5_group,
    t5_b6_group,
    t5_b7_group,
]

# ============================================================================
# Channel Name Mapping
# ============================================================================
CHANNEL_MAPPING = {
    12: "ACT0-L",
    13: "ACT0-R",
    14: "ACT1-L",
    15: "ACT1-R",
    16: "ACT2-L",
    17: "ACT2-R",
    18: "ACT3-L",
    19: "ACT3-R",
    20: "ACT4-L",
    21: "ACT4-R",
    22: "ACT5-L",
    23: "ACT5-R",
}
