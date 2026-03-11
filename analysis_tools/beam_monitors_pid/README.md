# WCTE Beam Monitors PID - Module Refactoring Guide

## Overview

The original `beam_monitors_pid.py` file (~4,900 lines) has been refactored into a modular, maintainable package structure. Each module focuses on a specific functional area, making the codebase easier to understand, test, and extend.

Run information used by `ReadBeamRunInfo` is now expected to live inside the
package under `beam_monitors_pid/data/google_sheet_beam_data.json`.  A
placeholder file is provided and the loader will fall back to the legacy EOS
location for backward compatibility.

## Package Structure

```
beam_monitors_pid/
├── __init__.py                 # Package initialization
├── constants.py                # Detector configurations and constants
├── file_utils.py              # File I/O and staging utilities
├── flag_utils.py              # Event quality bitmask utilities
├── detector_utils.py          # TDC processing and detector utilities
├── fitting.py                 # Fitting functions and curve definitions
└── beam_analysis.py           # Main BeamAnalysis class (to be created)
```

## Module Descriptions

### 1. **constants.py**
Centralized location for all constant data and detector configurations.

**Contains:**
- Physical constants (speed of light)
- Detector distances (L, L_t0t4, L_t4t1)
- Particle masses
- TOF cut values  
- TDC channel group definitions (T0, T1, T4, T5, ACT groups)
- QDC and HC channel assignments
- Channel name mappings

**Usage:**
```python
from beam_monitors_pid.constants import t0_group, t1_group, particle_masses
```

### 2. **file_utils.py**
File I/O operations for staging data from CERN's EOS storage.

**Functions:**
- `stage_local()` - Stage files from EOS to local /tmp with space checks
- `to_xrootd()` - Convert EOS paths to XRootD URLs
- `make_blocks()` - Partition indices into contiguous blocks for block-wise reading

**Usage:**
```python
from beam_monitors_pid.file_utils import stage_local, make_blocks

local_path = stage_local(eos_path, min_free_gb=20)
blocks = make_blocks(indices, max_block=500)
```

### 3. **flag_utils.py**
Bitmask-based event quality flag management.

**Functions:**
- `make_flag_map()` - Create deterministic flag-to-bit mapping
- `write_event_quality_mask()` - Pack flags into integer bitmask
- `read_event_quality_mask()` - Unpack bitmask back into flag dictionary

**Purpose:** Efficiently store multiple event quality flags in a single integer.

**Usage:**
```python
from beam_monitors_pid.flag_utils import write_event_quality_mask, read_event_quality_mask

flags = {
    "event_q_t0_or_t1_missing_tdc": False,
    "event_q_t4_missing_tdc": True,
    "event_q_t5_missing_tdc": False,
}
flag_map = {
    "event_q_t0_or_t1_missing_tdc": 0,
    "event_q_t4_missing_tdc": 1,
    "event_q_t5_missing_tdc": 2,
}
bitmask = write_event_quality_mask(flags, flag_map)
restored_flags = read_event_quality_mask(bitmask, flag_map)
```

### 4. **detector_utils.py**
TDC (Time-to-Digital Converter) hit processing utilities.

**Functions:**
- `deduplicate_tdc_hits()` - Remove duplicate TDC hits per channel
- `tdc_requirement_met()` - Check if detector groups meet presence requirements

**Purpose:** Handle cases where TDC channels record multiple hits and validate detector requirements.

**Usage:**
```python
from beam_monitors_pid.detector_utils import deduplicate_tdc_hits, NotTheFirstHit

ids, times, duplicates_removed = deduplicate_tdc_hits(tdc_ids, tdc_times)

result = tdc_requirement_met(
    {"mode": "all", "channels": [0, 1, 2, 3]},
    {0, 1, 2, 3}  # tdc_set
)
```

### 5. **fitting.py**
Curve fitting functions for detector response analysis.

**Curves:**
- `gaussian()` - Single Gaussian distribution
- `three_gaussians()` - Sum of three Gaussians (for multi-peak fitting)
- `landau_gauss_convolution()` - Convolution of Landau and Gaussian

**Fitting Functions:**
- `fit_gaussian()` - Fit single Gaussian with auto-guess
- `fit_three_gaussians()` - Fit three Gaussians with auto-guess

**Usage:**
```python
from beam_monitors_pid.fitting import fit_gaussian, gaussian
import numpy as np

bin_centers, entries = np.array([...]), np.array([...])
popt, pcov = fit_gaussian(entries, bin_centers)
amp, mean, sigma = popt

# Evaluate the fitted curve
y = gaussian(bin_centers, amp, mean, sigma)
```

### 6. **beam_analysis.py**
Main analysis class (to be created from extracted BeamAnalysis).

**Key Methods (organized by function):**

**Initialization & I/O:**
- `__init__()` - Initialize analysis run
- `end_analysis()` - Finalize and close output
- `open_file()` - Read ROOT file and extract detector data

**Calibration:**
- `adjust_1pe_calibration()` - Calibrate photoelectron response

**Particle Identification:**
- `tag_electrons_ACT02()` - Identify electrons using upstream ACT
- `tag_electrons_ACT35()` - Additional electron ID using downstream ACT
- `tag_protons_TOF()` - Identify protons via time-of-flight
- `tag_muons_pions_ACT35()` - Separate muons and pions using ACT charge

**Time-of-Flight Analysis:**
- `give_theoretical_TOF()` - Calculate theoretical TOF accounting for material losses
- `give_tof()` - Calculate TOF for given particle and momentum
- `measure_particle_TOF()` - Measure TOF distributions for each particle type

**Momentum Estimation:**
- `TOF_particle_in_ns()` - TOF for particle at given momentum
- `return_losses()` - Calculate momentum loss through materials
- `extrapolate_momentum()` - Extract momentum from measured TOF
- `extrapolate_trigger_momentum()` - Vectorized momentum extraction
- `estimate_momentum()` - Estimate mean momentum for each particle
- `estimate_particle_momentum()` - Enhanced momentum estimation

**Data Output:**
- `write_output_particles()` - Write selected particles to Parquet file
- `output_beam_ana_to_root()` - Write analysis results to ROOT
- `output_to_root()` - Write processed data to ROOT file

**Visualization:**
- `plot_ACT35_left_vs_right()` - 2D ACT charge plot
- `plot_all_TOFs()` - Plot all TOF distributions
- `plot_TOF_charge_distribution()` - TOF vs charge plots
- `study_electrons()` - Analyze electron sample
- `study_beam_structure()` - Analyze beam timing structure
- `plot_number_particles_per_POT()` - Particle yield analysis

## Migration Guide

### For Users

If you're importing from the old monolithic file:

**Old:**
```python
from beam_monitors_pid import BeamAnalysis, fit_gaussian, stage_local
```

**New:**
```python
from beam_monitors_pid import BeamAnalysis
from beam_monitors_pid.fitting import fit_gaussian
from beam_monitors_pid.file_utils import stage_local
from beam_monitors_pid.constants import particle_masses
```

### For Developers

**Adding new functionality:**
1. Identify which module it belongs in
2. Add the function/class to the appropriate module
3. Import in `__init__.py` if needed for public API
4. Add docstrings with parameters and return values
5. Consider adding to `constants.py` if it's configuration data

**Adding new detector constants:**
```python
# Add to constants.py
NEW_DETECTOR_GROUP = [24, 25, 26, 27]
NEW_DETECTOR_QDC_THRESHOLD = 150
```

**Adding new fitting functions:**
```python
# Add to fitting.py
def my_curve(x, param1, param2):
    """Your curve function."""
    return ...

def fit_my_curve(entries, bin_centers):
    """Fit your curve to histogram data."""
    popt, pcov = curve_fit(my_curve, bin_centers, entries, p0=[...])
    return popt, pcov
```

## Benefits of Modular Structure

1. **Maintainability** - Code is organized by function, making changes easier to locate and implement
2. **Readability** - Each module has a clear purpose, easier to understand
3. **Reusability** - Utility functions can be imported and used independently
4. **Testability** - Smaller modules are easier to write unit tests for
5. **Scalability** - New functionality can be added without cluttering existing code
6. **Debugging** - Errors can be traced to specific functional areas

## Example Usage

```python
from beam_monitors_pid import BeamAnalysis
from beam_monitors_pid.constants import t0_group, particle_masses, CHANNEL_MAPPING
from beam_monitors_pid.file_utils import stage_local
from beam_monitors_pid.fitting import fit_gaussian

# Initialize analysis
analysis = BeamAnalysis(
    run_number=12345,
    run_momentum=1000,  # MeV/c
    n_eveto=1.03,
    n_tagger=1.06,
    there_is_ACT5=True,
    output_dir="./output"
)

# Open and process data
analysis.open_file(input_file="data.root")

# Calibrate detectors
analysis.adjust_1pe_calibration()

# Identify particles
analysis.tag_electrons_ACT02()
analysis.tag_electrons_ACT35(cut_line=10.0)
analysis.tag_protons_TOF()
analysis.tag_muons_pions_ACT35()

# Measure TOF and estimate momentum
analysis.measure_particle_TOF()
analysis.estimate_momentum(verbose=True)

# Output results
analysis.output_to_root(output_name="analysis_results.root")
analysis.write_output_particles(
    particle_number_dict={"muon": 1000, "pion": 1000},
    store_PID_info=True,
    filename="selected_particles.parquet"
)

# Finalize
analysis.end_analysis()

# Access results
print(f"Muon mean momentum: {analysis.particle_mom_mean['muon']} MeV/c")
print(f"   with error: {analysis.particle_mom_mean_err['muon']} MeV/c")
```

## File Size Comparison

| File | Lines | Size |
|------|-------|------|
| Original `beam_monitors_pid.py` | ~4,857 | ~500 KB |
| `constants.py` | ~150 | ~6 KB |
| `file_utils.py` | ~130 | ~5 KB |
| `flag_utils.py` | ~80 | ~3 KB |
| `detector_utils.py` | ~120 | ~5 KB |
| `fitting.py` | ~190 | ~8 KB |
| `beam_analysis.py` | ~4,200 | ~420 KB |
| **Total** | ~4,870 | ~447 KB |

The modular approach doesn't significantly reduce total lines, but dramatically improves code organization and reusability.

## Notes for Contributors

1. **Keep modules focused** - If a module grows beyond 300-400 lines, consider splitting it
2. **Update docstrings** - Changes to functions should include updated docstrings
3. **Maintain backward compatibility** - When possible, keep old function signatures working
4. **Add type hints** - Use Python type hints for better IDE support and clarity
5. **Test utilities independently** - Utility functions should work standalone

## References

- WCTE Beam Monitoring System
- Time-of-Flight Measurement and Particle Identification
- Material Budget and Energy Loss Tables (Geant4)
- Detector Configuration YAML: `wcte_beam_detectors.yaml`
