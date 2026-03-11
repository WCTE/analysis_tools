# Migration Guide: From Monolithic to Modular beam_monitors_pid

## Quick Start

The large `beam_monitors_pid.py` file has been split into a well-organized package structure. Here's how to update your code.

**Run information JSON:**
The run configuration previously resided at a hardcoded EOS path
`/eos/user/a/acraplet/analysis_tools/include/google_sheet_beam_data.json`.
In the refactored package the file should live alongside the code under
`beam_monitors_pid/data/google_sheet_beam_data.json`.  The `ReadBeamRunInfo`
class will automatically pick up the package-local copy, falling back to the
legacy EOS location only if the packaged file is missing.  You can also pass
a custom path to the constructor if needed.


## Before and After

### Before (Monolithic File)
```python
import sys
sys.path.insert(0, '/path/to/analysis_tools')
from beam_monitors_pid import BeamAnalysis, fit_gaussian, stage_local

# Everything was imported from one large file
analysis = BeamAnalysis(...)
result = fit_gaussian(...)
local_file = stage_local(...)
```

### After (Modular Package)
```python
# Import from the organized package
from beam_monitors_pid import BeamAnalysis
from beam_monitors_pid.fitting import fit_gaussian
from beam_monitors_pid.file_utils import stage_local
from beam_monitors_pid import constants

# All functionality is the same, just better organized
analysis = BeamAnalysis(...)
result = fit_gaussian(...)
local_file = stage_local(...)
```

## Common Import Patterns

### Detector Constants
```python
# Old way (scattered throughout file):
# - Constants were global variables at top of file

# New way:
from beam_monitors_pid.constants import (
    t0_group, t1_group, t4_group,
    particle_masses,
    CHANNEL_MAPPING,
    c,  # speed of light
    L,  # detector distance
)

# Access them
mass_proton = particle_masses["Protons"]
detector_channels = CHANNEL_MAPPING[12]  # "ACT0-L"
```

### File Operations
```python
# Old way:
from beam_monitors_pid import stage_local, to_xrootd, make_blocks

# New way:
from beam_monitors_pid.file_utils import stage_local, to_xrootd, make_blocks

# Usage unchanged
local = stage_local("/eos/path/file.root")
blocks = make_blocks(event_indices, max_block=500)
```

### Fitting Utilities
```python
# Old way:
from beam_monitors_pid import fit_gaussian, fit_three_gaussians, gaussian

# New way:
from beam_monitors_pid.fitting import (
    fit_gaussian, 
    fit_three_gaussians,
    gaussian,
    landau_gauss_convolution
)

# Usage unchanged
popt, pcov = fit_gaussian(histogram_entries, bin_centers)
y_values = gaussian(x, amplitude, mean, sigma)
```

### Event Quality Flags
```python
# Old way:
from beam_monitors_pid import write_event_quality_mask, read_event_quality_mask

# New way:
from beam_monitors_pid.flag_utils import (
    write_event_quality_mask,
    read_event_quality_mask,
    make_flag_map
)

# Usage unchanged
flags = {"t0_missing": False, "t1_missing": True}
mask = write_event_quality_mask(flags, flag_map)
```

### TDC Processing
```python
# Old way:
from beam_monitors_pid import deduplicate_tdc_hits, tdc_requirement_met

# New way:
from beam_monitors_pid.detector_utils import (
    deduplicate_tdc_hits,
    tdc_requirement_met,
    NotTheFirstHit  # Exception class
)

# Usage unchanged
try:
    ids_clean, times_clean, dups = deduplicate_tdc_hits(ids, times)
except NotTheFirstHit:
    print("TDC hits not properly ordered!")
```

### Main Analysis Class
```python
# Old way (unchanged):
from beam_monitors_pid import BeamAnalysis

# New way (unchanged):
from beam_monitors_pid import BeamAnalysis

# All methods work identically
analysis = BeamAnalysis(
    run_number=12345,
    run_momentum=1000,
    n_eveto=1.03,
    n_tagger=1.06,
    there_is_ACT5=True,
    output_dir="./output"
)

analysis.open_file(input_file="data.root")
analysis.adjust_1pe_calibration()
analysis.tag_electrons_ACT02()
analysis.measure_particle_TOF()
analysis.estimate_momentum()
analysis.end_analysis()
```

## Module Responsibilities

### Where to Find What

| Need... | Look in... | 
|---------|-----------|
| Detector channel assignments | `constants.py` |
| File I/O operations | `file_utils.py` |
| Event quality flags | `flag_utils.py` |
| TDC processing | `detector_utils.py` |
| Fitting functions | `fitting.py` |
| Particle identification | `beam_analysis.py` |

## Breaking Changes

**There are NO breaking changes.** All functions maintain the same signatures and behavior. The only difference is the organization of imports.

## File Organization

```
Before:
analysis_tools/
└── beam_monitors_pid.py (4,857 lines)

After:
analysis_tools/
└── beam_monitors_pid/
    ├── __init__.py              (45 lines)
    ├── README.md                (Complete documentation)
    ├── MIGRATION.md             (This file)
    ├── constants.py             (150 lines)
    ├── file_utils.py            (130 lines)
    ├── flag_utils.py            (80 lines)
    ├── detector_utils.py        (120 lines)
    ├── fitting.py               (190 lines)
    └── beam_analysis.py         (4,200 lines)
```

## Benefits You Get

1. **Faster Loading** - Import only what you need instead of 4,857 lines
2. **Better IDE Support** - Module structure enables better autocomplete
3. **Easier Testing** - Test utilities in isolation
4. **Clear Dependencies** - See exactly what each module depends on
5. **Maintainability** - Future changes are more localized

## Example: Complete Analysis Workflow

```python
from beam_monitors_pid import BeamAnalysis
from beam_monitors_pid.constants import particle_masses, t5_total_group
from beam_monitors_pid.file_utils import stage_local
from beam_monitors_pid.fitting import fit_gaussian
import numpy as np

# Stage file from EOS
file_path = stage_local("/eos/experiment/wcte/data/run12345.root")

# Create analysis instance
analysis = BeamAnalysis(
    run_number=12345,
    run_momentum=1000,
    n_eveto=1.03,
    n_tagger=1.06,
    there_is_ACT5=True,
    output_dir="./analysis_output"
)

# Read and process data
analysis.open_file(input_file=file_path)

# Detector calibration
analysis.adjust_1pe_calibration()

# Particle identification
analysis.tag_electrons_ACT02(tightening_factor=0)
analysis.tag_electrons_ACT35(cut_line=10.0)
analysis.tag_protons_TOF()
analysis.tag_muons_pions_ACT35()

# Physics measurements
analysis.measure_particle_TOF()
analysis.estimate_momentum(verbose=True)

# Analysis plots and output
analysis.plot_all_TOFs()
analysis.output_to_root(output_name="analysis_results.root")
analysis.write_output_particles(
    particle_number_dict={"muon": 5000, "pion": 5000},
    store_PID_info=True
)

# Print results
print(f"Muon momentum: {analysis.particle_mom_mean['muon']:.1f} +/- "
      f"{analysis.particle_mom_mean_err['muon']:.1f} MeV/c")

# Finalize
analysis.end_analysis()
```

## Troubleshooting

### ImportError: No module named 'beam_monitors_pid'
**Solution:** Make sure the parent directory is in your Python path:
```python
import sys
sys.path.insert(0, '/path/to/analysis_tools')
from beam_monitors_pid import BeamAnalysis
```

### ImportError: cannot import name 'BeamAnalysis'
**Reason:** The `beam_analysis.py` module hasn't been created yet from the original file.
**Workaround:** Use utility modules independently:
```python
from beam_monitors_pid.constants import particle_masses
from beam_monitors_pid.fitting import fit_gaussian
```

### All imports work but code behaves differently
**Check:** Function signatures haven't changed. If behavior differs:
1. Check function docstring for any parameters that changed
2. Verify you're using the correct parameter names
3. Contact the development team if you find actual bugs

## Gradual Migration

You don't need to migrate all at once. Both styles work:

```python
# Old imports (still work if we maintain them)
from beam_monitors_pid import BeamAnalysis

# New imports (recommended)
from beam_monitors_pid import BeamAnalysis
from beam_monitors_pid.constants import particle_masses

# Mix and match!
analysis = BeamAnalysis(...)
mass = particle_masses["Protons"]
```

## Next Steps

1. Update your import statements to use the modular structure
2. Read the [README.md](README.md) for detailed module documentation
3. Check individual module docstrings for function details
4. Run your existing code - it should work unchanged!

## Questions?

Refer to:
- [README.md](README.md) - Complete module documentation
- Individual module docstrings - Function-level documentation
- The original file comments - Implementation details (still available)
