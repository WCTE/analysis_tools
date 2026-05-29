# Analysis Examples

This directory contains examples of how to use common analysis tools and functions.

## DataLoader

`DataLoader` loads processed WCTE ROOT files and applies data quality cuts.

```python
from analysis_tools import DataLoader

loader = DataLoader("/path/to/WCTE_merged_production_RXXXX.root")
```

### Data quality cuts

Enable any combination of cuts before iterating — they are applied automatically per batch:

```python
loader.apply_mPMT_data_quality_cuts()   # window/hit-level mPMT quality masks
loader.apply_vme_event_quality_cuts()   # VME digitisation and event quality
loader.apply_t5_event_quality_cuts()    # T5 event quality cuts
```

### Iterating over data

```python
for batch in loader.iterate(step_size="100 MB"):
    # batch is an awkward array of events passing all enabled cuts
    print(len(batch), "windows")
```

Pass `verbose=True` to print event counts after each cut.

### Metadata helpers

```python
loader.get_configuration()              # run configuration record
loader.get_data_quality_metrics()       # DQM summary
loader.get_vme_analysis_scalar_results()
loader.get_vme_analysis_run_info()
loader.get_good_wcte_pmts()             # returns (slot_ids, position_ids)
```

### Particle ID based on VME analysis

`BeamSelection` defines particle selections using series of cuts on WCTE beam monitor (VME) data. The examples in the code show the nominal selection cuts, they should be adapted to suit your own analysis needs.


```
#read in the scalar results (i.e. nominal cut lines) of the VME analysis 
vme_scalar_results = loader.get_vme_analysis_scalar_results()

# --- Define your particle selections ---
# Every cut is a [variable, operator, value] triplet.
# You can apply a cut to any VME variable 
# Operators: ">", "<", ">=", "<=", "between" (value must be [low, high] for "between").
# Omit the TOF cut entirely if proton_tof_cut is 0 
# This case of TOF separation unavailable happensfor negative polarity and low momentum runs in production 1.0,
# To be improved in production 1.1 

tof_cut    = vme_scalar_results['proton_tof_cut']
if tof_cut == 0:
    print("WARNING: TOF separation unavailable for this run, setting TOF cut to default value of 999 ns.")
    tof_cut = 999

eveto_cut  = vme_scalar_results['act_eveto_cut']
tagger_cut = vme_scalar_results['act_tagger_cut']

# PIONS: fast particles that do not produce Cherenkov light in either ACT.
pion_sel = BeamSelection.pion(
    ["vme_act_eveto",  "<", eveto_cut],
    ["vme_act_tagger", "<", tagger_cut],
    ["vme_tof_corr",        "<", tof_cut],
)

# PROTONS: slow particles identified by their TOF falling in a window above the
#          fast/slow separation value. Only meaningful when proton_tof_cut > 0.

proton_sel = BeamSelection.proton(
    ["vme_tof_corr", "between", [tof_cut, tof_cut + 10]],
)
```

The selections are applied within the iterating loop described above and the `SelectionMonitor` is used to monitor visually the selections. The code write out the selected batches to an external .parquet file, choose this if you want to work with large datasets. You can do quick analysis on individual batches within the 'loader.iterate' loop. 

```# Enable parquet output for the selections you want to save.
# Default filename is "<particle>.parquet". Pass a path to override.
pion_sel.enable_parquet_output(f"run{run_number}_pions.parquet")
muon_sel.enable_parquet_output(f"run{run_number}_muons.parquet")
ele_sel.enable_parquet_output(f"run{run_number}_electrons.parquet")

# Decide which selections you want to monitor live during loading. This is optional but useful for understanding cut lines.
selections = [pion_sel, muon_sel, ele_sel, proton_sel] 
monitor    = SelectionMonitor(selections, update_every=10, vme_run_info=vme_run_info)
```

```for i_batch, batch in enumerate(loader.iterate(verbose=False, step_size="100 MB")):
    n_windows_passing += len(batch)

    monitor.update(batch)
    for sel in selections:
        sel._write_to_parquet(batch[sel.mask(batch)])

    #alternatively, do analysis on individual bath
    pion_batch = batch[pion_sel.mask(batch)]
    #do something with pion_batch...

for sel in selections:
    sel.close_parquet_writer()
```

