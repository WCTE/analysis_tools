from beam_monitors_pid import BeamAnalysis
analysis = BeamAnalysis(run_number=12345, run_momentum=1000, n_eveto=1.03, 
                       n_tagger=1.06, there_is_ACT5=True, output_dir="./output")
analysis.open_file(input_file="data.root")
# ... rest of analysis workflow

