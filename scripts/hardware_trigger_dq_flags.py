import numpy as np
import uproot
import awkward as ak
import gc
import tracemalloc
import argparse
import os
import time
import json
from array import array
from analysis_tools import CalibrationDBInterface
from analysis_tools import PMTMapping
from enum import Flag, auto
import subprocess
import hashlib

from data_quality_flags import HitMask, TriggerMask

def get_git_descriptor(debug=False):
    try:
        # Get commit hash / tag
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--tags"],
            stderr=subprocess.STDOUT
        ).decode().strip()

        # Check if there are uncommitted changes (dirty repo)
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        if status:
            if debug:
                print("Warning: Repository has uncommitted changes, but continuing due to debug mode.")
            else:
                raise Exception("Repository has uncommitted changes")
        return desc

    except subprocess.CalledProcessError as e:
        raise RuntimeError("Git command failed") from e


def file_sha256(path, chunk_size=1024 * 1024):
    #get the hash of a file, used to identify input slow control file used
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def get_run_database_data(json_path,run_number):

    with open(json_path, 'r') as f:
        data = json.load(f)

    run_key = str(run_number)
    if run_key not in data:
        raise ValueError(f"Run number {run_number} not found in the JSON data.")
    return data[run_key]    
       
def get_stable_mpmt_list_slow_control(run_data:dict,run_number:int):
    """
    Reads the slow control good run list JSON file to get the stable mPMT list for a given run number
    run_data - dict of data for specific run
    run_number - the run number integer
    returns a length-2 tuple
            set of enabled channels (in card-channel format)
            set of masked channels in card-channel format)
    """
    # with open(good_run_list_path, 'r') as f:
    #     data = json.load(f)

    # run_key = str(run_number)
    # print("run_number",run_number)
    # if run_key not in data:
    #     raise ValueError(f"Run number {run_number} not found in the JSON data.")

    enabled_channels = set(run_data["enabled_channels"])
    channel_mask = set(run_data["channel_mask"])

    return enabled_channels, channel_mask

def mask_windows_missing_waveforms(good_channel_list, readout_window_events):
    """
    Inputs:
    good_channel_list - list or array of integers in slot-position format (e.g. 203 for slot 2 position 3)
    readout_window_events - awkward array with fields:
        pmt_waveform_mpmt_slot_ids - awkward array of mPMT slot IDs for waveforms in each readout window
        pmt_waveform_pmt_position_ids - awkward array of PMT position IDs for waveforms in each readout window
    Returns:
    has_all_good_channels - awkward array of booleans indicating whether one waveform for every good channel
                            was present in the readout window
    """
    
    wf_mpmt_slot = readout_window_events["pmt_waveform_mpmt_slot_ids"]
    wf_pmt_pos = readout_window_events["pmt_waveform_pmt_position_ids"]
    wf_glbl_pmt_ids = (100 * wf_mpmt_slot) + wf_pmt_pos
    
    wf_glbl_pmt_ids_flat = ak.flatten(wf_glbl_pmt_ids).to_numpy()
    has_all_good_channels = np.full(len(wf_glbl_pmt_ids), True, dtype=bool)
    #iterate over each good channel and check if it is present exactly once in each readout window
    #if any channel is not then that window is marked as bad
    for ich, channel in enumerate(good_channel_list):
        if ich%100==0:
            print("Checking good channel",ich,"/",len(good_channel_list))
        channel_mask = wf_glbl_pmt_ids_flat == channel
        channel_counts = ak.sum(ak.unflatten(channel_mask, ak.num(wf_glbl_pmt_ids)), axis=1).to_numpy()
        has_all_good_channels = has_all_good_channels & (channel_counts == 1)
        
    return has_all_good_channels    

def slot_pos_from_card_chan_list(card_chan_list):
    """
    Converts a list of card-channel identifiers to slot-position identifiers
    card_chan_list - list or array of integers in card-channel format (e.g. 201 for card 2 channel 1)
    returns a numpy array of integers in slot-position format (e.g. 203 for slot 2 position 3)
    """
    mapping = PMTMapping()
    slot_pos_list = []
    for ch in card_chan_list:
        card = ch // 100
        pmt_chan = ch % 100
        slot, pmt_pos = mapping.get_slot_pmt_pos_from_card_pmt_chan(card, pmt_chan)
        slot_pos_list.append(100 * slot + pmt_pos)
    return np.array(slot_pos_list)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add a new branch to a ROOT TTree in batches.")
    parser.add_argument("-i","--input_files",required=True, nargs='+', help="Path to WCTEReadoutWindows ROOT file")
    parser.add_argument("-c","--input_calibrated_file_directory",required=True, help="Path to calibrated hits ROOT file")
    parser.add_argument("-hw","--input_wf_processed_file_directory",required=True, help="Path to hardware trigger processed ROOT file")
    parser.add_argument("-r","--run_number",required=True, help="Run Number")
    parser.add_argument("-o","--output_dir",required=True, help="Directory to write output file")
    parser.add_argument("--debug", action="store_true",help="Enable debug - disables checks allowing for test runs")
    args = parser.parse_args()
    
    git_hash = get_git_descriptor(debug=args.debug)

    
    #check that the run number is correct 
    for input_file in args.input_files:
        if f"R{args.run_number}" not in input_file:
            raise Exception(f"Input file {input_file} does not match run number {args.run_number}")
    
    #make a list of calibrated input files - these are needed to determine for the channel list
    #of channels with calibration constants and for the hit list for which the mask is to be applied
    #make list of waveform processed files - these are needed to determine list of channels with short
    #waveforms masked out 
    calibrated_input_files = []
    wf_processed_input_files = []
    
    #counters for statistics
    run_total_triggers = 0
    run_total_bad_triggers = 0
    run_total_hits = 0
    run_total_bad_hits = 0
    
    for input_file in args.input_files:
        base = os.path.splitext(os.path.basename(input_file))[0]
        calibrated_file_name = f"{base}_processed_waveforms_calibrated_hits.root" 
        calibrated_input_file_path = os.path.join(args.input_calibrated_file_directory, calibrated_file_name)
        calibrated_input_files.append(calibrated_input_file_path)
        if not os.path.exists(calibrated_input_file_path):
            raise Exception(f"Calibrated input file {calibrated_input_file_path} does not exist")

        wf_processed_file_name = f"{base}_processed_waveforms.root" 
        wf_processed_file_path = os.path.join(args.input_wf_processed_file_directory, wf_processed_file_name)        
        wf_processed_input_files.append(wf_processed_file_path)
        if not os.path.exists(wf_processed_file_path):
            raise Exception(f"Waveform processed input file {wf_processed_file_path} does not exist")
        
    #slow control file for good run list
    good_run_list_path = '/eos/experiment/wcte/configuration/slow_control_summary/all_run_list.json' 
    
    #get hash of slow control file used
    full_hash = file_sha256(good_run_list_path)
    short_hash = full_hash[:10]   # e.g. first 8–12 chars
    
    #get run configuration from slow control
    run_data = get_run_database_data(good_run_list_path,args.run_number)
    run_configuration = run_data["trigger_name"]
    
    #get stable list of channels from slow control
    enabled_channels, channel_mask = get_stable_mpmt_list_slow_control(run_data,args.run_number)
    #the channels that are enabled less the channels that are determined as unstable
    stable_channels_card_chan = enabled_channels - channel_mask
    #map slow control channel list in card and channel to the mpmt slot and position
    slow_control_stable_channels = slot_pos_from_card_chan_list(stable_channels_card_chan)
    
    #loop over each file    
    first_file_pmts_with_timing_constant = None
    first_file_bad_wf_processed_pmts = None
    for readout_window_file_name, calibrated_input_file_name, wf_processed_input_file_name in zip(args.input_files, calibrated_input_files, wf_processed_input_files):
        #open the original file, the calibrated hits file and the waveform processed file
        with uproot.open(readout_window_file_name) as readout_window_file:
            with uproot.open(calibrated_input_file_name) as calibrated_file:
                with uproot.open(wf_processed_input_file_name) as wf_processed_input_file:
                    
                    #get the list of pmts with timing constants from the calibrated file
                    config_tree = calibrated_file["Configuration"]
                    pmts_with_timing_constant = config_tree["wcte_pmts_with_timing_constant"].array().to_numpy()[0]
                    
                    #get the list of pmts masked out due to short waveforms from the waveform processed file
                    wf_processed_config_tree = wf_processed_input_file["Configuration"]
                    wf_processed_bad_channel_card_chan = wf_processed_config_tree["bad_channel_mask_card_chan"].array().to_numpy()[0]
                    #change to slot position notation
                    wf_processed_bad_channel = slot_pos_from_card_chan_list(wf_processed_bad_channel_card_chan)
                
                    #Check the lists are the same between files in the same run 
                    if first_file_pmts_with_timing_constant is None:
                        first_file_pmts_with_timing_constant = pmts_with_timing_constant
                    else:
                        if not np.array_equal(first_file_pmts_with_timing_constant, pmts_with_timing_constant):
                            raise Exception("PMTs with timing constants do not match between files in the same run")
                    
                    if first_file_bad_wf_processed_pmts is None:
                        first_file_bad_wf_processed_pmts = wf_processed_bad_channel
                    else:   
                        if not np.array_equal(first_file_bad_wf_processed_pmts, wf_processed_bad_channel):
                            raise Exception("Bad waveform processed PMTs do not match between files in the same run")
                    
                    #construct the good wcte pmt list
                    good_wcte_pmts = (set(pmts_with_timing_constant) & set(slow_control_stable_channels))- set(wf_processed_bad_channel)
                    
                    # Construct output path
                    base = os.path.splitext(os.path.basename(readout_window_file_name))[0]
                    new_filename = f"{base}_hw_trigger_dq_flags.root" 
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_file_name = os.path.join(args.output_dir, new_filename)
                    with uproot.recreate(output_file_name) as outfile:
                
                        config_tree = outfile.mktree("Configuration", {
                            "git_hash": "string",
                            "run_configuration": "string",
                            "good_wcte_pmts": "var * int32", #the global pmt id (slot*100 + pos) of good pmts with timing constants and stable in slow control
                            "wcte_pmts_with_timing_constant": "var * int32", #the global pmt id (slot*100 + pos) of pmts with timing constants
                            "wcte_pmts_slow_control_stable": "var * int32", #the global pmt id (slot*100 + pos) of pmts stable in slow control
                            "wcte_pmts_masked_with_short_waveforms": "var * int32", #the global pmt id (slot*100 + pos) of pmts masked out due to bad short waveforms
                            "slow_control_file_name": "string",
                            "slow_control_file_hash": "string",
                            "readout_window_file": "string",
                            "calibrated_hit_file": "string",
                            "wf_processed_file_name": "string"
                        })
                        print("There were",len(stable_channels_card_chan),"enabled channels not masked out")
                        print("There were",len(pmts_with_timing_constant),"channels with timing constants")
                        print("There were",len(wf_processed_bad_channel),"channels with short waveforms masked out")
                        print("In total there are",len(list(good_wcte_pmts)),"good channels with timing constants and stable in slow control")
                        
                        config_tree.extend({
                            "git_hash": [git_hash],
                            "run_configuration": [run_configuration],
                            "good_wcte_pmts": ak.Array([list(good_wcte_pmts)]), 
                            "wcte_pmts_with_timing_constant": ak.Array([pmts_with_timing_constant]),
                            "wcte_pmts_slow_control_stable": ak.Array([slow_control_stable_channels]),
                            "wcte_pmts_masked_with_short_waveforms": ak.Array([wf_processed_bad_channel]),
                            "slow_control_file_name": [good_run_list_path],
                            "slow_control_file_hash": [short_hash],
                            "readout_window_file": [readout_window_file_name],
                            "calibrated_hit_file": [calibrated_input_file_name],
                            "wf_processed_file_name": [wf_processed_input_file_name]
                        })
                        
                        # Create a TTree to store the flags
                        tree = outfile.mktree("DataQualityFlags", { #only WCTE detector hits are stored here (not trigger mainboard hits)
                            "hit_pmt_readout_mask": "var * int32",
                            "window_data_quality_mask": "int32",
                            "readout_number": "int32" #the unique readout window number for this event in the run
                        })
                        
                        #batch load over all entries to apply data quality flags to each trigger and hits
                        readout_window_tree_entries = readout_window_file["WCTEReadoutWindows"].num_entries 
                        calibrated_tree_entries = calibrated_file["CalibratedHits"].num_entries
                        
                        if  readout_window_tree_entries!=calibrated_tree_entries:
                            if args.debug:
                                print("Warning: Input file problem different number of entries between calibrated and original file, but continuing due to debug mode.")
                                print("debug mode: override")
                                readout_window_tree_entries = min(readout_window_tree_entries,calibrated_tree_entries)
                            else:
                                raise Exception("Input file problem different number of entries between calibrated and original file")
                            
                        batch_size = 10_000 #can use large batches as only a couple of branches are loaded
                        for start in range(0, readout_window_tree_entries, batch_size):  
                            stop = min(start + batch_size, readout_window_tree_entries)
                            print(f"Loading entries {start} → {stop}")
                            branches_to_load = ["window_time","readout_number","pmt_waveform_mpmt_slot_ids","pmt_waveform_pmt_position_ids"]
                            readout_window_tree = readout_window_file["WCTEReadoutWindows"]
                            readout_window_events = readout_window_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                            
                            branches_to_load = ["readout_number","hit_mpmt_slot","hit_pmt_pos"]
                            calibrated_tree = calibrated_file["CalibratedHits"]
                            calibrated_file_events = calibrated_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                            
                            if not np.array_equal(readout_window_events["readout_number"].to_numpy(),calibrated_file_events["readout_number"].to_numpy()):
                                raise Exception("Batch start",start,"different events being compared between two files")
                            
                            #trigger level flags
                            missing_waveforms_mask = mask_windows_missing_waveforms(good_wcte_pmts, readout_window_events)
                            print("Checked",len(missing_waveforms_mask),"windows for missing waveforms", np.sum(~missing_waveforms_mask), "bad windows found")
                            
                            #make the trigger level bitmask 
                            trigger_mask = np.zeros_like(missing_waveforms_mask, dtype=np.int32)                        
                            trigger_mask |= ~missing_waveforms_mask * TriggerMask.MISSING_WAVEFORMS.value
                            
                            #hit level flags
                            hit_global_id = (100*calibrated_file_events["hit_mpmt_slot"])+calibrated_file_events["hit_pmt_pos"]
                            hit_global_id_flat = ak.to_numpy(ak.flatten(hit_global_id))
                            
                            has_time_constant = np.isin(hit_global_id_flat,pmts_with_timing_constant)
                            is_sc_stable = np.isin(hit_global_id_flat,slow_control_stable_channels)
                            is_short_wf_masked = np.isin(hit_global_id_flat,wf_processed_bad_channel)
                            #make the trigger level bitmask 
                            hit_mask_flat = np.zeros_like(hit_global_id_flat, dtype=np.int32)
                            hit_mask_flat |= ~has_time_constant * HitMask.NO_TIMING_CONSTANT.value
                            hit_mask_flat |= ~is_sc_stable * HitMask.SLOW_CONTROL_EXCLUDED.value
                            hit_mask_flat |= is_short_wf_masked * HitMask.SHORT_WAVEFORM.value
                            
                            hit_mask = ak.unflatten(hit_mask_flat,ak.num(hit_global_id))
                            
                            #append to tree
                            tree.extend({
                                "hit_pmt_readout_mask": hit_mask,
                                "window_data_quality_mask": trigger_mask,
                                "readout_number": readout_window_events["readout_number"].to_numpy()
                            })
                            
                            run_total_triggers += len(trigger_mask)
                            run_total_bad_triggers +=np.sum(trigger_mask!=0)
                            run_total_hits += len(hit_mask_flat)
                            run_total_bad_hits +=np.sum(hit_mask_flat!=0)
                        
                            print("Batch processed",np.sum(trigger_mask==0),"/",len(trigger_mask),"good triggers", f"{np.sum(trigger_mask==0)/len(trigger_mask):.2%}")
                            print("Processed",np.sum(hit_mask_flat==0),"/",len(hit_mask_flat),"good hits", f"{np.sum(hit_mask_flat==0)/len(hit_mask_flat):.2%}")
    
    print("Run",args.run_number,"has",len(good_wcte_pmts),"good WCTE PMTs")
    print("In total processed",run_total_triggers,"triggers with",run_total_bad_triggers,"bad triggers", f"{run_total_bad_triggers/run_total_triggers:.2%}")
    print("In total processed",run_total_hits,"hits with",run_total_bad_hits," bad hits", f"{run_total_bad_hits/run_total_hits:.2%}")
    print("Finished processing run",args.run_number,"across",len(args.input_files),"files")
    print("*** Hardware trigger DQ flags script complete ***")
