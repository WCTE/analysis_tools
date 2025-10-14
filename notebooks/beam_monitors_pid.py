#this code holds the functions necessary for reading in the data and identifying the paticle types based on the WCTE beam monitors 

import json
from matplotlib.backends.backend_pdf import PdfPages
import uproot
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import pandas as pd
import awkward as ak
import pyarrow as pa, pyarrow.parquet as pq
import gc

#for the nice progress bar
from tqdm import tqdm

import os, shutil, subprocess, time, hashlib



#Helper functions for file reading, written by Sahar
def stage_local(src_eos_path: str, min_free_gb=20, min_bytes=1_000_000) -> str:
    st = shutil.disk_usage("/tmp")
    if st.free/1e9 < min_free_gb:
        print("Not enough /tmp space; will stream from EOS")
        return ""

    # Use a unique local name to avoid collisions across different directories
    h = hashlib.md5(src_eos_path.encode()).hexdigest()[:8]
    local = f"/tmp/{os.path.basename(src_eos_path)}.{h}"

    def good(p): return os.path.exists(p) and os.path.getsize(p) >= min_bytes

    if not good(local):
        if os.path.exists(local):
            print("Cached local copy is too small; re-staging…")
            os.remove(local)
        print("Staging ROOT file to local disk (xrdcp)…")
        subprocess.run(["xrdcp", "-f", to_xrootd(src_eos_path), local], check=True)
        if not good(local):
            raise OSError(f"Local file too small after xrdcp: {local}")

    print("Using local copy:", local)
    return local


def to_xrootd(path: str) -> str:
    assert path.startswith("/eos/")
    return "root://eosuser.cern.ch//eos" + path[4:]


def make_blocks(idx: np.ndarray, max_block: int):
    if idx.size == 0:
        return []
    blocks = []
    start = idx[0]
    last  = idx[0]
    for v in idx[1:]:
        # if extending the block stays ≤ max_block, keep extending
        if (v - start) < max_block:
            last = v
        else:
            blocks.append((int(start), int(last)+1))
            start = last = v
    blocks.append((int(start), int(last)+1))
    return blocks




#Default informations

proton_tof_cut = 17.5 #ad-hoc but works for most analyses
deuteron_tof_cut = 35 #ad-hoc but works for most analyses
        

c = 0.299792458 #in m.ns^-1 do not change the units please as these are called throuhout
L =  444.03 #4.3431
L_t0t4 = 305.68 #2.9485
L_t4t1 = 143.38 #1.3946

# Particle masses in GeV/c^2
particle_masses = {
    "Electrons": 0.000511,
    "Muons": 0.105658,
    "Pions": 0.13957,
    "Protons": 0.938272
}

reference_ids = (31, 46)          # (TDC ref for IDs <31, ref1 for IDs >31)
t0_group     = [0, 1, 2, 3]       # must all be present
t1_group     = [4, 5, 6, 7]       # must all be present
t4_group     = [42, 43]           # must all be present
t4_qdc_cut   = 200                # Only hits above this value
ACT0_group   = (12, 13)                
ACT1_group   = (14, 15)
ACT2_group   = (16, 17)                

ACT3_group   = (18, 19)                
ACT4_group   = (20, 21)           #actually...      
ACT5_group   = (22, 23)           #actually...      
act_eveto_group = [12, 13, 14, 15, 16, 17]   # ACT-eveto channels
act_tagger_group = [18, 19, 20, 21, 22, 23]
hc_group = [9, 10]
hc_charge_cut = 150

#basic functions, necessary in general
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def fit_gaussian(entries, bin_centers):
    # Get bin centers from edges

    amp_guess = np.max(entries)
    mean_guess = bin_centers[np.argmax(entries)]
    sigma_guess = np.std(np.repeat(bin_centers, entries.astype(int)))

    popt, pcov = curve_fit(gaussian, bin_centers, entries,
                           p0=[amp_guess, mean_guess, sigma_guess])
    
    return popt, pcov




class BeamAnalysis:
    def __init__(self, run_number, run_momentum, n_eveto, n_tagger, there_is_ACT5):
        #Store the run characteristics
        self.run_number, self.run_momentum = run_number, run_momentum
        self.n_eveto, self.n_tagger = n_eveto, n_tagger
        self.there_is_ACT5 = there_is_ACT5
        
        self.pdf_global = PdfPages(f"plots/PID_run{run_number}_p{run_momentum}.pdf")
        self.channel_mapping = {12: "ACT0-L", 13: "ACT0-R", 14: "ACT1-L", 15: "ACT1-R", 16: "ACT2-L", 17: "ACT2-R", 18: "ACT3-L", 19: "ACT3-R", 20: "ACT4-L", 21: "ACT4-R", 22: "ACT5-L", 23: "ACT5-R"}
        
    def end_analysis(self):
        self.pdf_global.close()
        
    def open_file(self, n_events = -1):
        '''Read in the data as a pandas dataframe, read in the TOF and the ACt information'''
        
        
        file_path    = f"/eos/experiment/wcte/data/2025_commissioning/offline_data_vme_match/WCTE_offline_R{self.run_number}S0_VME_matched.root"
        tree_name    = "WCTEReadoutWindows"

        # Open the file and grab the tree
        f    = uproot.open(file_path)
        tree = f[tree_name]

        # Load all four branches into NumPy arrays
        branches = [
            "beamline_pmt_tdc_times",
            "beamline_pmt_tdc_ids",
            "beamline_pmt_qdc_charges",
            "beamline_pmt_qdc_ids",
        ]

        #low number of entries for testing
        if n_events == -1:
            data = tree.arrays(branches, library="np")
        else:
            data = tree.arrays(branches, library="np", entry_start = 0, entry_stop = n_events)
        
        
        #read the calibration file
        
        with open('1pe_calibration.json', 'r') as file:
            calib_constants = json.load(file)

        # Access the calibration constants
        calibration = calib_constants["BeamCalibrationConstants"][0]

        
        #Read all the entries
        act0_l, act1_l, act2_l, act3_l, act4_l, act5_l = [], [], [], [], [], []
        act0_r, act1_r, act2_r, act3_r, act4_r, act5_r = [], [], [], [], [], []
        
#         TOF_0, TOF_1, TOF_2, TOF_3, TOF_4, TOF_5, TOF_6, TOF_7, TOF_8, TOF_9, TOF_A, TOF_B, TOF_C, TOF_D, TOF_E, TOF_F = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]


        total_TOF_charge = []

        act0_time_l, act0_time_r = [], []
        mu_tag_l, mu_tag_r = [], []

        is_kept = []
        is_kept_event_id = [] #keeping track of which events we want to keep

        #also save the time of flight information
        t0_avgs  = []
        t1_avgs  = []
        tof_vals = []
        tof_t0t4_vals = []
        tof_t4t1_vals = []
        act_eveto_sums = []
        act_tagger_sums = []
        event_id = []
        ref0_times = []
        ref1_times = []


        nEvents = len(data[branches[0]])
        
        pbar = tqdm(total=nEvents, desc="Reading in events", unit="evt")

        for evt_idx in range(nEvents):
            pbar.update(1)
            keep_event = True
            tdc_times = data["beamline_pmt_tdc_times"][evt_idx]
            tdc_ids   = data["beamline_pmt_tdc_ids"][evt_idx]

            # reference-time subtraction & first-hit only
            mask0 = (tdc_ids == reference_ids[0])
            mask1 = (tdc_ids == reference_ids[1])
            if not mask0.any() or not mask1.any():
                keep_event = False
                ref0 = 0
                ref1 = 0
                
                
               
            else:
                ref0 = tdc_times[mask0][0]
                ref1 = tdc_times[mask1][0]
                
            ref0_times.append(ref0)
            ref1_times.append(ref1)

            corrected = {}
            for ch, t in zip(tdc_ids, tdc_times):
                if ch in reference_ids or ch in corrected:
                    continue
                corrected[ch] = t - (ref0 if ch < reference_ids[0] else ref1)


            save_t0t1 = True
            # require all channels on T0/T1 before computing averages
            if not all(ch in corrected for ch in t0_group+t1_group):
#                 is_kept.append(False)
                  keep_event = False
                  save_t0t1 = False
#                 continue

            save_t4 = True
            #require already that there is a hit in all t4 PMTs
            if not all(ch in corrected for ch in t4_group):
#                 is_kept.append(False)
                  keep_event = False
                  save_t4 = False
    
#                 continue


            # compute the averages
            if save_t0t1:
                t0 = np.mean([corrected[ch] for ch in t0_group])
                t1 = np.mean([corrected[ch] for ch in t1_group])
            else:
                t0, t1 = 0, 0
                
            if save_t4:
                t4 = np.mean([corrected[ch] for ch in t4_group])
            else:
                t4 = 0


            qdc_charges = data["beamline_pmt_qdc_charges"][evt_idx]
            qdc_ids     = data["beamline_pmt_qdc_ids"][evt_idx]

            qdc_dict = {}
            tdc_dict = {}
            pe_dict = {}
            for ch, q in zip(qdc_ids, qdc_charges):
                if ch not in qdc_dict:
                    qdc_dict[ch] = q

                    if (ch in act_eveto_group) or (ch in act_tagger_group):  
                        #use mean gain and mean pedestal, find it in the calibration data base
                        calib_index = calibration["channel_id"].index(ch)
                        gain = calibration["gain_value"][calib_index]
                        pedestal = calibration["pedestal_mean"][calib_index]
                        pe_dict[ch] = (q-pedestal)/gain 


           

            #------T0 / T1 CUT------#
            if not(all(ch in corrected for ch in t0_group) and all(ch in corrected for ch in t1_group)):
                keep_event = False
#                 is_kept.append(keep_event)
#                 continue

            #--------- T4 cut ----------
            if not(all(ch in qdc_dict for ch in t4_group) and all(qdc_dict[ch] > t4_qdc_cut for ch in t4_group)):
                keep_event = False 
#                 is_kept.append(keep_event)

#                 continue

            #--------- HC cut ----------
            if (any(qdc_dict.get(ch, 0) >= hc_charge_cut for ch in hc_group)):
                keep_event = False  # if either HC channel fired with charge ≥ threshold, skip event
#                 is_kept.append(keep_event)

#                 continue

#             if keep_event: #apply the cuts and save the outputs of interest
#                 #save the time of flight

            #Keep all of the entries but then df is only the ones that we keep 
            t0_avgs.append(t0)
            t1_avgs.append(t1)
            tof_vals.append(t1 - t0)
            tof_t0t4_vals.append(t4 - t0)
            tof_t4t1_vals.append(t1 - t4)

            #svae the charge 
            act0_l.append(pe_dict.get(12, 0))
            act0_r.append(pe_dict.get(13, 0))
            act1_l.append(pe_dict.get(14, 0))
            act1_r.append(pe_dict.get(15, 0))
            act2_l.append(pe_dict.get(16, 0))
            act2_r.append(pe_dict.get(17, 0))
            act3_l.append(pe_dict.get(18, 0))
            act3_r.append(pe_dict.get(19, 0))
            act4_l.append(pe_dict.get(20, 0))
            act4_r.append(pe_dict.get(21, 0))
            act5_l.append(pe_dict.get(22, 0))
            act5_r.append(pe_dict.get(23, 0))
            
            
#             TOF_0.append(pe_dict.get(48, 0))
#             TOF_1.append(pe_dict.get(49, 0))
#             TOF_2.append(pe_dict.get(50, 0))
#             TOF_3.append(pe_dict.get(51, 0))
#             TOF_4.append(pe_dict.get(52, 0))
#             TOF_5.append(pe_dict.get(53, 0))
#             TOF_6.append(pe_dict.get(54, 0))
#             TOF_7.append(pe_dict.get(55, 0))
#             TOF_8.append(pe_dict.get(56, 0))
#             TOF_9.append(pe_dict.get(57, 0))
#             TOF_A.append(pe_dict.get(58, 0))
#             TOF_B.append(pe_dict.get(59, 0))
#             TOF_C.append(pe_dict.get(60, 0))
#             TOF_D.append(pe_dict.get(61, 0))
#             TOF_E.append(pe_dict.get(62, 0))
#             TOF_F.append(pe_dict.get(63, 0))

            total_TOF_charge.append(qdc_dict.get(48, 0)+ qdc_dict.get(49, 0)+ qdc_dict.get(50, 0)+ qdc_dict.get(51, 0)+ qdc_dict.get(52, 0)+ qdc_dict.get(53, 0)+ qdc_dict.get(54, 0)+ qdc_dict.get(55, 0)+ qdc_dict.get(56, 0)+ qdc_dict.get(57, 0)+ qdc_dict.get(58, 0)+ qdc_dict.get(59, 0)+ qdc_dict.get(60, 0)+ qdc_dict.get(61, 0)+ qdc_dict.get(62, 0)+ qdc_dict.get(63, 0))
            
            event_id.append(evt_idx)

            
            
            
            
            

            mu_tag_l.append(qdc_dict.get(24, 0))        
            mu_tag_r.append(qdc_dict.get(25, 0))

            act0_time_l.append(corrected.get(12, 0))
            act0_time_r.append(corrected.get(13, 0))
                
            is_kept.append(keep_event)
#             is_kept_event_id.append(evt_idx)


        act_arrays = [act0_l, act1_l, act2_l, act3_l, act4_l, act5_l,
                      act0_r, act1_r, act2_r, act3_r, act4_r, act5_r]

        act_arrays = [np.array(arr, dtype=float) for arr in act_arrays]

        act0_time_l = np.array(act0_time_l)
        act0_time_r = np.array(act0_time_r)

        #store them back in their initial name
        (act0_l, act1_l, act2_l, act3_l, act4_l, act5_l,
         act0_r, act1_r, act2_r, act3_r, act4_r, act5_r) = act_arrays

        tof_vals = np.array(tof_vals)
        tof_t0t4_vals = np.array(tof_t0t4_vals)
        tof_t4t1_vals = np.array(tof_t4t1_vals)
        mu_tag_l, mu_tag_r = np.array(mu_tag_l), np.array(mu_tag_r) 
        
        data_dict = {
            "act0_l": act0_l,
            "act1_l": act1_l,
            "act2_l": act2_l,
            "act3_l": act3_l,
            "act4_l": act4_l,
            "act5_l": act5_l,
            "act0_r": act0_r,
            "act1_r": act1_r,
            "act2_r": act2_r,
            "act3_r": act3_r,
            "act4_r": act4_r,
            "act5_r": act5_r,
            "event_id":event_id,
            "total_TOF_charge":total_TOF_charge,
#             "TOF_0": TOF_0,
#             "TOF_1": TOF_1,
#             "TOF_2": TOF_2,
#             "TOF_3": TOF_3,
#             "TOF_4": TOF_4,
#             "TOF_5": TOF_5,
#             "TOF_6": TOF_6,
#             "TOF_7": TOF_7,
#             "TOF_8": TOF_8,
#             "TOF_9": TOF_9,
#             "TOF_A": TOF_A,
#             "TOF_B": TOF_B,
#             "TOF_C": TOF_C,
#             "TOF_D": TOF_D,
#             "TOF_E": TOF_E,
#             "TOF_F": TOF_F,
            "act0_time_l": act0_time_l,
            "act0_time_r": act0_time_r,
            "tof": tof_vals,
            "tof_t0t4": tof_t0t4_vals,
            "tof_t4t1": tof_t4t1_vals,
            "mu_tag_l": mu_tag_l,
            "mu_tag_r": mu_tag_r,
            "is_kept": is_kept,
            "ref0_time":ref0_times,
            "ref1_time":ref1_times,
            
        }
        
        #all arrays are supposed to the be same lenngth
#         for i, (key, value) in enumerate(data_dict.items()):
#             print(key, len(value))

        
        # create DataFrame, much more robust than having many arrays 
        self.df_all = pd.DataFrame(data_dict)
        
        
        self.df_all["act_eveto"] = self.df_all["act0_l"]+self.df_all["act0_r"]+self.df_all["act1_l"]+self.df_all["act1_r"]+self.df_all["act2_l"]+self.df_all["act2_r"]
        
        
        if self.there_is_ACT5:
            self.PMT_list = ["act0_l", "act0_r", "act1_l",  "act1_r", "act2_l", "act2_r", "act3_l", "act3_r", "act4_l", "act4_r", "act5_l", "act5_r"]
            self.df_all["act_tagger"] = self.df_all["act3_l"]+self.df_all["act3_r"]+self.df_all["act4_l"]+self.df_all["act4_r"]+self.df_all["act5_l"]+self.df_all["act5_r"]
            
        else:
            self.PMT_list = ["act0_l", "act0_r", "act1_l",  "act1_r", "act2_l", "act2_r", "act3_l", "act3_r", "act4_l", "act4_r"]
            self.df_all["act_tagger"] = self.df_all["act3_l"]+self.df_all["act3_r"]+self.df_all["act4_l"]+self.df_all["act4_r"]
        
        self.df = self.df_all[self.df_all["is_kept"] == 1].copy()
        
        #this will be necessary for identifying events later
        self.is_kept = is_kept
        
        #store the id of the events of interest
        self.is_kept_event_id = is_kept_event_id
        
        
        
       
                             
        pbar.close()
        
        print(f"Total weight of self.df: {self.df.memory_usage(deep=True).sum()/ (1024**2):.2f}Mb")
        
        
#     def open_file_new_new(self, n_events=-1):
#         """Read in the data as a pandas DataFrame using vectorized operations (no Python loop)."""

#         # ----------------- File and tree -----------------
# #         file_path = f"/eos/experiment/wcte/data/2025_commissioning/offline_data_vme_match/WCTE_offline_R{self.run_number}S0_VME_matched.root"
#         #make sure we are using the most up to date file 
#         file_path =  f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v0_5/{self.run_number}/WCTE_offline_R{self.run_number}S0_VME_matched.root"
#         tree_name = "WCTEReadoutWindows"

#         f = uproot.open(file_path)
#         tree = f[tree_name]

#         branches = [
#             "beamline_pmt_tdc_times",
#             "beamline_pmt_tdc_ids",
#             "beamline_pmt_qdc_charges",
#             "beamline_pmt_qdc_ids",
#         ]

#         # Load as awkward arrays (fast for jagged arrays)
#         if n_events == -1:
#             data = tree.arrays(branches, library="ak")
#         else:
#             data = tree.arrays(branches, library="ak", entry_start=0, entry_stop=n_events)

#         nEvents = len(data[branches[0]])

#         # ----------------- Calibration -----------------
#         with open('1pe_calibration.json', 'r') as file:
#             calib_constants = json.load(file)
#         calibration = calib_constants["BeamCalibrationConstants"][0]
#         calib_lookup = {ch: (gain, ped) for ch, gain, ped in zip(
#             calibration["channel_id"], calibration["gain_value"], calibration["pedestal_mean"]
#         )}

#         # ----------------- Reference subtraction -----------------
#         tdc_ids = data["beamline_pmt_tdc_ids"]
#         tdc_times = data["beamline_pmt_tdc_times"]

#         # Mask reference channels and get first hit per event
#         ref0_times = ak.firsts(tdc_times[tdc_ids == reference_ids[0]])
#         ref1_times = ak.firsts(tdc_times[tdc_ids == reference_ids[1]])

#         # Subtract reference times for all channels
#         corrected_times = tdc_times - ak.where(tdc_ids < reference_ids[0], ref0_times, ref1_times)

#         # ----------------- Extract ACT0 times -----------------
#         act0_time_l = ak.firsts(corrected_times[tdc_ids == 12], 0)  # default 0 if missing
#         act0_time_r = ak.firsts(corrected_times[tdc_ids == 13], 0)

#         # ----------------- Channel averages for T0/T1/T4 -----------------
#         def channel_mean(ch_group):
#             # Get times for all channels in group, replace missing with nan, then take mean
#             times = ak.fill_none(ak.concatenate([corrected_times[tdc_ids == ch] for ch in ch_group], axis=1), np.nan)
#             return ak.mean(times, axis=1)

#         t0_avg = channel_mean(t0_group)
#         t1_avg = channel_mean(t1_group)
#         t4_avg = channel_mean(t4_group)

#         # Only keep events where all channels exist
#         mask_keep = (~np.isnan(t0_avg)) & (~np.isnan(t1_avg)) & (~np.isnan(t4_avg))

#         # ----------------- TOF -----------------
#         tof_vals = t1_avg - t0_avg
#         tof_t0t4_vals = t4_avg - t0_avg
#         tof_t4t1_vals = t1_avg - t4_avg

#         # ----------------- QDC -> PE -----------------
#         qdc_ids = data["beamline_pmt_qdc_ids"]
#         qdc_charges = data["beamline_pmt_qdc_charges"]

# #         # Vectorized: create dict of pe for all relevant channels
# #         def pe_array(ch_list):
# #             return ak.fill_none(
# #                 ak.concatenate([((qdc_charges[qdc_ids == ch] - calib_lookup.get(ch, (1.0,0.0))[1]) /
# #                                  calib_lookup.get(ch, (1.0,0.0))[0]) for ch in ch_list], axis=1),
# #                 0
# #             )

        
#         act_chs = [12,14,16,18,20,22,13,15,17,19,21,23]
#         act_arrays = np.stack([
#         ak.to_numpy(ak.firsts((qdc_charges[qdc_ids == ch] - calib_lookup.get(ch, (1.0,0.0))[1]) /
#                               calib_lookup.get(ch, (1.0,0.0))[0], 0))
#         for ch in act_chs
#     ], axis=1)
        
        
#         print("act_arrays shape:", act_arrays.shape)  # Should be (nEvents, 12)


#         # ----------------- mu_tag -----------------
#         mu_tag_l = ak.firsts((qdc_charges[qdc_ids == 24])) # - calib_lookup.get(24,(1,0))[1])/calib_lookup.get(24,(1,0))[0],0)
#         mu_tag_r = ak.firsts((qdc_charges[qdc_ids == 25])) # - calib_lookup.get(25,(1,0))[1])/calib_lookup.get(25,(1,0))[0],0)

#         # ----------------- Create DataFrame -----------------
#         act_columns = ["act0_l","act1_l","act2_l","act3_l","act4_l","act5_l",
#                        "act0_r","act1_r","act2_r","act3_r","act4_r","act5_r"]
        
#         print("mu_tag_l shape:", mu_tag_l.shape)
#         print("mu_tag_r shape:", mu_tag_r.shape)
        
#         df_act = pd.DataFrame(ak.to_numpy(act_arrays), columns=act_columns)

#         df_extra = pd.DataFrame({
#             "act0_time_l": ak.to_numpy(act0_time_l),
#             "act0_time_r": ak.to_numpy(act0_time_r),
#             "tof": ak.to_numpy(tof_vals),
#             "tof_t0t4": ak.to_numpy(tof_t0t4_vals),
#             "tof_t4t1": ak.to_numpy(tof_t4t1_vals),
#             "mu_tag_l": ak.to_numpy(mu_tag_l),
#             "mu_tag_r": ak.to_numpy(mu_tag_r),
#         })

#         self.df = pd.concat([df_act, df_extra], axis=1)
#         self.is_kept = mask_keep
#         self.is_kept_event_id = np.where(mask_keep)[0].tolist()

#         # ----------------- ACT sums -----------------
#         self.df["act_eveto"] = self.df["act0_l"] + self.df["act0_r"] + self.df["act1_l"] + self.df["act1_r"] + self.df["act2_l"] + self.df["act2_r"]
        
#         if self.there_is_ACT5:
#             self.PMT_list = act_columns
#             self.df["act_tagger"] = self.df["act3_l"] + self.df["act3_r"] + self.df["act4_l"] + self.df["act4_r"] + self.df["act5_l"] + self.df["act5_r"]
#         else:
#             self.PMT_list = act_columns[:-2]
#             self.df["act_tagger"] = self.df["act3_l"] + self.df["act3_r"] + self.df["act4_l"] + self.df["act4_r"]

#         print(f"Total memory usage of df: {self.df.memory_usage(deep=True).sum()/1024**2:.2f} MiB")

        
        
#     def open_file_new(self, n_events=-1):
#         """Read in the data as a pandas DataFrame with TOF and ACT info."""
        
#         # ----------------- File and tree -----------------
# #         file_path = f"/eos/experiment/wcte/data/2025_commissioning/offline_data_vme_match/WCTE_offline_R{self.run_number}S0_VME_matched.root"


#         file_path =  f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v0_5/{self.run_number}/WCTE_offline_R{self.run_number}S0_VME_matched.root"
#         tree_name = "WCTEReadoutWindows"

#         f = uproot.open(file_path)
#         tree = f[tree_name]

#         # Branches to load
#         branches = [
#             "beamline_pmt_tdc_times",
#             "beamline_pmt_tdc_ids",
#             "beamline_pmt_qdc_charges",
#             "beamline_pmt_qdc_ids",
#         ]

#         # Read arrays with limit on n_events
#         if n_events == -1:
#             data = tree.arrays(branches, library="np")
#         else:
#             data = tree.arrays(branches, library="np", entry_start=0, entry_stop=n_events)

#         nEvents = len(data[branches[0]])

#         # ----------------- Load calibration -----------------
#         with open('1pe_calibration.json', 'r') as file:
#             calib_constants = json.load(file)
#         calibration = calib_constants["BeamCalibrationConstants"][0]

#         # Precompute channel lookup: {ch: (gain, pedestal)}
#         calib_lookup = {ch: (gain, ped) for ch, gain, ped in zip(
#             calibration["channel_id"],
#             calibration["gain_value"],
#             calibration["pedestal_mean"]
#         )}

#         # ----------------- Pre-allocate arrays -----------------
#         act_arrays = np.zeros((nEvents, 12), dtype=float)  # act0_l...act5_r
#         act0_time_l = np.zeros(nEvents, dtype=float)
#         act0_time_r = np.zeros(nEvents, dtype=float)
#         mu_tag_l = np.zeros(nEvents, dtype=float)
#         mu_tag_r = np.zeros(nEvents, dtype=float)
#         tof_vals = np.zeros(nEvents, dtype=float)
#         tof_t0t4_vals = np.zeros(nEvents, dtype=float)
#         tof_t4t1_vals = np.zeros(nEvents, dtype=float)

#         is_kept = np.zeros(nEvents, dtype=bool)
#         is_kept_event_id = []

#         # ----------------- Event loop -----------------
#         pbar = tqdm(total=nEvents, desc="Reading in events", unit="evt")
#         for evt_idx in range(nEvents):
#             pbar.update(1)

#             tdc_times = data["beamline_pmt_tdc_times"][evt_idx]
#             tdc_ids = data["beamline_pmt_tdc_ids"][evt_idx]

#             # reference times
#             mask0 = (tdc_ids == reference_ids[0])
#             mask1 = (tdc_ids == reference_ids[1])
#             if not mask0.any() or not mask1.any():
#                 continue
#             ref0 = tdc_times[mask0][0]
#             ref1 = tdc_times[mask1][0]

#             # subtract reference & first-hit only
#             corrected = {ch: t - (ref0 if ch < reference_ids[0] else ref1)
#                          for ch, t in zip(tdc_ids, tdc_times)
#                          if ch not in reference_ids}

#             # require all T0/T1/T4 channels
#             if not all(ch in corrected for ch in t0_group+t1_group) or not all(ch in corrected for ch in t4_group):
#                 continue

#             t0 = np.mean([corrected[ch] for ch in t0_group])
#             t1 = np.mean([corrected[ch] for ch in t1_group])
#             t4 = np.mean([corrected[ch] for ch in t4_group])

#             qdc_charges = data["beamline_pmt_qdc_charges"][evt_idx]
#             qdc_ids = data["beamline_pmt_qdc_ids"][evt_idx]

#             qdc_dict = {}
#             pe_dict = {}
#             for ch, q in zip(qdc_ids, qdc_charges):
#                 if ch not in qdc_dict:
#                     qdc_dict[ch] = q
#                     if ch in act_eveto_group + act_tagger_group:
#                         gain, ped = calib_lookup.get(ch, (1.0, 0.0))
#                         pe_dict[ch] = (q - ped) / gain

#             # Apply cuts
#             keep_event = True
#             if not all(ch in qdc_dict for ch in t4_group) or not all(qdc_dict[ch] > t4_qdc_cut for ch in t4_group):
#                 keep_event = False
#             if any(qdc_dict.get(ch, 0) >= hc_charge_cut for ch in hc_group):
#                 keep_event = False

#             if keep_event:
#                 # Save outputs
#                 is_kept[evt_idx] = True
#                 is_kept_event_id.append(evt_idx)

#                 tof_vals[evt_idx] = t1 - t0
#                 tof_t0t4_vals[evt_idx] = t4 - t0
#                 tof_t4t1_vals[evt_idx] = t1 - t4

#                 act_arrays[evt_idx, :] = [
#                     pe_dict.get(12,0), pe_dict.get(14,0), pe_dict.get(16,0), pe_dict.get(18,0), pe_dict.get(20,0), pe_dict.get(22,0),
#                     pe_dict.get(13,0), pe_dict.get(15,0), pe_dict.get(17,0), pe_dict.get(19,0), pe_dict.get(21,0), pe_dict.get(23,0)
#                 ]

#                 act0_time_l[evt_idx] = corrected.get(12, 0)
#                 act0_time_r[evt_idx] = corrected.get(13, 0)
#                 mu_tag_l[evt_idx] = qdc_dict.get(24, 0)
#                 mu_tag_r[evt_idx] = qdc_dict.get(25, 0)

#         pbar.close()

#         # ----------------- Build DataFrame -----------------
#         columns = [
#             "act0_l","act1_l","act2_l","act3_l","act4_l","act5_l",
#             "act0_r","act1_r","act2_r","act3_r","act4_r","act5_r"
#         ]
#         df_act = pd.DataFrame(act_arrays, columns=columns)
#         df_extra = pd.DataFrame({
#             "act0_time_l": act0_time_l,
#             "act0_time_r": act0_time_r,
#             "tof": tof_vals,
#             "tof_t0t4": tof_t0t4_vals,
#             "tof_t4t1": tof_t4t1_vals,
#             "mu_tag_l": mu_tag_l,
#             "mu_tag_r": mu_tag_r
#         })
#         self.df = pd.concat([df_act, df_extra], axis=1)

#         self.is_kept = is_kept
#         self.is_kept_event_id = is_kept_event_id

#         # Compute sums
#         self.df["act_eveto"] = self.df["act0_l"] + self.df["act0_r"] + self.df["act1_l"] + self.df["act1_r"] + self.df["act2_l"] + self.df["act2_r"]

#         if self.there_is_ACT5:
#             self.PMT_list = columns
#             self.df["act_tagger"] = self.df["act3_l"] + self.df["act3_r"] + self.df["act4_l"] + self.df["act4_r"] + self.df["act5_l"] + self.df["act5_r"]
#         else:
#             self.PMT_list = columns[:-2]  # exclude act5
#             self.df["act_tagger"] = self.df["act3_l"] + self.df["act3_r"] + self.df["act4_l"] + self.df["act4_r"]

#         print(f"Total memory usage of df: {self.df.memory_usage(deep=True).sum()/1024**2:.2f} MiB")
        
        
    def adjust_1pe_calibration(self):

        bins = np.linspace(-1,3, 100)
        
        self.PMT_value_ped = []
        self.PMT_1pe_scale = []
        for PMT in self.PMT_list:
            fig, ax = plt.subplots(figsize = (8, 6))    
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "Config-file 1pe calib")

            index_ped = np.argmax(h)
            value_ped = 0.5 * (bins[index_ped] + bins[index_ped+1])
            #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            self.df[PMT] -= value_ped 
            self.df_all[PMT] -= value_ped 
#             self.PMT_value_ped.append(value_ped)
            
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "+ pedestal shift")
            #plot the pedestal and 1 pe peak, for now reading from the pre-made calibration, eventually, will have to be made run 
            #check the position of the 1pe peak 
            #restict the portion to fit
            xmin = 0.5
            xmax = 1.5
            bin_centers = (bins[:-1] + bins[1:]) / 2
            is_onepe_region = (bin_centers >= xmin) & (bin_centers <= xmax)
            bins_onepe = bin_centers[is_onepe_region]
            entries_onepe = h[is_onepe_region]

            #fit a gaussian 
            popt, pcov = fit_gaussian(entries_onepe, bins_onepe)
            #ax.plot(bins_onepe, gaussian(bins_onepe, *popt), "k--", label = f"Gaussian fit to 1pe peak:\nMean: {popt[1]:.2f} PE, std: {popt[2]:.2f} PE")

            self.df[PMT] /= popt[1] #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            self.df_all[PMT] /= popt[1] #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            
            
            self.PMT_1pe_scale.append(popt[1])
            
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "+ charge scale")
            entries_onepe = h[is_onepe_region]
            popt, pcov = fit_gaussian(entries_onepe, bins_onepe)
            ax.plot(bins_onepe, gaussian(bins_onepe, *popt), "k--", label = f"Gaussian fit to 1pe peak:\nMean: {popt[1]:.2f} PE, std: {popt[2]:.2f} PE")

            ax.set_yscale("log")
            ax.legend(fontsize = 16)
            ax.set_xlabel("Charge collected (PE)", fontsize = 18)
            ax.set_ylabel("Number of entries", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {PMT}", fontsize = 18)
            ax.grid()
            self.pdf_global.savefig(fig)
            plt.close()
            
        print("One PE calibration finished, please don't forget to check that it is correct")
        
        
        
    def tag_electrons_ACT02(self):
        bins = np.linspace(0, 40, 200)
        fig, ax = plt.subplots(figsize = (8, 6))    
        h, _, _ = ax.hist(self.df["act_eveto"], bins = bins, histtype = "step")

        #automatically find the middle of the least populated bin within 4 and 15 PE
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x_min = 0.2
        x_max = 15
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        # Find the bin index with the minimum count in that range
        min_index = np.argmin(h[mask])

        # Get the actual bin index in the original array
        index = np.where(mask)[0]
        self.eveto_cut = bin_centers[index[min_index]]

        ax.axvline(self.eveto_cut, linestyle = '--', color = 'black', label = f'Optimal electron rejection veto: {self.eveto_cut:.1f} PE')
        ax.set_yscale("log")
        ax.set_xlabel("ACT0-2 total charge (PE)", fontsize = 18)
        ax.set_ylabel("Number of entries", fontsize = 18)
        ax.legend(fontsize = 16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)- ACT0-2", fontsize = 20)
#         pdf.savefig()
        self.pdf_global.savefig(fig)
        plt.close()
        
        #make sure that the particle is not already a slow travelling particle, proton or deuterium
        self.df["is_electron"] = np.where(self.df["act_eveto"]>self.eveto_cut, (self.df["tof"]<proton_tof_cut), False)
        n_electrons = sum(self.df["is_electron"])
        n_triggers = len(self.df["is_electron"])
        print(f"A total of {n_electrons} electrons are tagged with ACT02 out of {n_triggers}, i.e. {n_electrons/n_triggers * 100:.1f}% of the dataset")
        
        
        
    def plot_ACT35_left_vs_right(self, cut_line = None):
        bins = np.linspace(0, 70, 100)
        fig, ax = plt.subplots(figsize = (8, 6))
        act_tagger_l = self.df["act3_l"]+self.df["act4_l"]+self.df["act5_l"] * int(self.there_is_ACT5)
        act_tagger_r =self.df["act3_r"]+self.df["act4_r"]+self.df["act5_r"]* int(self.there_is_ACT5)
        
        h = ax.hist2d(act_tagger_l, act_tagger_r, bins = (bins, bins),norm=LogNorm())
        fig.colorbar(h[3], ax=ax)
        if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"pion/muon cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
        ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
        ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - ACT3-5 all particles", fontsize = 20)
        self.pdf_global.savefig(fig)

        plt.close()
        
        try:
#         if True:
        
            not_electrons = ~self.df["is_electron"]

            fig, ax = plt.subplots(figsize = (8, 6))
            h = ax.hist2d(act_tagger_l[not_electrons], act_tagger_r[not_electrons], bins = (bins, bins), norm=LogNorm())
            fig.colorbar(h[3], ax=ax)
            if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"pion/muon cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
            ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
            ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - ACT3-5 After eveto", fontsize = 20)
            self.pdf_global.savefig(fig)

            plt.close()
            print("ACT35 left vs right plots have been made please check that they are sensible, electrons should be in the top right corner")
            
        except:
            print("Please make the electron selection using tag_electrons_ACT02 before checking the ACT35 left vs right plot, otherwise you will have all of the entries")
            return 0
        
        try:
            not_protons = ~self.df["is_proton"]

            fig, ax = plt.subplots(figsize = (8, 6))
            h = ax.hist2d(act_tagger_l[not_electrons&not_protons], act_tagger_r[not_electrons&not_protons], bins = (bins, bins), norm=LogNorm())
            fig.colorbar(h[3], ax=ax)
            if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"pion/muon cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
            ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
            ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n ACT3-5 After eveto and p removal", fontsize = 20)
            self.pdf_global.savefig(fig)

            plt.close()
            print("ACT35 left vs right plots have been made please check that they are sensible, protons should be in the bottom left corner")
            
        except:
            print("Please make the proton selection using tag_protons_TOF before checking the ACT35 left vs right plot to get the most of out it")
            return 0
        
        
        
        
        
    def tag_protons_TOF(self):
        '''Simple identification of the protons based on the time of flight, cutline fixed at 17.5ns for now, can be modified later if needed'''
        
        
        if self.run_momentum < 0:
            self.df["is_proton"] = False
            print(self.df["is_proton"], sum(self.df["is_proton"]==True))
            self.df["is_deuteron"] = False
        else:
             self.df["is_proton"] = np.where(self.df["tof"]>proton_tof_cut, self.df["tof"]<deuteron_tof_cut, False)
             self.df["is_deuteron"] = np.where(self.df["tof"]>deuteron_tof_cut, True, False)
                
        n_protons = sum(self.df["is_proton"]==True)
        n_deuteron = sum(self.df["is_deuteron"]==True)
        n_triggers = len(self.df["is_proton"])
        
        print(f"A total of {n_protons} protons and {n_deuteron} deuterons nuclei are tagged using the TOF out of {n_triggers}, i.e. {n_protons/n_triggers * 100:.1f}% of the dataset are protons and {n_deuteron/n_triggers * 100:.1f}% are deuteron")
        
    def tag_muons_pions_ACT35(self):
        '''Function to identify the muons and pions based on the charge deposited in ACT35, potentially using the muon tagger'''
        
        #step 1: find the optimal cut line in the muon tagger and decide if it is useful to implement it (that is, in case there are still some non-electrons left after the cut
        bins = np.linspace(120, 800, 100)
           
        
        #make sure they are boolean first
        self.df["is_proton"] = self.df["is_proton"].astype(bool)
        self.df["is_electron"] = self.df["is_electron"].astype(bool)
        self.df["is_deuteron"] = self.df["is_deuteron"].astype(bool)
        

        mu_tag_tot = self.df["mu_tag_l"]+self.df["mu_tag_r"]
        #cannot be any other particle already
        muons_pions = (~self.df["is_proton"]) & (~self.df["is_electron"])  & (~self.df["is_deuteron"])
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.hist(mu_tag_tot, bins = bins, label = 'All particles', histtype = "step")
        ax.hist(mu_tag_tot[self.df["is_electron"]], bins = bins, label = 'Electrons', histtype = "step")
        ax.hist(mu_tag_tot[self.df["is_proton"]], bins = bins, label = 'Protons', histtype = "step")
        h, _, _ = ax.hist(mu_tag_tot[muons_pions], bins = bins, label = 'Muons and pions', histtype = "step")
        ax.set_xlabel(f"Total charge in muon-tagger (QDC)", fontsize = 18)
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - Muon Tagger charge", fontsize = 20)


        #implement automatic muon tagger cut
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x_min = 150
        x_max = 300
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        # Find the bin index with the minimum count in that range
        min_index = np.argmin(h[mask])

        # Get the actual bin index in the original array
        index = np.where(mask)[0]
        mu_tag_cut = bin_centers[index[min_index]]
        
        #minimum fraction of muons and pions that have to be above the mu tag threshold (set to 0.5%)
        min_fraction_above_cut = 0.005
        
        n_electrons_above_cut = np.sum(self.df["is_electron"][mu_tag_tot>mu_tag_cut])
        n_muons_pions_above_cut = np.sum(muons_pions[mu_tag_tot>mu_tag_cut])
        self.n_muons_pions = np.sum(muons_pions)

        ax.axvline(mu_tag_cut, color = "k", linestyle = "--", label = f"Muon tagger cut: {mu_tag_cut:.1f} QDC \n {n_muons_pions_above_cut/self.n_muons_pions * 100:.1f}% of all muons and pions are above cut")
        ax.legend(fontsize = 16)

        ax.set_yscale("log")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
       
        print(f"The muon tagger charge has been plotted. The optimal cut line is at {mu_tag_cut:.1f} a.u., there are {n_electrons_above_cut} electrons above the cut line and {n_muons_pions_above_cut} muons and pions, i.e. {n_muons_pions_above_cut/self.n_muons_pions * 100:.1f}% of all muons and pions ({self.n_muons_pions})...")
        
        
        self.mu_tag_cut = mu_tag_cut
        
        bins = np.linspace(0, 80, 100)
        bins_act35 = np.linspace(0, 50, 100)
        
        if n_muons_pions_above_cut/self.n_muons_pions> min_fraction_above_cut:
            print(f"there are more than {min_fraction_above_cut * 100:.1f}% of muons and pions above the muon tagger cut, we are applying it. (Please verify this on the plots)")
            
            self.using_mu_tag_cut = True
                  
            #apply the cut on the muon tagger and then find the optimal ACT35 cut (based on pions and muons)
            fig, ax = plt.subplots(figsize = (8, 6))
            
            electron_above_mu_tag = (mu_tag_tot>mu_tag_cut) & (self.df["is_electron"])
            muons_pions_above_mu_tag = (mu_tag_tot>mu_tag_cut) & (muons_pions)
            
            h, _, _ = ax.hist(self.df["act_tagger"][muons_pions_above_mu_tag], bins = bins, label = "Muons/pions passing muon tagger cut", histtype = "step")
            ax.hist(self.df["act_tagger"][muons_pions], bins = bins, label = "All muons/pions", histtype = "step")
            
            
            ax.hist(self.df["act_tagger"][electron_above_mu_tag], 
                    bins = bins, label = "Electrons passing muon tagger cut", histtype = "step", color = "red")

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 0.5
            x_max = 10
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the minimum count in that range
            min_index = np.argmin(h[mask])

            # Get the actual bin index in the original array
            index = np.where(mask)[-1]
            self.act35_cut_pi_mu = bin_centers[index[min_index]]

            ax.axvline(self.act35_cut_pi_mu, label = f"pion/muon cut line: ACT3-5 = {self.act35_cut_pi_mu:.1f} PE", color = "k", linestyle = "--")
            ax.legend(fontsize = 12)
            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("ACT3-5 total charge (PE)", fontsize = 18)
            ax.set_yscale("log")
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - ACT3-5 with mu-tag cut", fontsize = 20)
            self.pdf_global.savefig(fig)
            plt.close()

            
        else:
            print(f"there are not more than {min_fraction_above_cut * 100:.1f}% of muons and pions above the muon tagger cut, we are not applying it. (Please verify this on the plots)" )#
            
            self.using_mu_tag_cut = False
            
            fig, ax = plt.subplots(figsize = (8, 6))

            
            h, _, _ = ax.hist(self.df["act_tagger"][muons_pions], bins = bins, label = "Muons and pions", histtype = "step")
            
#             print(sum(muons_pions))
            
            
            ax.hist(self.df["act_tagger"][self.df["is_electron"]], 
                    bins = bins, label = "Electrons", histtype = "step", color = "red")

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 2
            x_max = 40
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the minimum count in that range
            min_index = np.argmin(h[mask])

            # Get the actual bin index in the original array
            index = np.where(mask)[-1]
            self.act35_cut_pi_mu = bin_centers[index[min_index]]

            ax.axvline(self.act35_cut_pi_mu, label = f"pion/muon cut line: ACT3-5 = {self.act35_cut_pi_mu:.1f} PE", color = "k", linestyle = "--")
            ax.legend(fontsize = 12)
            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("ACT3-5 total charge (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - ACT3-5 without muon tagger cut", fontsize = 20)
            ax.set_yscale("log")
            self.pdf_global.savefig(fig)
            plt.close()
            
        #at the end check visually that things are ok  
        self.plot_ACT35_left_vs_right(self.act35_cut_pi_mu)
        self.df["is_muon"] = (~self.df["is_electron"]) & (~self.df["is_proton"]) & (~self.df["is_deuteron"]) & (self.df["act_tagger"]>self.act35_cut_pi_mu)
        self.df["is_pion"] = (~self.df["is_electron"]) & (~self.df["is_proton"])  & (~self.df["is_deuteron"]) & (self.df["act_tagger"]<=self.act35_cut_pi_mu)
        
        f_muons = sum(self.df["is_muon"])/self.n_muons_pions * 100
        f_pions = sum(self.df["is_pion"])/self.n_muons_pions * 100
        print(f"The pion/muon separtion cut line in ACT35 is {self.act35_cut_pi_mu:.1f}, out of {self.n_muons_pions:.1f} pions and muons, {f_muons:.1f}% are muons and {f_pions:.1f} are pions")
            
            
        
        
    def write_output_particles(self, particle_number_dict, store_PID_info, filename = None):
        """This functions writes out the WCTE tank information as well as the additional beam variables (TOF, ACT charges) necessary for making the selection, This function also stores the particle type guess obtained from the beam data but we encourage each analyser to develop their own selection"""
        
        
        if store_PID_info:
            #get the particle identification from the beam analysis
            index_mu = np.array(self.is_kept_event_id) * np.array(self.df["is_muon"])
            index_pi =  np.array(self.is_kept_event_id) * np.array(self.df["is_pion"])
            index_electron =  np.array(self.is_kept_event_id) * np.array(self.df["is_electron"])
            index_proton =  np.array(self.is_kept_event_id) * np.array(self.df["is_proton"])

            #Remove the events that are not of the given particle type
            index_mu = index_mu[index_mu !=0][:particle_number_dict["muon"]]
            index_pi = index_pi[index_pi !=0][:particle_number_dict["pion"]]
            index_electron = index_electron[index_electron !=0][:particle_number_dict["electron"]]
            index_proton = index_proton[index_proton !=0][:particle_number_dict["proton"]]
            

        # ---- inputs ------------------------------------------------------
        file_path = f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v0_5/{self.run_number}/WCTE_offline_R{self.run_number}S0_VME_matched.root"
        tree_name = "WCTEReadoutWindows"

        #these are the WCTE tank info branches
        branches = [
            "hit_pmt_calibrated_times",
            "window_data_quality",
            "hit_mpmt_card_ids", "hit_pmt_readout_mask",
            "hit_mpmt_slot_ids", "hit_pmt_position_ids",
            "hit_pmt_channel_ids", "hit_pmt_charges",
        ]

        BLOCK_MAX_EVENTS = 500
        WRITE_PARQUET   = True
        if filename == None:
             filename = f"Beam_PID_R{self.run_number}.parquet"
        PARQUET_FILE = filename

        # choose source
        local_copy = stage_local(file_path)  # "" if not staged
        file_for_uproot = local_copy or to_xrootd(file_path)

        # Combine all indices and particle labels
        
        if store_beam_PID:
            all_keep_idx = np.concatenate([
                index_electron,
                index_mu,
                index_pi,
                index_proton
            ])


            particle_labels = (
                ["electron"] * len(index_electron) +
                ["muon"] * len(index_mu) +
                ["pion"] * len(index_pi) +
                ["proton"] * len(index_proton)
            )
            
        else: #if we do not want to save the particle ID
            #only keep the data that is kept? no, all of the data would be better 
            all_keep_idx = self.is_kept_event_id[:particle_number_dict["triggers"]]
            
            #do not store the PID, just say that we keep them 
            particle_labels = (
                ["particle"] * len(all_keep_idx)
            )
            
            
            
            
        all_keep_idx = np.array(all_keep_idx)
        particle_labels = np.array(particle_labels)

        # Sort by index for block-wise reading
        sorted_order = np.argsort(all_keep_idx)
        all_keep_idx = all_keep_idx[sorted_order]
        particle_labels = particle_labels[sorted_order]

        # Open the file to get total entries
        with uproot.open(file_for_uproot) as f:
            tree = f[tree_name]
            n_entries = tree.num_entries

        # Keep only valid indices
        mask_valid = (all_keep_idx >= 0) & (all_keep_idx < n_entries)
        all_keep_idx = all_keep_idx[mask_valid]
        particle_labels = particle_labels[mask_valid]

        # Split into blocks
        blocks = make_blocks(all_keep_idx, BLOCK_MAX_EVENTS)
        print(f"{len(all_keep_idx)} selected entries → {len(blocks)} blocks (max {BLOCK_MAX_EVENTS} ev/block)")

        t0 = time.time()

        if WRITE_PARQUET:
            writer = None
            written = 0

            for (s, e) in blocks:
                # Read block
                with uproot.open(file_for_uproot) as f:
                    arr_block = f[tree_name].arrays(branches, library="ak", entry_start=s, entry_stop=e)

                # Select events in this block
                mask_block = (all_keep_idx >= s) & (all_keep_idx < e)
                if not np.any(mask_block):
                    del arr_block
                    gc.collect()
                    continue

                local_idx = all_keep_idx[mask_block] - s
                sel = arr_block[local_idx]

                # Add particle type column
                sel_particle = particle_labels[mask_block]
                sel = ak.with_field(sel, sel_particle, "beam_pid")    # Awkward Array

                # Convert to Arrow table
                tbl = ak.to_arrow_table(sel, list_to32=True, string_to32=True)

                # Add run number column
                run_arr = pa.array([self.run_number] * len(sel), type=pa.int32())
                tbl = tbl.append_column("run", run_arr)

                # Create Parquet writer on first block
                if writer is None:
                    meta = dict(tbl.schema.metadata or {})
                    meta.update({
                        b"wcte.run_number": str(self.run_number).encode(),
                        b"wcte.source_path": file_path.encode(),
                        b"wcte.tree_name":  tree_name.encode(),
                    })
                    schema_with_meta = tbl.schema.with_metadata(meta)
                    writer = pq.ParquetWriter(PARQUET_FILE, schema_with_meta, compression="snappy")

                writer.write_table(tbl)
                written += len(sel)

                # Clean memory
                del arr_block, sel, tbl, run_arr
                gc.collect()

            if writer is not None:
                writer.close()

            print(f"Wrote {written} rows to {PARQUET_FILE} in {time.time()-t0:.2f}s")
            
            
            
    def TOF_particle_in_ns(self, particle_name, momentum, L = 4.3431):
        ''' returns the TOF of particles of a given momentum'''
        momentum  = momentum #(give them in MeV/c)

        # masses in MeV/c^2
        masses = {
            "Electrons": 0.511,
            "Muons": 105.658,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123
        }

        if particle_name not in masses:
            raise ValueError(f"Unknown particle: {particle_name}")

        m = masses[particle_name]
        c = 2.99792458e8  # m/s

        # gamma and beta
        gamma = np.sqrt(1 + (momentum/m)**2)
        beta = np.sqrt(1 - 1/gamma**2)
        v = beta * c

        tof_seconds = L / v
        return tof_seconds * 1e9  # ns

      
    def return_losses(self, n_step, dist, particle_name, momentum, total_tof, total_length, psp, verbose = False):
        '''This function takes in the material that we are crossing, the associated energy lost table and the number of steps that we want to divide the crossing in and returns the time taken to travel the whole material and the total energy lost within it'''
        
        masses = { #MeV/c
            "Electrons": 0.511,
            "Muons": 105.658,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123
        }
        
        g4_energy = psp["#Kinetic_energy [GeV]"].to_numpy() * 1e3 

        
        for step in range(n_step):
            delta_L = dist/n_step
            delta_t = self.TOF_particle_in_ns(particle_name, momentum, delta_L)

            total_tof += delta_t
            total_length += delta_L


            #account for the momentum lost
            for i in range(len(momentum)):
                p = np.argmin(np.abs(g4_energy - momentum[i]))  # index of closest value to the g4 energy

                particle_kinetic_energy = np.sqrt(momentum[i]**2 + masses[particle_name]**2) - masses[particle_name] 


                #now modify the momentum 
                stoppingPower = (psp["Total_st_pw [MeV/m]"][p+1] - psp["Total_st_pw [MeV/m]"][p]) / (psp["#Kinetic_energy [GeV]"][p+1] - psp["#Kinetic_energy [GeV]"][p]) * (particle_kinetic_energy *10**(-3) - psp["#Kinetic_energy [GeV]"][p]) + psp["Total_st_pw [MeV/m]"][p]

                particle_kinetic_energy -= stoppingPower * delta_L

                momentum[i] = np.sqrt((particle_kinetic_energy + masses[particle_name])**2 - masses[particle_name]**2)

        return momentum, total_tof, total_length
    
    
    def give_tof(self, particle, initial_momentum, n_eveto_group, n_tagger_group, there_is_ACT5, run_monentum, T_pair = "T0T1", verbose = True):
        '''This function returns the T0-T1 TOF that a given particle would have as a function of its initial momentum, later to be compared with the recorded TOF for estimating the inital momentum. It is a stepper function that propagates the momentum at each step and adds up the total TOF, taking into account the momentum lost at each step'''
        #the default number of steps per material is 10 but we do modify it based on the material density
        #to speed up the process whilst keeping up the accuracy
        n_step = 10
        
     
    
        momentum = initial_momentum.copy()

        #Set up the ACTs with dimensions given by Sirous
        if (self.n_eveto == 1.01):
            ACT02_material = "1p01"
            ACT02_thick_per_box = 0.04 #cm

        elif (self.n_eveto == 1.03):
            ACT02_material = "1p03"
            ACT02_thick_per_box = 0.04 #cm
        elif (self.n_eveto == 1.075):
            ACT02_material = "1p075"
            ACT02_thick_per_box = 0.10/3 #cm
            
            
        



        if (self.n_tagger == 1.047):
            ACT35_material = "1p047"
            ACT35_thick_per_box = 0.16/3


        elif (self.n_tagger == 1.06):
            ACT35_material = "1p06"
            if not self.there_is_ACT5:
                ACT35_thick_per_box = 0.10/2 #cm (assume we take the two 6cm instead of one of each but i do not know for sure)

        elif (self.n_tagger == 1.015):
            ACT35_material = "1p015"
            if self.there_is_ACT5:
                ACT35_thick_per_box = 0.18/3

        elif (self.n_tagger == 1.03):
            ACT35_material = "1p03"
            if self.there_is_ACT5:
                ACT35_thick_per_box = 0.12/3 #cm
            else:
                ACT35_thick_per_box = 0.08/2 #cm
                
        elif (self.n_tagger == 1.15):
            ACT35_material = "1p15"
            ACT35_thick_per_box = 0.02 #cm
            if self.there_is_ACT5:
                print("In principle we only have one downstream n = 1.15 ACT, please check the run info")
                raise Error


        #store the total thickness plus 0.34 mm which is for the various reflective sheets and black tape
        ACT02_thickness = 3 * (ACT02_thick_per_box + 0.34e-3)


        if self.there_is_ACT5:
            ACT35_thickness = (ACT35_thick_per_box + 0.34e-3) * 3
        else:
            ACT35_thickness = (ACT35_thick_per_box + 0.34e-3) * 2

        #Trigger scintillators assumed to be all the same thickness, from Bruno's slides
        t0_thickness = 6.4e-3 #mm to m
        t4_thickness = 6.4e-3 #mm to m

        #Distance between the TS
        L =  444.03e-2  #T0 to T1
        L_t0t4 = 305.68e-2 
        L_t4t1 = 143.38e-2 
        
        t1_TOF_distance = 215.13e-2
        TOF_thickness = 6.4e-3
        beam_window_thickness = 1.2e-3 #https://wcte.hyperk.ca/wg/simulation-and-analysis/meetings/2024/20241122/meeting/simulation-update-beam-pipe-and-camera-housing/pipe_housing_wcsim_20241122.pdf/view


        #simplification, consider all the air as a single blob split in half before and after the aerogels
        L_air_T4_to_T1 = L_t4t1 - ACT02_thickness - ACT35_thickness - 0.06 - 0.003  
        #0.06 is the default value, 0.003 is the fine tune based on momentum measurements, see Alie's slides: https://wcte.hyperk.ca/wg/beam/meetings/2025/20250922/meeting/momentum-estimation-from-tof/acraplet_20250922_wctebeam.pdf


        #check what particle we have
        masses = { #MeV/c
            "Electrons": 0.511,
            "Muons": 105.658,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123
        }

        
        #Check run polarity
        if self.run_momentum < 0:
            if particle == "Electrons":
                p_name = "electron"
            if particle == "Muons":
                p_name = "muMinus"
            if particle == "Pions":
                p_name = "piMinus"
            if particle == "Proton":
                return 0

        elif self.run_momentum > 0:
            if particle == "Electrons":
                p_name = "positron"
            if particle == "Muons":
                p_name = "muPlus"
            if particle == "Pions":
                p_name = "piPlus"
            if particle == "Protons":
                p_name = "proton"
            if particle == "Deuteron":
                p_name = "deuteron"

        # Read in the theoretical losses from G4 tables provided by Arturo
        losses_dataset_air = f"../include/{p_name}StoppingPowerAirGeant4.csv"
        losses_dataset_plasticScintillator = f"../include/{p_name}StoppingPowerPlasticScintillatorGeant4.csv"
        losses_dataset_mylar = f"../include/{p_name}StoppingPowerMylarGeant4.csv"
        #ACT tables
        losses_dataset_upstream = f"../include/{p_name}StoppingPowerAerogel{ACT02_material}Geant4.csv"
        losses_dataset_downstream = f"../include/{p_name}StoppingPowerAerogel{ACT35_material}Geant4.csv"
   
        particle_name = particle

        #Open all the files
        with open(losses_dataset_air, mode = 'r') as file:
            psp_air = pd.read_csv(file) #psp = particle stopping power

        with open(losses_dataset_plasticScintillator, mode = 'r') as file:
            psp_plasticScintillator = pd.read_csv(file) 

        with open(losses_dataset_upstream, mode = 'r') as file:
            psp_upstreamACT = pd.read_csv(file)

        with open(losses_dataset_downstream, mode = 'r') as file:
            psp_downstreamACT = pd.read_csv(file)

        with open(losses_dataset_mylar, mode = 'r') as file:
            psp_mylar = pd.read_csv(file)



        if verbose: print(f"\n The initial momenta considered for the {p_name} are {momentum}") 

            
        #reset the time of flight and the total length, will do again laterbut necessary for the Mylar to have
        total_tof = np.zeros(len(momentum))
        total_length = 0
        
        
        #We look into the initial momenta as the particle exits the beam pipe 
        #need to take into account the energy lost to the Mylar window

        #Note that the TOF only starts as we start crossing T0, 
        #we however still need to account for the momentum lost 

        ###################  Exit Beam Pipe  ############################################# 
        #1 step is enough in the Mylar
        momentum, _, total_length = self.return_losses(1, 0.25e-6, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        if verbose: print(f"After crossing the mylar beam pipe window, the momentum is {momentum} and the TOF is {total_tof}")

        ##################    Cross T0    ###############################################
        #First half, 10 steps
        momentum, _, total_length = self.return_losses(10, t0_thickness/2, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)


        ################## beginning of TOF calculation
        #reset the time of flight and the total length
        total_tof = np.zeros(len(momentum))
        total_length = 0

        #second half, 25 steps
        momentum, _, total_length = self.return_losses(25, t0_thickness/2, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        #At the moment we do not add to the tof the half-crossing of T0 we might have to ? Have that 

        if verbose: print(f"After crossing the T0, the momentum is {momentum} and the TOF is {total_tof}")


    ###################    Air  between T0 and T4  ######################################### 
        momentum, total_tof, total_length = self.return_losses(100, L_t0t4, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
        if verbose: print(f"After crossing the air between T0 and T4, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

    ########################    Cross T4  #####################################
        momentum, total_tof, total_length = self.return_losses(35, t4_thickness, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After crossing the T4, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

    ####################    Half of the total air between T4 and T1  ########################
        momentum, total_tof, total_length = self.return_losses(25, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
        if verbose: print(f"After crossing a seventh the air gap between T4 and T1, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

    ###################    Cross ACT0  ####################
        #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT02_thick_per_box, particle_name, momentum, total_tof, total_length, psp_upstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT0, the  momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(n_step, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)

    #########################################    Cross ACT1  ################################################################## 

        #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT02_thick_per_box, particle_name, momentum, total_tof, total_length, psp_upstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        if verbose: print(f"After ACT1, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)

    #########################################    Cross ACT2  ################################################################## 

         #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT02_thick_per_box, particle_name, momentum, total_tof, total_length, psp_upstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT2, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)

    #########################################    Cross ACT3  ################################################################## 

         #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT35_thick_per_box, particle_name, momentum, total_tof, total_length, psp_downstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT3, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
    #########################################    Cross ACT4  ################################################################## 

         #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT35_thick_per_box, particle_name, momentum, total_tof, total_length, psp_downstreamACT, verbose = verbose)
        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT4, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
    #########################################    Cross ACT5  ################################################################## 


        if there_is_ACT5:
             #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
            momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

            #reflector of 0.02mm
            momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

            #aerogel
            momentum, total_tof, total_length = self.return_losses(25, ACT35_thick_per_box, particle_name, momentum, total_tof, total_length, psp_downstreamACT, verbose = verbose)
            if verbose: print(f"After ACT5, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")    
            #reflector of 0.02mm
            momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

            # 2 layers pf black sheet = 0.12mm 
            momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)


        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
    #     momentum, total_tof, total_length = return_losses(n_step, L_air_T4_to_T1/2, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
        if verbose: print(f"After crossing a seventh the air gap between T4 and T1, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        if verbose: print(f"The total travel length is {total_length}")
        if verbose: print(f"Initial momenta for {particle_name}: {initial_momentum}\n Final momenta: {momentum} \n TOF between T0 and T1: {total_tof}\n")
            
        new_tof = np.zeros(len(momentum))
            
        #continue the propagation, we are almost there to the beam window, useful to have for later extrapolation
        ############################## cross T1 #####################
        momentum, _, total_length = self.return_losses(20, t0_thickness, particle_name, momentum, new_tof, total_length, psp_plasticScintillator, verbose = verbose)
        
        
        ############################## cross air gap between T1 and TOF 
        momentum, _, total_length = self.return_losses(20, t1_TOF_distance, particle_name, momentum, new_tof, total_length, psp_air, verbose = verbose)
        
        
        ######################### cross TOF #############################
        momentum, _, total_length = self.return_losses(20, TOF_thickness, particle_name, momentum, new_tof, total_length, psp_plasticScintillator, verbose = verbose)
        
        ################ cross beam window into the tank
        momentum, _, total_length = self.return_losses(5, beam_window_thickness, particle_name, momentum, new_tof, total_length, psp_mylar, verbose = verbose)
        
        
        
        
        
            
        return initial_momentum, momentum, total_tof
     
        
        
    def extrapolate_momentum(self, initial_momentum, theoretical_tof, measured_tof, err_measured_tof):
        '''From the theoretical TOF and the measaured tof, extrapolate the value of the momentum with the associated error '''
        
        
        diff_m_exp = list(abs(theoretical_tof-measured_tof))
        A = diff_m_exp.index(min(diff_m_exp))
        B = diff_m_exp.index(min(diff_m_exp[A+1], diff_m_exp[A-1]))

        #simple linear extrapolation: m = ( y_b x_a - y_a x_b ) / (x_a-x_b) 
        x_a, y_a = initial_momentum[A], theoretical_tof[A]
        x_b, y_b = initial_momentum[B], theoretical_tof[B]

        intercept =  (y_b*x_a - y_a*x_b) / (x_a-x_b) 
        gradient = (y_a - intercept)/x_a
        
        
        momentum_guess = (measured_tof - intercept)/gradient
        momentum_minus = (measured_tof - err_measured_tof - intercept)/gradient
        momentum_plus = (measured_tof + err_measured_tof - intercept)/gradient
        err_mom = momentum_guess-momentum_plus
        
        return momentum_guess, err_mom
    
    
    def extrapolate_trigger_momentum(self, initial_momentum, theoretical_tof, measured_tof, err_measured_tof):
        '''From the theoretical TOF and the measaured tof, extrapolate the value of the momentum with the associated error, same version as abopve but working with arrays instead of single values, re-written with help from generative AI'''
        
        # Make sure they're numpy arrays
        initial_momentum = np.asarray(initial_momentum)
        theoretical_tof = np.asarray(theoretical_tof)

        # For each measured TOF, find closest index in theoretical_tof
        idx_closest = np.abs(theoretical_tof[:, None] - measured_tof).argmin(axis=0)

        # Pick neighbor for interpolation (next or previous)
        # use np.clip to avoid going out of bounds
        idx_neighbor = np.clip(idx_closest + 1, 0, len(theoretical_tof)-1)

        x_a = initial_momentum[idx_closest]
        y_a = theoretical_tof[idx_closest]
        x_b = initial_momentum[idx_neighbor]
        y_b = theoretical_tof[idx_neighbor]

        # linear interpolation parameters
        intercept = (y_b * x_a - y_a * x_b) / (x_a - x_b)
        gradient = (y_a - intercept) / x_a

        momentum_guess = (measured_tof - intercept) / gradient
        momentum_minus = (measured_tof - err_measured_tof - intercept) / gradient
        momentum_plus = (measured_tof + err_measured_tof - intercept) / gradient
        err_mom = np.abs(momentum_guess - momentum_plus)

        return momentum_guess, err_mom

            
    def estimate_momentum(self):
        '''This function is estimating the momentum for each trigger based on the pre-computed PID and the material budget in the beam line. It attemps at calculating errors as well'''
        
        momentum_points = np.linspace(0.6 * abs(self.run_momentum), 1.4 * abs(self.run_momentum),18)
        
        init_mom, pion_mom, pion_tof = self.give_tof( "Pions", momentum_points,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)
        
        init_mom, muon_mom, muon_tof = self.give_tof( "Muons", momentum_points,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)
        
        print("pion_mom", pion_mom)
        
        
        
        ##### Now extrapolate the mean momentum of muons and pion
        #Step 1 plot
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(momentum_points, pion_tof, color = "g", marker = "+", label = "Expected pion TOF")
        ax.plot(momentum_points, muon_tof, color = "orange", marker = "x", label = "Expected muon TOF")
        
        
        #read in the measured TOF
        pion_tof_m = self.particle_tof_mean["pion"]
        pion_tof_error = self.particle_tof_eom["pion"]
        muon_tof_m = self.particle_tof_mean["muon"]
        muon_tof_error = self.particle_tof_eom["muon"]
        
        #Plot the measured TOF
        ax.axhline(pion_tof_m, color = "g", linestyle = "--", label = f"Measured pions TOF: {pion_tof_m:.2f} +/- {pion_tof_error:.1e} ns (EOM)")
        ax.axhspan(pion_tof_m - pion_tof_error, pion_tof_m + pion_tof_error, color = "g", alpha = 0.2)
        
        ax.axhline(muon_tof_m, color = "orange", linestyle = "--", label = f"Measured muons TOF: {muon_tof_m:.2f} +/- {muon_tof_error:.1e} ns (EOM)")
        ax.axhspan(muon_tof_m - muon_tof_error, muon_tof_m + muon_tof_error, color = "orange", alpha = 0.2)

        #Extrapolate each momenta at the start !
        mean_mom_pion, err_mom_pion = self.extrapolate_momentum(momentum_points, pion_tof, pion_tof_m, pion_tof_error)
        mean_mom_muon, err_mom_muon = self.extrapolate_momentum(momentum_points, muon_tof, muon_tof_m, muon_tof_error)
        
        
        mean_mom_pion_final, err_mom_pion_final = self.extrapolate_momentum(pion_mom, pion_tof, pion_tof_m, pion_tof_error)
        mean_mom_muon_final, err_mom_muon_final = self.extrapolate_momentum(muon_mom, muon_tof, muon_tof_m, muon_tof_error)

        
         #plot the extrapolated momenta
        ax.axvline(mean_mom_pion, color = "green", linestyle = "-.", label = f"Estimated mean pion momentum: {mean_mom_pion:.2f} +/- {err_mom_pion:.1f} MeV/c")
        ax.axvspan(mean_mom_pion - err_mom_pion, mean_mom_pion + err_mom_pion, color = "green", alpha = 0.2)
        ax.axvline(mean_mom_muon, color = "orange", linestyle = "-.", label = f"Estimated mean muon momentum: {mean_mom_muon:.2f} +/- {err_mom_muon:.1f} MeV/c")
        ax.axvspan(mean_mom_muon - err_mom_muon, mean_mom_muon + err_mom_muon, color = "orange", alpha = 0.2)
        
        #make a clean plot
        ax.set_ylabel("TOF (ns)", fontsize = 18)
        ax.set_xlabel("Initial momentum", fontsize = 18)
        ax.legend(fontsize = 12)
        ax.grid()
        ax.set_ylim(min(muon_tof) * 0.98,max(pion_tof) * 1.04)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

        self.pdf_global.savefig(fig)
        plt.close()
        
        
        self.particle_mom_mean = {
            "electron": 0,
            "muon": mean_mom_muon,
            "pion": mean_mom_pion,
            "proton": 0,
            "deuteron":0
            
        }
        
        self.particle_mom_mean_err = {
            "electron": 0,
            "muon": err_mom_muon,
            "pion": err_mom_pion,
            "proton": 0,
            "deuteron":0
        }
        
        
        self.particle_mom_final_mean = {
            "electron": 0,
            "muon": mean_mom_muon_final,
            "pion": mean_mom_pion_final,
            "proton": 0,
            "deuteron":0
            
        }
        
        self.particle_mom_final_mean_err = {
            "electron": 0,
            "muon": err_mom_muon_final,
            "pion": err_mom_pion_final,
            "proton": 0,
            "deuteron":0
        }
     
    

        if sum(self.df["is_proton"]==True) > 100 :
            init_mom, proton_mom, proton_tof = self.give_tof( "Protons", momentum_points,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)
            
        

            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(momentum_points, proton_tof, color = "r", marker = "+", label = "Expected proton TOF")

            #read in the measured TOF
            proton_tof_m = self.particle_tof_mean["proton"]
            proton_tof_error = self.particle_tof_eom["proton"]


            #Plot the measured TOF
            ax.axhline(proton_tof_m, color = "r", linestyle = "--", label = f"Measured proton TOF: {proton_tof_m:.2f} +/- {proton_tof_error:.1e} ns (EOM)")
            ax.axhspan(proton_tof_m - proton_tof_error, proton_tof_m + proton_tof_error, color = "g", alpha = 0.2)


            #Extrapolate each momenta
            mean_mom_proton, err_mom_proton = self.extrapolate_momentum(momentum_points, proton_tof, proton_tof_m, proton_tof_error)
            
            #we consider that we have the same error at the end than at the begining
            mean_mom_proton_final, err_mom_proton_final = self.extrapolate_momentum(proton_mom, proton_tof, proton_tof_m, proton_tof_error)


             #plot the extrapolated momenta
            ax.axvline(mean_mom_proton, color = "red", linestyle = "-.", label = f"Estimated mean proton momentum: {mean_mom_proton:.2f} +/- {err_mom_proton:.1f} MeV/c")
            ax.axvspan(mean_mom_proton - err_mom_proton, mean_mom_proton + err_mom_proton, color = "red", alpha = 0.2)

            #make a clean plot
            ax.set_ylabel("TOF (ns)", fontsize = 18)
            ax.set_xlabel("Initial momentum", fontsize = 18)
            ax.legend(fontsize = 12)
            ax.grid()
            ax.set_ylim(min(proton_tof) * 0.98,max(proton_tof) * 1.04)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

            self.pdf_global.savefig(fig)
            plt.close()

            self.particle_mom_mean["proton"] = mean_mom_proton
            self.particle_mom_mean_err["proton"] = err_mom_proton
            
            self.particle_mom_final_mean["proton"] = mean_mom_proton_final
            self.particle_mom_final_mean_err["proton"] = err_mom_proton_final
            
            
            
            
        
        if sum(self.df["is_deuteron"]==True) > 100 :
            
            #we cannot go too low momentum, otherwise unphysical... 
            momentum_points_deuteron = np.linspace(0.75 * abs(self.run_momentum), 1.4 * abs(self.run_momentum),12)
            
            init_mom, deuteron_mom, deuteron_tof = self.give_tof( "Deuteron", momentum_points_deuteron,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)
        
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(momentum_points_deuteron, deuteron_tof, color = "k", marker = "+", label = "Expected deuteron TOF")

            #read in the measured TOF
            deuteron_tof_m = self.particle_tof_mean["deuteron"]
            deuteron_tof_error = self.particle_tof_eom["deuteron"]


            #Plot the measured TOF
            ax.axhline(deuteron_tof_m, color = "k", linestyle = "--", label = f"Measured deuteron TOF: {deuteron_tof_m:.2f} +/- {deuteron_tof_error:.1e} ns (EOM)")
            ax.axhspan(deuteron_tof_m - deuteron_tof_error, deuteron_tof_m + deuteron_tof_error, color = "g", alpha = 0.2)


            #Extrapolate each momenta
            mean_mom_deuteron, err_mom_deuteron = self.extrapolate_momentum(momentum_points_deuteron, deuteron_tof, deuteron_tof_m, deuteron_tof_error)
            mean_mom_deuteron_final, err_mom_deuteron_final = self.extrapolate_momentum(deuteron_mom, deuteron_tof, deuteron_tof_m, deuteron_tof_error)


             #plot the extrapolated momenta
            ax.axvline(mean_mom_deuteron, color = "black", linestyle = "-.", label = f"Estimated mean deuteron momentum: {mean_mom_deuteron:.2f} +/- {err_mom_deuteron:.1f} MeV/c")
            ax.axvspan(mean_mom_deuteron - err_mom_deuteron, mean_mom_deuteron + err_mom_deuteron, color = "black", alpha = 0.2)

            #make a clean plot
            ax.set_ylabel("TOF (ns)", fontsize = 18)
            ax.set_xlabel("Initial momentum", fontsize = 18)
            ax.legend(fontsize = 12)
            ax.grid()
            ax.set_ylim(min(deuteron_tof) * 0.98,max(deuteron_tof) * 1.04)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

            self.pdf_global.savefig(fig)
            plt.close()

            self.particle_mom_mean["deuteron"] = mean_mom_deuteron
            self.particle_mom_mean_err["deuteron"] = err_mom_deuteron
            
            self.particle_mom_final_mean["deuteron"] = mean_mom_deuteron_final
            self.particle_mom_final_mean_err["deuteron"] = err_mom_deuteron_final

        print(f"The estimated mean particle momenta are {self.particle_mom_mean} MeV/c with an error {self.particle_mom_mean_err} MeV/c")
        
        
        ######### Now calculate the momentum for each trigger individually
        #The error on the tof is taken as the width of the electron tof distribution 
        
        self.df["initial_momentum"] = np.nan  # initialize column
        self.df["initial_momentum_error"] = np.nan  # initialize column

        mask_muon = self.df["is_muon"] == 1
        mask_pion = self.df["is_pion"] == 1
        mask_proton = self.df["is_proton"] == 1
        mask_deuteron = self.df["is_deuteron"] == 1
        
        #this is the error assumed on each TOF
        electron_tof_std = self.particle_tof_std["electron"]

        #Store the correct trigger TOFs, don't forget to use the corrected one, 
        #otherwise non-physical results due to offset of electron TOF with L/c
        trigger_muon_tof = self.df.loc[mask_muon, "tof_corr"].to_numpy()
        trigger_pion_tof = self.df.loc[mask_pion, "tof_corr"].to_numpy()
        trigger_proton_tof = self.df.loc[mask_proton, "tof_corr"].to_numpy()
        trigger_deuteron_tof = self.df.loc[mask_deuteron, "tof_corr"].to_numpy()
        
        #now compute the momentum for each trigger
        mom, mom_err = self.extrapolate_trigger_momentum(momentum_points, muon_tof, trigger_muon_tof, electron_tof_std)
        self.df.loc[mask_muon, "initial_momentum"] = mom
        self.df.loc[mask_muon, "initial_momentum_error"] = mom_err
        
        
        mom, mom_err = self.extrapolate_trigger_momentum(muon_mom, muon_tof, trigger_muon_tof, electron_tof_std)
        self.df.loc[mask_muon, "final_momentum"] = mom
        self.df.loc[mask_muon, "final_momentum_error"] = mom_err
        
        
        
        mom, mom_err = self.extrapolate_trigger_momentum(momentum_points, pion_tof, trigger_pion_tof, electron_tof_std)
        self.df.loc[mask_pion, "initial_momentum"] = mom
        self.df.loc[mask_pion, "initial_momentum_error"] = mom_err
        
        
        mom, mom_err = self.extrapolate_trigger_momentum(pion_mom, pion_tof, trigger_pion_tof, electron_tof_std)
        self.df.loc[mask_pion, "final_momentum"] = mom
        self.df.loc[mask_pion, "final_momentum_error"] = mom_err

        
        if sum(self.df["is_proton"]) > 100:
            mom, mom_err = self.extrapolate_trigger_momentum(momentum_points, proton_tof, trigger_proton_tof, electron_tof_std)

            self.df.loc[mask_proton, "initial_momentum"] = mom
            self.df.loc[mask_proton, "initial_momentum_error"] = mom_err
            
            mom, mom_err = self.extrapolate_trigger_momentum(proton_mom, proton_tof, trigger_proton_tof, electron_tof_std)

            self.df.loc[mask_proton, "final_momentum"] = mom
            self.df.loc[mask_proton, "final_momentum_error"] = mom_err
            
        if sum(self.df["is_deuteron"]) > 100:
            mom, mom_err = self.extrapolate_trigger_momentum(momentum_points_deuteron, deuteron_tof, trigger_deuteron_tof, electron_tof_std)

            self.df.loc[mask_deuteron, "initial_momentum"] = mom
            self.df.loc[mask_deuteron, "initial_momentum_error"] = mom_err
            
            
            mom, mom_err = self.extrapolate_trigger_momentum(deuteron_mom, deuteron_tof, trigger_deuteron_tof, electron_tof_std)

            self.df.loc[mask_deuteron, "final_momentum"] = mom
            self.df.loc[mask_deuteron, "final_momentum_error"] = mom_err
            
            
            
        print("\n \n", self.df.head())
        
        #### Next, plot the reconstructed momenta (without error for each of the triggers, this will help chekch the overlap)
        
        fig, ax = plt.subplots(figsize = (8, 6))
        momentum_bins = np.linspace(min(momentum_points), max(momentum_points), 100)
        
        n_mu = sum(self.df["is_muon"])
        n_pi = sum(self.df["is_pion"])
        n_p = sum(self.df["is_proton"])
        n_D = sum(self.df["is_deuteron"])
        
        
        f_mu = n_mu/len(self.df["is_muon"])
        f_pi = n_pi/len(self.df["is_pion"])
        
        
        #muons
        ax.hist(self.df["initial_momentum"][self.df["is_muon"] == 1], 
                bins = momentum_bins, color = "orange", 
                histtype = "step",
                label = f"Muons {n_mu} triggers ({f_mu * 100:.1f}% of particles) p = {mean_mom_muon:.2f} +/- {err_mom_muon:.1f} MeV/c")
        
        ax.axvline(self.particle_mom_mean["muon"], color = "orange", linestyle = "-.")
        ax.axvspan(self.particle_mom_mean["muon"] - self.particle_mom_mean_err["muon"], self.particle_mom_mean["muon"] + self.particle_mom_mean_err["muon"], color = "orange", alpha = 0.2)
        
        
        #Pions
        ax.hist(
            self.df["initial_momentum"][self.df["is_pion"] == 1],
            bins=momentum_bins,
            color="blue",
            histtype="step",
            label=f"Pions {n_pi} triggers ({f_pi * 100:.1f}% of particles) p = {self.particle_mom_mean['pion']:.2f} +/- {self.particle_mom_mean_err['pion']:.1f} MeV/c"
        )
        ax.axvline(
            self.particle_mom_mean["pion"],
            color="blue",
            linestyle="-.",
           
        )
        ax.axvspan(
            self.particle_mom_mean["pion"] - self.particle_mom_mean_err["pion"],
            self.particle_mom_mean["pion"] + self.particle_mom_mean_err["pion"],
            color="blue",
            alpha=0.2
        )
        
        
        #Protons:
        if sum(self.df["is_proton"]) > 100:
            f_p = n_p/len(self.df["is_proton"])
            ax.hist(
                self.df["initial_momentum"][self.df["is_proton"] == 1],
                bins=momentum_bins,
                color="red",
                histtype="step",
                label=f"Protons: {n_p} triggers ({f_p * 100:.1f}% of particles) p = {self.particle_mom_mean['proton']:.2f} +/- {self.particle_mom_mean_err['proton']:.1f} MeV/c"
            )
            ax.axvline(
                self.particle_mom_mean["proton"],
                color="red",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_mean["proton"] - self.particle_mom_mean_err["proton"],
                self.particle_mom_mean["proton"] + self.particle_mom_mean_err["proton"],
                color="red",
                alpha=0.2
            )
            
            
        #Deuterons:
        if sum(self.df["is_deuteron"]) > 100:
            f_D = n_D/len(self.df["is_deuteron"])
            ax.hist(
                self.df["initial_momentum"][self.df["is_deuteron"] == 1],
                bins=momentum_bins,
                color="black",
                histtype="step",
                label=f"Deuterons: {n_D} triggers ({f_D * 100:.1f}% of particles) p = {self.particle_mom_mean['deuteron']:.2f} +/- {self.particle_mom_mean_err['deuteron']:.1f} MeV/c"
            )
            
            print("Reconstructed momenta for deuterons", self.df["initial_momentum"][self.df["is_deuteron"] == 1])
            ax.axvline(
                self.particle_mom_mean["deuteron"],
                color="black",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_mean["deuteron"] - self.particle_mom_mean_err["deuteron"],
                self.particle_mom_mean["deuteron"] + self.particle_mom_mean_err["deuteron"],
                color="black",
                alpha=0.2
            )
        
        #Add a caption to the figure
        plt.figtext(0.5, -0.05, f"Mean momentum for each particle type is obtained from mean TOF and error on mean (EOM).", 
                    ha="center", va="top", fontsize=10)
       
        
        #make a clean plot
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Initial momentum (MeV/c)", fontsize = 18)
        ax.legend(fontsize = 12)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, len(self.df["is_electron"]) * 20)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
        ######## Momentum after beam window
        
        fig, ax = plt.subplots(figsize = (8, 6))
        #muons
        ax.hist(self.df["final_momentum"][self.df["is_muon"] == 1], 
                bins = momentum_bins, color = "orange", 
                histtype = "step",
                label = f"Muons {n_mu} triggers ({f_mu * 100:.1f}% of particles) p = {mean_mom_muon_final:.2f} +/- {err_mom_muon_final:.1f} MeV/c")
        
        ax.axvline(self.particle_mom_final_mean["muon"], color = "orange", linestyle = "-.")
        ax.axvspan(self.particle_mom_final_mean["muon"] - self.particle_mom_final_mean_err["muon"], self.particle_mom_final_mean["muon"] + self.particle_mom_final_mean_err["muon"], color = "orange", alpha = 0.2)
        
        
        #Pions
        ax.hist(
            self.df["final_momentum"][self.df["is_pion"] == 1],
            bins=momentum_bins,
            color="blue",
            histtype="step",
            label=f"Pions {n_pi} triggers ({f_pi * 100:.1f}% of particles) p = {self.particle_mom_final_mean['pion']:.2f} +/- {self.particle_mom_final_mean_err['pion']:.1f} MeV/c"
        )
        ax.axvline(
            self.particle_mom_final_mean["pion"],
            color="blue",
            linestyle="-.",
           
        )
        ax.axvspan(
            self.particle_mom_final_mean["pion"] - self.particle_mom_final_mean_err["pion"],
            self.particle_mom_final_mean["pion"] + self.particle_mom_final_mean_err["pion"],
            color="blue",
            alpha=0.2
        )
        
        
        #Protons:
        if sum(self.df["is_proton"]) > 100:
            f_p = n_p/len(self.df["is_proton"])
            ax.hist(
                self.df["final_momentum"][self.df["is_proton"] == 1],
                bins=momentum_bins,
                color="red",
                histtype="step",
                label=f"Protons: {n_p} triggers ({f_p * 100:.1f}% of particles) p = {self.particle_mom_final_mean['proton']:.2f} +/- {self.particle_mom_final_mean_err['proton']:.1f} MeV/c"
            )
            ax.axvline(
                self.particle_mom_final_mean["proton"],
                color="red",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_final_mean["proton"] - self.particle_mom_final_mean_err["proton"],
                self.particle_mom_final_mean["proton"] + self.particle_mom_final_mean_err["proton"],
                color="red",
                alpha=0.2
            )
            
            
        #Deuterons:
        if sum(self.df["is_deuteron"]) > 100:
            f_D = n_D/len(self.df["is_deuteron"])
            ax.hist(
                self.df["final_momentum"][self.df["is_deuteron"] == 1],
                bins=momentum_bins,
                color="black",
                histtype="step",
                label=f"Deuterons: {n_D} triggers ({f_D * 100:.1f}% of particles) p = {self.particle_mom_final_mean['deuteron']:.2f} +/- {self.particle_mom_final_mean_err['deuteron']:.1f} MeV/c"
            )
            
            #print("Reconstructed momenta for deuterons", self.df["initial_momentum"][self.df["is_deuteron"] == 1])
            ax.axvline(
                self.particle_mom_final_mean["deuteron"],
                color="black",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_final_mean["deuteron"] - self.particle_mom_final_mean_err["deuteron"],
                self.particle_mom_final_mean["deuteron"] + self.particle_mom_final_mean_err["deuteron"],
                color="black",
                alpha=0.2
            )
        
        #Add a caption to the figure
        plt.figtext(0.5, -0.05, f"Mean momentum for each particle type is obtained from mean TOF and error on mean (EOM).", 
                    ha="center", va="top", fontsize=10)
       
        
        #make a clean plot
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Momentum after beam window (MeV/c)", fontsize = 18)
        ax.legend(fontsize = 12)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, len(self.df["is_electron"]) * 20)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

        self.pdf_global.savefig(fig)
        plt.close()
        
        
        

        
        
        
        
            
            
            
            
    def measure_particle_TOF(self):
        '''Measure the TOF for each of the particles accounting for any offsets between the electron TOF and L/c'''
        
        there_is_proton = True
        #Define the bounds inside which we will attempt the fits 
        if self.run_momentum > 600:
            times_of_flight_min = [ 12, 5, -70]
            times_of_flight_max = [45, 50, 70 ]

        elif self.run_momentum > 300:
            times_of_flight_min = [ 12, 5, -70]
            times_of_flight_max = [26, 50, 70 ]

        else:
            there_is_proton = False
            times_of_flight_min = [ 12, 5, -70]
            times_of_flight_max = [18, 20, 70 ]
            
            
            
        ##### First do T0-T1 ###########
        time_of_flight = self.df["tof"]
        
        #Define the bins
        bins_tof = np.linspace(times_of_flight_min[0], times_of_flight_max[0], 200)
        bin_centers = (bins_tof[1:] + bins_tof[:-1])/2
        
        #Fit the electron TOF
        electron_tof = time_of_flight[self.df["is_electron"] == 1]
        h, _ = np.histogram(electron_tof, bins = bins_tof)
        
    
        popt, pcov = fit_gaussian(h, bin_centers)
        #L is in cm, c is in m.ns^-1
        t0 = popt[1] - L/(c * 10**2) #convert m.ns^-1 into cm.ns^-1
        
        print(f"The time difference between the reconstructed electron TOF and L/c = {L/(c * 10**2):.2f} is {t0:.2f} ns")
        
        #Correct the TOF by this offset, decide to make a new column, cleaner
        self.df["tof_corr"] = self.df["tof"] - t0
        
        #Check TOF for each particle type
        h_mu, _ = np.histogram(self.df["tof_corr"][self.df["is_muon"]==1], bins = bins_tof)
        popt_mu, pcov = fit_gaussian(h_mu, bin_centers)
        
        h_pi, _ = np.histogram(self.df["tof_corr"][self.df["is_pion"]==1], bins = bins_tof)
        popt_pi, pcov = fit_gaussian(h_pi, bin_centers)
        
        if there_is_proton:
            h_p, _ = np.histogram(self.df["tof_corr"][self.df["is_proton"]==1], bins = bins_tof)
            popt_p, pcov = fit_gaussian(h_p, bin_centers)
            
        if sum(self.df["is_deuteron"])>100:
            h_D, _ = np.histogram(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof)
            popt_D, pcov = fit_gaussian(h_D, bin_centers)
            
            
            
        #Here, plot the TOF 
        fig, ax = plt.subplots(figsize = (8, 6))
        
        #plot the distributions
        ax.hist(self.df["tof_corr"][self.df["is_electron"]==1], bins = bins_tof, histtype = "step", label = f"Electrons: p = {popt[1]:.2f} "+ r"$\pm$"+ f" {popt[2]:.2f} ns")  
        ax.hist(self.df["tof_corr"][self.df["is_muon"]==1], bins = bins_tof, histtype = "step", label = f"Muons: p = {popt_mu[1]:.2f} "+ r"$\pm$"+ f" {popt_mu[2]:.2f} ns")
        ax.hist(self.df["tof_corr"][self.df["is_pion"]==1], bins = bins_tof, histtype = "step", label = f"Pions: p = {popt_pi[1]:.2f} "+ r"$\pm$"+ f" {popt_pi[2]:.2f} ns")
        
        
        if there_is_proton:
            ax.hist(self.df["tof_corr"][self.df["is_proton"]==1], bins = bins_tof, histtype = "step", label = f"Protons: p = {popt_p[1]:.2f} "+ r"$\pm$"+ f" {popt_p[2]:.2f} ns")
            
        if sum(self.df["is_deuteron"])>100:
            ax.hist(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof, histtype = "step", label = f"Deuterons: p = {popt_D[1]:.2f} "+ r"$\pm$"+ f" {popt_D[2]:.2f} ns")
          
            
            
        
        
        #plot the fits, for visual inspection
        
        ax.plot(bins_tof, gaussian(bins_tof, popt[0], popt[1]-t0, popt[2]), "--", color = "k")
        
        for p in [popt_mu, popt_pi]:
            ax.plot(bins_tof, gaussian(bins_tof, p[0], p[1], p[2]), "--", color = "k")
        
        
        if there_is_proton:
            ax.plot(bins_tof, gaussian(bins_tof, popt_p[0], popt_p[1], popt_p[2]), "--", color = "k")
            
        if sum(self.df["is_deuteron"])>100:
            ax.plot(bins_tof, gaussian(bins_tof, popt_D[0], popt_D[1], popt_D[2]), "--", color = "k")
            
            
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Time of flight (ns)", fontsize = 18)
        ax.legend(fontsize = 10)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, 5e5)
        ax.set_title(f"Run {self.run_number} T0-T1 TOF ({self.run_momentum} MeV/c)", fontsize = 20)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        #Here save the mean TOF and std for each particle population
        self.particle_tof_mean = {
            "electron": popt[1],
            "muon": popt_mu[1],
            "pion": popt_pi[1],
            "proton": popt_p[1] if there_is_proton else 0,
            "deuteron": popt_D[1] if sum(self.df["is_deuteron"])>100 else 0,
        }
        
        self.particle_tof_std = {
            "electron": popt[2],
            "muon": popt_mu[2],
            "pion": popt_pi[2],
            "proton": popt_p[2] if there_is_proton else 0,
            "deuteron": popt_D[2] if sum(self.df["is_deuteron"])>100 else 0,
            
        }
        
        self.particle_tof_eom = {
            "electron": popt[2]/np.sqrt(sum(self.df["is_electron"])),
            "muon": popt_mu[2]/np.sqrt(sum(self.df["is_muon"])),
            "pion": popt_pi[2]/np.sqrt(sum(self.df["is_pion"])),
            "proton": popt_p[2]/np.sqrt(sum(self.df["is_proton"])) if there_is_proton else 0,
            "deuteron": popt_D[2]/np.sqrt(sum(self.df["is_deuteron"])) if sum(self.df["is_deuteron"])>100 else 0,
            
        }
        
        
    def plot_TOF_charge_distribution(self):
        '''Check visually the total charge deposited in the TOF detector, can be handy to identify events that do not actually cross the TOF'''
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        
        bins = np.linspace( min(self.df_all["total_TOF_charge"]), max(self.df_all["total_TOF_charge"]), 100)
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_electron"] == 1], bins = bins, label = 'electron', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_muon"] == 1], bins = bins, label = 'muon', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_pion"] == 1], bins = bins, label = 'pion', histtype = "step")

        if sum(self.df["is_proton"] == 1)>100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_proton"] == 1], bins = bins, label = 'proton', histtype = "step")
        if sum(self.df["is_deuteron"] == 1) >100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_deuteron"] == 1], bins = bins, label = 'deuteron', histtype = "step")
        _ = ax.hist(self.df_all["total_TOF_charge"][self.df_all["is_kept"] == 0], bins = bins, label = 'Triggers not kept for analysis', color = "k", histtype = "step")

        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Total charge in TOF detector (a.u.)")
        self.pdf_global.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(figsize = (8, 6))
        bins = np.linspace( min(self.df["total_TOF_charge"]), max(self.df["total_TOF_charge"]), 100)
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_electron"] == 1], bins = bins, label = 'electron', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_muon"] == 1], bins = bins, label = 'muon', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_pion"] == 1], bins = bins, label = 'pion', histtype = "step")

        if sum(self.df["is_proton"] == 1)>100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_proton"] == 1], bins = bins, label = 'proton', histtype = "step")
        if sum(self.df["is_deuteron"] == 1) >100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_deuteron"] == 1], bins = bins, label = 'deuteron', histtype = "step")

        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Total charge in TOF detector (a.u.)")
        self.pdf_global.savefig(fig)
        plt.close()
        
                   
    def output_beam_ana_to_root(self, output_name = None):
        ''''Output the results of the beam analysis as a root file with three branches, the 1D run information (number, nominal momentum, refractive index, whether ACT5 is in the beam line), the 1D results (mean measured [T0T1] TOF and momentum for each particle type with EOM and std for the TOF only), number of triggers kept, total number of triggers'''
        if output_name == None:
            output_name = f"beam_analysis_output_R{self.run_number}.root"
            
            
        for col in self.df.columns:
            if col not in self.df_all.columns:
                self.df_all[col] = np.nan  # create empty column first if you want aligned length
                self.df_all.loc[self.df_all["is_kept"], col] = self.df[col].values

                
                
        for is_particle in ["is_muon", "is_electron", "is_pion", "is_proton", "is_deuteron"]:
            self.df_all[is_particle] = self.df_all[is_particle].fillna(0)
            self.df_all[is_particle] = self.df_all[is_particle].astype(np.int32)

        
        
        self.df_all["is_kept"] = self.df_all["is_kept"].astype(np.int32)
        

       # --- Convert DataFrame to dictionary of numpy arrays ---
        branches = {col: self.df_all[col].to_numpy() for col in self.df_all.columns}

        # --- Create ROOT file and save the tree ---
        with uproot.recreate(output_name) as f:
            # Write the main TTree
            f["beam_analysis"] = branches

            # Write scalar metadata as a separate tree (recommended)
            f["run_info"] = {
                "run_number": np.array([self.run_number], dtype=np.int32),
                "run_momentum": np.array([self.run_momentum], dtype=np.float64),          
                "n_eveto": np.array([self.n_eveto], dtype=np.float64),
                "n_tagger": np.array([self.n_tagger], dtype=np.float64),
                "there_is_ACT5":np.array([self.there_is_ACT5], dtype = np.int32),

            }


            #save as a separate branch the 1d results of interest

            results = {
                "act_eveto_cut":np.array([self.eveto_cut], dtype=np.float64),
                "act_tagger_cut":np.array([self.act35_cut_pi_mu], dtype=np.float64),
                "proton_tof_cut":np.array([proton_tof_cut], dtype=np.float64),
                "deuteron_tof_cut":np.array([deuteron_tof_cut], dtype=np.float64),
                "mu_tag_cut": np.array([self.mu_tag_cut], dtype=np.float64),
                "using_mu_tag_cut": np.array([self.using_mu_tag_cut], dtype=np.float64),
            }
            for prefix, d in [("tof_mean", self.particle_tof_mean),
                  ("tof_std", self.particle_tof_std),
                  ("tof_eom", self.particle_tof_eom),
                  ("momentum_mean", self.particle_mom_mean),
                  ("momentum_eom", self.particle_mom_mean_err),
                  ("momentum_after_beam_window_mean", self.particle_mom_mean),
                  ("momentum_after_beam_window_eom", self.particle_mom_mean_err)]:
                for key, value in d.items():
                    results[f"{prefix}_{key}"] = np.array([value], dtype=np.float64)

                    
            results["n_electrons"] = np.array([sum(self.df_all["is_electron"]) ], dtype=np.float64) 
            results["n_muons"] =  np.array([sum(self.df_all["is_muon"])], dtype=np.float64) 
            results["n_pions"] =  np.array([sum(self.df_all["is_pion"])], dtype=np.float64)   
            results["n_protons"] =  np.array([sum(self.df_all["is_proton"])], dtype=np.float64)   
            results["n_deuterons"] =   np.array([sum(self.df_all["is_deuteron"])], dtype=np.float64)   
            results["n_triggers_kept"] =  np.array([sum(self.df_all["is_kept"])], dtype=np.float64) 
            
            results["n_triggers_total"] =  np.array([len(self.df_all["is_proton"]) ], dtype=np.float64)  
            
            
            f["scalar_results"] = results 
            
            print(f"Saved output file to {output_name}")

            
           


        
            
            
            
            
            
            
            
            
           
    
    
 


        
            
            
       
        





        
        
        
        
