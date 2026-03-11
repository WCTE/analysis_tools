#!/usr/bin/env python
# coding: utf-8

# ## PID using the beam monitors
# This code is the default way of identifying particles using the beam monitor. It is providing only a template event selection that is not optimised for any Physics analyses. It should serve as an exmple around which to build your own beam particle identification code. 

# In[3]:


#Step 0, import libraries
import numpy as np
import importlib

import os, sys
path_to_tools = os.path.abspath(os.path.join('../', 'analysis_tools'))

if path_to_tools not in sys.path:
    sys.path.append(path_to_tools)

#this is the file with the necessary functions for performing PID 
import beam_monitors_pid as bm


# In[4]:


#Step 1, read in the data 

#### Example 1: medium momentum negative polarity
# run_number = 1478
# run_momentum = -410
# n_eveto_group = 1.01 #refractive index of ACT0-2
# n_tagger_group = 1.06 # of ACT3-5
# there_is_ACT5 = False  #important to keep track of whether there is ACT5 in the run  


### Example 2: relatively high momentum, positive polarity
run_number = 1610
run_momentum = 760
n_eveto_group = 1.01
n_tagger_group = 1.015
there_is_ACT5 = True

##### Example 2.5: relatively high momentum, positive polarity
# run_number = 1602
# run_momentum = 770
# n_eveto_group = 1.01
# n_tagger_group = 1.015
# there_is_ACT5 = True

######## Example 2.5.1: relatively high momentum positive polarity
# run_number = 1606
# run_momentum = 780
# n_eveto_group = 1.01
# n_tagger_group = 1.015
# there_is_ACT5 = True


###Example 3: low momentum, positive polarity 
# run_number = 1308
# run_momentum = 220
# n_eveto_group = 1.01
# n_tagger_group = 1.15
# there_is_ACT5 = False

#choose the number of events to read in, set to -1 if you want to read all events
n_events = 100000 #-1

#Set up a beam analysis class 
ana = bm.BeamAnalysis(run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5)

#Store into memory the number of events desired
ana.open_file(n_events)


# In[3]:


#Step 2: Adjust the 1pe calibration: need to check the accuracy on the plots
# which are stored in plots/PID_run{run_number}_p{run_momentum}.pdf
ana.adjust_1pe_calibration()


# In[4]:


#Step 3: proton and heavier particle tagging with T0-T1 TOF
#We need to tag protons before any other particles to avoid double-counting
ana.tag_protons_TOF()
#TODO: identify protons that produce knock-on electrons 


# In[5]:


#Step 4: tag electrons using ACT0-2 finding the minimum in the cut line
#If we want a tighter cut, add a coefficient of reduction of the optimal cut line (e.g. 5%) to remove more electrons (and also some more muons and pions) 
tightening_factor = 0 #in units of percent of the cut line, how much you want to reduce the cut position to increase the purity of the muon/pion sample
#this is interseting but not really resolving the issue of electron contamination: leave at 0% for now
ana.tag_electrons_ACT02(tightening_factor)

#instead use ACT35 to tag electrons (when depositing more than cutline PE, for now TBD by analyser)
cut_line = 30 #PE
ana.tag_electrons_ACT35(cut_line)


# In[6]:


#Step 5: check visually that the electron and proton removal makes sense in ACT35
ana.plot_ACT35_left_vs_right()


# In[7]:


#Step 6: make the muon/pion separation, using the muon tagger in case 
#at least 0.5% of muons and pions are above the cut line. This is necessary in case the 
#Number of particles is too high to clearly see a minimum between the muons and pions
#A more thorough analysis might want to remove events that are close to the cut line for a higher purity
ana.tag_muons_pions_ACT35()


# In[8]:


#Step 7: estimate the momentum for each particle from the T0-T1 TOF
#Note: we will save:
# 1. the momentum as the particle escapes the beam pipe and its error
# 2. the momentum as the particle escapes the WCTE beam window and its error
# 3. the mean momentum for this particle type and the associated error 

# first measure the particle TOF, make the plot
#This corrects any offset in the TOF (e.g. from cable length) that can cause the TOF 
#of electrons to be different from L/c This has to be calibrated to give meaningful momentum 
#estimates later on
ana.measure_particle_TOF()


# In[9]:


#This function extimates both the mean momentum for each particle type and for each trigger
#We take the the error on the tof for each trigger is the resolution of the TS0-TS1 measurement
#Taken as the std of the gaussian fit to the electron TOF
#This is still a somewhat coarse way of estimating uncertainty... 
#This also saves the momentum after exiting the beam window, recosntructed using the same techinque
#Final momentum is after exiting through the beam pipe

ana.estimate_momentum(-1.012e-2, True)


#note, because of fluctuations in the TOF, the reconstructed momentum will be unphysical for 
#some fo the events on the faster side of the distribution, this means that the 
#distibution of momenta event by event will be non-symetrical with a tail at low momenta
#The mean momentum will be fine though 


# In[10]:


#Visually, it looks like all the particles reach the TOF
ana.plot_TOF_charge_distribution()


# In[11]:


#Step X: end_analysis, necessary to cleanly close files 
ana.end_analysis()


# In[12]:


#Output to a root file
ana.output_beam_ana_to_root()


# ### List of the relevant beam PID information saved to the output file
# After the beam PID has been performed we are saving the relevant variables to an output file, ideally root, to compare with other reconstructions:
# 
# ##### Branch: beam_analysis
# 
# 1. ACT i right (charge in ACT i right PMT, in units of PE) 
# 2. ACT i left  (for precise selections)
# 3. ACT02 total (for coarse selections)
# 4. ACT35 total 
# 5. T0-T1 TOF (called "tof")
# 6. T0-T4 TOF #this still has bugs I think
# 7. T4-T1 TOF  #this still has bugs I think
# 8. Muon tagger information (left and right, and total)
# 
# 16. Estimated PID from the beam information #todo, have a likelihood for each particle type
# 10. Estimated initial momenta for each trigger and error on mean
# 10. Estimated momenta exiting the beampipe for each trigger and error
# 17. Whether event passes beam data quality cuts
# 18. Total TOF detector charge, one can add a selection cut to remove events which do not cross the TOF for further analysis
# 19. ref0 and ref1 times (reference times of each digitiser) 
# 
# ##### Branch: run_info
# 14. run number
# 13. run momentum (nominal, from CERN)
# 14. Aerogel refrective index information
# 15. Whether ACT5 is in the beamline
# 
# ##### Branch: scalar_results
# 
# 18. Position of each cut line and whether we apply the muon tagger cut for pion/muon separation (see step 6)
# 9. Mean tof for each particle type and error on mean
# 9. Mean momenta for each particle type, gaussian std and error on mean
# 
# 

# In[ ]:





# #### This is the end of the analysis please check the plots on plots/PID_run{run_number}\_p{run_momentum}.pdf and the analysis output in beam_analysis_output_R{run_number}.root

# In[ ]:




