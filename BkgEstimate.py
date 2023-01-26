#!/usr/bin/env python3
import h5py
import numpy as np

run2File = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-run2.h5"
ttbarFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_ttbar.h5"
dijetFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_dijet.h5"

with h5py.File(run2File, "r") as run2:
    with h5py.File(ttbarFile, "r") as ttbar:
            with h5py.File(dijetFile, "r") as dijet:

                CR_4b_Data = run2["N_CR_4b"]["histogram"][1]
                CR_4b_Data_Check = np.sum(run2["mhh_CR_4b"]["histogram"][:])
                CR_2b_Data = run2["N_CR_2b"]["histogram"][1]
                CR_4b_ttbar = ttbar["N_CR_4b"]["histogram"][1]
                CR_2b_ttbar = ttbar["N_CR_2b"]["histogram"][1]
                CR_4b_dijet = dijet["N_CR_4b"]["histogram"][1]
                CR_2b_dijet = dijet["N_CR_2b"]["histogram"][1]
                
                VR_4b_Data = run2["N_VR_4b"]["histogram"][1]
                VR_2b_Data = run2["N_VR_2b"]["histogram"][1]
                VR_4b_ttbar = ttbar["N_VR_4b"]["histogram"][1]
                VR_2b_ttbar = ttbar["N_VR_2b"]["histogram"][1]
                VR_4b_dijet = dijet["N_VR_4b"]["histogram"][1]
                VR_2b_dijet = dijet["N_VR_2b"]["histogram"][1]
                
                
                CR_4b_Data_noVBF = np.sum(run2["mhh_CR_4b_noVBF"]["histogram"][:])
                CR_2b_Data_noVBF = np.sum(run2["mhh_CR_2b_noVBF"]["histogram"][:])
                CR_4b_ttbar_noVBF = np.sum(ttbar["mhh_CR_4b_noVBF"]["histogram"][:])
                CR_2b_ttbar_noVBF = np.sum(ttbar["mhh_CR_2b_noVBF"]["histogram"][:])
                CR_4b_dijet_noVBF = np.sum(dijet["mhh_CR_4b_noVBF"]["histogram"][:])
                CR_2b_dijet_noVBF = np.sum(dijet["mhh_CR_2b_noVBF"]["histogram"][:])
                
                VR_4b_Data_noVBF = np.sum(run2["mhh_VR_4b_noVBF"]["histogram"][:])
                VR_2b_Data_noVBF = np.sum(run2["mhh_VR_2b_noVBF"]["histogram"][:])
                VR_4b_ttbar_noVBF = np.sum(ttbar["mhh_VR_4b_noVBF"]["histogram"][:])
                VR_2b_ttbar_noVBF = np.sum(ttbar["mhh_VR_2b_noVBF"]["histogram"][:])
                VR_4b_dijet_noVBF = np.sum(dijet["mhh_VR_4b_noVBF"]["histogram"][:])
                VR_2b_dijet_noVBF = np.sum(dijet["mhh_VR_2b_noVBF"]["histogram"][:])
# make multijet only
CR1 = CR_4b_Data - CR_4b_ttbar
CR2 = CR_2b_Data - CR_2b_ttbar
VR1 = VR_4b_Data - VR_4b_ttbar
VR2 = VR_2b_Data - VR_2b_ttbar
w_CR = CR1 / CR2
w_VR = VR1 / VR2

w_dijet_CR= CR_4b_dijet/CR_2b_dijet
errCR1 = np.sqrt(CR_4b_Data) + np.sqrt(CR_4b_ttbar)
errCR2 = np.sqrt(CR_2b_Data) + np.sqrt(CR_2b_ttbar)
errVR1 = np.sqrt(VR_4b_Data) + np.sqrt(VR_4b_ttbar)
errVR2 = np.sqrt(VR_2b_Data) + np.sqrt(VR_2b_ttbar)

err_w_CR = w_CR * np.sqrt(np.square(errCR1 / CR1) + np.square(errCR2 / CR2))
err_w_VR = w_VR * np.sqrt(np.square(errVR1 / VR1) + np.square(errVR2 / VR2))

tmp = globals().copy()
[
    print(k, ": ", v)
    # print(k, ": ", v, " type:", type(v))
    for k, v in tmp.items()
    if not k.startswith("_")
    and k != "tmp"
    and k != "In"
    and k != "Out"
    and not hasattr(v, "__call__")
]


# CR_4b_Data :  476.0
# CR_2b_Data :  27245.0
# CR_4b_ttbar :  577.0988661938811
# CR_2b_ttbar :  91436.87817296207
# CR_4b_dijet :  43872.260094637044
# CR_2b_dijet :  1271378.787915694
# VR_4b_Data :  114.0
# VR_2b_Data :  6813.0
# VR_4b_ttbar :  169.76461182792306
# VR_2b_ttbar :  32377.64522090067
# VR_4b_dijet :  6393.157872720549
# VR_2b_dijet :  325028.1555551327
# CR1 :  -101.09886619388112
# CR2 :  -64191.87817296207
# VR1 :  -55.76461182792306
# VR2 :  -25564.64522090067
# w_CR :  0.0015749479384521959
# w_VR :  0.002181317649670806
# w_dijet_CR :  0.034507623150266245
# errCR1 :  45.840306366806075
# errCR2 :  467.44590901691686
# errVR1 :  23.70645321125031
# errVR2 :  262.4787916297997
# err_w_CR :  0.000714205894294129
# err_w_VR :  0.0009275844042944473