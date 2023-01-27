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
# CR_4b_Data_Check :  476.0
# CR_2b_Data :  27245.0
# CR_4b_ttbar :  25.2243379587919
# CR_2b_ttbar :  4312.166586272076
# CR_4b_dijet :  355.0387082070425
# CR_2b_dijet :  19283.309642546592
# VR_4b_Data :  114.0
# VR_2b_Data :  6813.0
# VR_4b_ttbar :  8.835739726469441
# VR_2b_ttbar :  1527.4468765862598
# VR_4b_dijet :  79.74883132587807
# VR_2b_dijet :  4229.4590801394415
# CR_4b_Data_noVBF :  6099.0
# CR_2b_Data_noVBF :  380451.0
# CR_4b_ttbar_noVBF :  353.77971939962123
# CR_2b_ttbar_noVBF :  60569.48074958573
# CR_4b_dijet_noVBF :  4182.1945377480115
# CR_2b_dijet_noVBF :  261029.37280687067
# VR_4b_Data_noVBF :  1528.0
# VR_2b_Data_noVBF :  93542.0
# VR_4b_ttbar_noVBF :  123.56778874286414
# VR_2b_ttbar_noVBF :  22872.70776443189
# VR_4b_dijet_noVBF :  968.80372878178
# VR_2b_dijet_noVBF :  52602.31350941885
# CR1 :  450.7756620412081
# CR2 :  22932.833413727923
# VR1 :  105.16426027353056
# VR2 :  5285.55312341374
# w_CR :  0.019656343981087277
# w_VR :  0.019896547781854273
# errCR1 :  26.839807922179794
# errCR2 :  230.72768399911382
# errVR1 :  13.649575472630422
# errVR2 :  121.62346382022398
# err_w_CR :  0.0011869568916442735
# err_w_VR :  0.0026227004831766423