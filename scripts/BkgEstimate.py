#!/usr/bin/env python3
import h5py
import numpy as np

run2File = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-run2.h5"
ttbarFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_ttbar.h5"
dijetFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_dijet.h5"

with h5py.File(run2File, "r") as run2:
    with h5py.File(ttbarFile, "r") as ttbar:
        with h5py.File(dijetFile, "r") as dijet:
            # works with any kinematic var as vars are filled with the nr of
            # selected events
            # print(run2.keys())
            CR_2b2b_Data = np.sum(run2["mhh_CR_2b2b"]["histogram"][:])
            CR_2b2j_Data = np.sum(run2["mhh_CR_2b2j"]["histogram"][:])
            CR_2b2b_ttbar = np.sum(ttbar["mhh_CR_2b2b"]["histogram"][:])
            CR_2b2j_ttbar = np.sum(ttbar["mhh_CR_2b2j"]["histogram"][:])
            CR_2b2b_dijet = np.sum(dijet["mhh_CR_2b2b"]["histogram"][:])
            CR_2b2j_dijet = np.sum(dijet["mhh_CR_2b2j"]["histogram"][:])

            VR_2b2b_Data = np.sum(run2["mhh_VR_2b2b"]["histogram"][:])
            VR_2b2j_Data = np.sum(run2["mhh_VR_2b2j"]["histogram"][:])
            VR_2b2b_ttbar = np.sum(ttbar["mhh_VR_2b2b"]["histogram"][:])
            VR_2b2j_ttbar = np.sum(ttbar["mhh_VR_2b2j"]["histogram"][:])
            VR_2b2b_dijet = np.sum(dijet["mhh_VR_2b2b"]["histogram"][:])
            VR_2b2j_dijet = np.sum(dijet["mhh_VR_2b2j"]["histogram"][:])
            
            SR_2b2b_Data = np.sum(run2["mhh_SR_2b2b"]["histogram"][:])
            SR_2b2j_Data = np.sum(run2["mhh_SR_2b2j"]["histogram"][:])
            SR_2b2b_ttbar = np.sum(ttbar["mhh_SR_2b2b"]["histogram"][:])
            SR_2b2j_ttbar = np.sum(ttbar["mhh_SR_2b2j"]["histogram"][:])
            SR_2b2b_dijet = np.sum(dijet["mhh_SR_2b2b"]["histogram"][:])
            SR_2b2j_dijet = np.sum(dijet["mhh_SR_2b2j"]["histogram"][:])
# make multijet only
CR1 = CR_2b2b_Data - CR_2b2b_ttbar
CR2 = CR_2b2j_Data - CR_2b2j_ttbar
VR1 = VR_2b2b_Data - VR_2b2b_ttbar
VR2 = VR_2b2j_Data - VR_2b2j_ttbar
w_CR = CR1 / CR2
w_VR = VR1 / VR2

errCR1 = np.sqrt(CR_2b2b_Data) + np.sqrt(CR_2b2b_ttbar)
errCR2 = np.sqrt(CR_2b2j_Data) + np.sqrt(CR_2b2j_ttbar)
errVR1 = np.sqrt(VR_2b2b_Data) + np.sqrt(VR_2b2b_ttbar)
errVR2 = np.sqrt(VR_2b2j_Data) + np.sqrt(VR_2b2j_ttbar)

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

# with >=
# CR_2b2b_Data :  422.0
# CR_2b2j_Data :  51068.0
# CR_2b2b_ttbar :  23.389917929619916
# CR_2b2j_ttbar :  1616.5546985863693
# CR_2b2b_dijet :  301.01295777480243
# CR_2b2j_dijet :  50043.96716752163
# VR_2b2b_Data :  100.0
# VR_2b2j_Data :  12017.0
# VR_2b2b_ttbar :  8.205115660238086
# VR_2b2j_ttbar :  369.98453930424955
# VR_2b2b_dijet :  75.10225686177348
# VR_2b2j_dijet :  11703.753269491555
# SR_2b2b_Data :  0.0
# SR_2b2j_Data :  4881.0
# SR_2b2b_ttbar :  1.60228221395706
# SR_2b2j_ttbar :  68.95969051783325
# SR_2b2b_dijet :  25.728457586195567
# SR_2b2j_dijet :  4892.262786981583
# CR1 :  398.6100820703801
# CR2 :  49451.44530141363
# VR1 :  91.79488433976192
# VR2 :  11647.015460695751
# w_CR :  0.008060635632402544
# w_VR :  0.007881408301511642
# errCR1 :  25.3789510151095
# errCR2 :  266.18870140591633
# errVR1 :  12.864457306408683
# errVR2 :  128.8570602440667
# err_w_CR :  0.0005150403753024878
# err_w_VR :  0.001107964697105445

# with ==
# CR_2b2b_Data :  393.0
# CR_2b2j_Data :  25861.0
# CR_2b2b_ttbar :  22.676181152330216
# CR_2b2j_ttbar :  935.5754906961702
# CR_2b2b_dijet :  290.60213392208226
# CR_2b2j_dijet :  25923.139412338314
# VR_2b2b_Data :  93.0
# VR_2b2j_Data :  5748.0
# VR_2b2b_ttbar :  8.205115660238086
# VR_2b2j_ttbar :  224.09271509960655
# VR_2b2b_dijet :  72.86379160674761
# VR_2b2j_dijet :  5581.716131645005
# CR1 :  370.32381884766977
# CR2 :  24925.42450930383
# VR1 :  84.79488433976192
# VR2 :  5523.907284900393
# w_CR :  0.014857272288760428
# w_VR :  0.015350526351437657
# errCR1 :  24.58617900331766
# errCR2 :  191.40073517688205
# errVR1 :  12.508108067401638
# errVR2 :  90.78529230483662
# err_w_CR :  0.0009929654997671551
# err_w_VR :  0.0022783697623748847
