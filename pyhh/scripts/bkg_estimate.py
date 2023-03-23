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
            CR_2b2b_Data = np.sum(run2["m_hh.CR_2b2b"]["histogram"][:])
            CR_2b2j_Data = np.sum(run2["m_hh.CR_2b2j"]["histogram"][:])
            CR_2b2b_ttbar = np.sum(ttbar["m_hh.CR_2b2b"]["histogram"][:])
            CR_2b2j_ttbar = np.sum(ttbar["m_hh.CR_2b2j"]["histogram"][:])
            CR_2b2b_dijet = np.sum(dijet["m_hh.CR_2b2b"]["histogram"][:])
            CR_2b2j_dijet = np.sum(dijet["m_hh.CR_2b2j"]["histogram"][:])

            VR_2b2b_Data = np.sum(run2["m_hh.VR_2b2b"]["histogram"][:])
            VR_2b2j_Data = np.sum(run2["m_hh.VR_2b2j"]["histogram"][:])
            VR_2b2b_ttbar = np.sum(ttbar["m_hh.VR_2b2b"]["histogram"][:])
            VR_2b2j_ttbar = np.sum(ttbar["m_hh.VR_2b2j"]["histogram"][:])
            VR_2b2b_dijet = np.sum(dijet["m_hh.VR_2b2b"]["histogram"][:])
            VR_2b2j_dijet = np.sum(dijet["m_hh.VR_2b2j"]["histogram"][:])
            
            SR_2b2b_Data = np.sum(run2["m_hh.SR_2b2b"]["histogram"][:])
            SR_2b2j_Data = np.sum(run2["m_hh.SR_2b2j"]["histogram"][:])
            SR_2b2b_ttbar = np.sum(ttbar["m_hh.SR_2b2b"]["histogram"][:])
            SR_2b2j_ttbar = np.sum(ttbar["m_hh.SR_2b2j"]["histogram"][:])
            SR_2b2b_dijet = np.sum(dijet["m_hh.SR_2b2b"]["histogram"][:])
            SR_2b2j_dijet = np.sum(dijet["m_hh.SR_2b2j"]["histogram"][:])
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


# CR_2b2b_Data :  422.0
# CR_2b2j_Data :  51068.0
# CR_2b2b_ttbar :  12.646221207021231
# CR_2b2j_ttbar :  772.0608981793496
# CR_2b2b_dijet :  301.01295777480243
# CR_2b2j_dijet :  50043.96716752165
# VR_2b2b_Data :  100.0
# VR_2b2j_Data :  12017.0
# VR_2b2b_ttbar :  3.928231397855248
# VR_2b2j_ttbar :  178.71181653892523
# VR_2b2b_dijet :  75.10225686177348
# VR_2b2j_dijet :  11703.753269491554
# SR_2b2b_Data :  0.0
# SR_2b2j_Data :  4881.0
# SR_2b2b_ttbar :  0.8228519052958185
# SR_2b2j_ttbar :  31.615168163356643
# SR_2b2b_dijet :  25.728457586195567
# SR_2b2j_dijet :  4892.262786981583
# CR1 :  409.35377879297874
# CR2 :  50295.93910182065
# VR1 :  96.07176860214476
# VR2 :  11838.288183461074
# w_CR :  0.008138903181910379
# w_VR :  0.00811534295442848
# errCR1 :  24.098791169064334
# errCR2 :  253.7682840358571
# errVR1 :  11.981976639079091
# errVR2 :  122.99039196861689
# err_w_CR :  0.00048089641813414284
# err_w_VR :  0.0010156431633817733