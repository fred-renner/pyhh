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



# CR_2b2b_Data :  424.0
# CR_2b2j_Data :  51208.0
# CR_2b2b_ttbar :  23.389917929619916
# CR_2b2j_ttbar :  1618.4386981390915
# CR_2b2b_dijet :  303.54647944993513
# CR_2b2j_dijet :  50135.86818185366
# VR_2b2b_Data :  100.0
# VR_2b2j_Data :  12038.0
# VR_2b2b_ttbar :  8.205115660238086
# VR_2b2j_ttbar :  369.98453930424955
# VR_2b2b_dijet :  75.10225686177348
# VR_2b2j_dijet :  11708.261919075809
# CR1 :  400.6100820703801
# CR2 :  49589.56130186091
# VR1 :  91.79488433976192
# VR2 :  11668.015460695751
# w_CR :  0.008078516356129706
# w_VR :  0.007867223406497637
# errCR1 :  25.427572712909363
# errCR2 :  266.5216704500192
# errVR1 :  12.864457306408683
# errVR2 :  128.95280205937758
# err_w_CR :  0.000514595550525431
# err_w_VR :  0.0011059633469388869