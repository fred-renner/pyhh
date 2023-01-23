#!/usr/bin/env python3
import h5py

run2File = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-run2.h5"
ttbarFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_ttbar.h5"
with h5py.File(run2File, "r") as run2:
    with h5py.File(ttbarFile, "r") as ttbar:

        CR_4b_Data = run2["N_CR_4b"]["histogramRaw"][1]
        CR_2b_Data = run2["N_CR_2b"]["histogramRaw"][1]
        CR_4b_ttbar = ttbar["N_CR_4b"]["histogramRaw"][1]
        CR_2b_ttbar = ttbar["N_CR_2b"]["histogramRaw"][1]

        VR_4b_Data = run2["N_VR_4b"]["histogramRaw"][1]
        VR_2b_Data = run2["N_VR_2b"]["histogramRaw"][1]
        VR_4b_ttbar = ttbar["N_VR_4b"]["histogramRaw"][1]
        VR_2b_ttbar = ttbar["N_VR_2b"]["histogramRaw"][1]

w_CR = (CR_4b_Data - CR_4b_ttbar) / (CR_2b_Data - CR_2b_ttbar)
w_VR = (VR_4b_Data - VR_4b_ttbar) / (VR_2b_Data - VR_2b_ttbar)


def printvars():

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


printvars()
