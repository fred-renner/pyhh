import numpy as np
import h5py


# fmt: off
# SMsignalFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r13144_p5440_TREE.h5"
SMsignalFilePath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_SM.h5"
ttbarFilePath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_ttbar.h5"
dijetFilePath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_dijet.h5"
run2FilePath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-run2.h5"
# fmt: on


def getHist(file, name):
    # access [1:-1] to remove underflow and overflow bins
    h = np.array(file[name]["histogram"][1:-1])
    hRaw = np.array(file[name]["histogramRaw"][1:-1])
    edges = np.array(file[name]["edges"][:])
    err = np.sqrt(file[name]["w2sum"][1:-1])
    return {"h": h, "hRaw": hRaw, "edges": edges, "err": err}


def get2dHist(file, name):
    h = np.array(file[name]["histogram"][1:-1, 1:-1])
    hRaw = np.array(file[name]["histogramRaw"][1:-1, 1:-1])
    xbins = np.array(file[name]["edges"][0][1:-1])
    ybins = np.array(file[name]["edges"][1][1:-1])
    err = np.sqrt(file[name]["w2sum"][1:-1, 1:-1])
    return {"h": h, "hRaw": hRaw, "xbins": xbins, "ybins": ybins, "err": err}


def load(file):
    hists = {}
    for key in file.keys():
        if "massplane" in key:
            hists[key] = get2dHist(file, key)
        else:
            hists[key] = getHist(file, key)
    return hists


def run():
    with h5py.File(SMsignalFilePath, "r") as f_SMsignal, h5py.File(
        run2FilePath, "r"
    ) as f_run2, h5py.File(ttbarFilePath, "r") as f_ttbar, h5py.File(
        dijetFilePath, "r"
    ) as f_dijet:
        SMsignal = load(f_SMsignal)
        run2 = load(f_run2)
        ttbar = load(f_ttbar)
        dijet = load(f_dijet)

        hists = {}
        hists["SMsignal"] = SMsignal
        hists["run2"] = run2
        hists["ttbar"] = ttbar
        hists["dijet"] = dijet

        return hists
