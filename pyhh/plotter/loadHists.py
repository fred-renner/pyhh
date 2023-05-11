import h5py
import numpy as np

files = {}
files["SM"] = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_SM.h5"
files["k2v0"] = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_k2v0.h5"
files["ttbar"] = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_ttbar.h5"
files["dijet"] = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_dijet.h5"
files["run2"] = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-run2.h5"


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


def load(file, histKey=None):
    hists = {}
    if histKey == None:
        for key in file.keys():
            if "massplane" in key:
                hists[key] = get2dHist(file, key)
            else:
                hists[key] = getHist(file, key)
    else:
        if "massplane" in histKey:
            hists[histKey] = get2dHist(file, histKey)
        else:
            hists[histKey] = getHist(file, histKey)

    return hists


def run(histKey=None):
    allHists = {}
    # loops over datatype files
    for key, file in files.items():
        with h5py.File(file, "r") as f:
            allHists[key] = load(f, histKey)

    return allHists
