#!/usr/bin/env python3
from tqdm.auto import tqdm
import numpy as np
import uproot
import numpy as np
import Loader
from Histograms import FloatHistogram, IntHistogram, FloatHistogram2D
from h5py import File
import Analysis
import multiprocessing
import argparse
import glob
import subprocess
import HistFillerTools as tools
import os
import time


# TODO
# make yaml config

parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, default=None)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--file", type=str, default=None)
parser.add_argument("--batchMode", action="store_true")

args = parser.parse_args()

# files to load

topPath = "/lustre/fs22/group/atlas/freder/hh/samples/"
# mc21 signal
# pattern = "user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE/*"
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_25_.601480.PhPy8EG_HH4b_cHHH10d0.e8472_s3873_r13829_p5440_TREE"
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_30.801172.Py8EG_A14NNPDF23LO_jj_JZ7.e8453_s3873_r13829_p5278_TREE"

# mc20 signal
# 1cvv1cv1
pattern = "user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r*/*"

# mc20 bkg
# # ttbar
# topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
# pattern = "*ttbar*/*"
# histOutFileName = "hists-MC20-bkg-ttbar.h5"
# # dijet
# topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
# pattern = "*jetjet*/*"
# histOutFileName = "hists-MC20-bkg-dijet.h5"

# data 17
# topPath = "/lustre/fs22/group/atlas/freder/hh/run/testfiles/"
# pattern = "data*"
# histOutFileName = "hists-data17.h5"

# get all files also from subdirectories with wildcard
filelist = []
for file in glob.iglob(topPath + "/" + pattern):
    filelist += [file]


if args.file:
    filelist = [args.file]
    fileParts = filelist[0].split("/")
    dataset = fileParts[-2]
    datasetPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/" + dataset
    file = fileParts[-1]
    if not os.path.isdir(datasetPath):
        os.makedirs(datasetPath)

    histOutFile = (
        "/lustre/fs22/group/atlas/freder/hh/run/histograms/"
        + dataset
        + "/"
        + file
        + ".h5"
    )
else:
    # make hist out file name from filename
    if "histOutFileName" not in locals():
        dataset = filelist[0].split("/")
        histOutFileName = "hists-" + dataset[-2] + ".h5"

    histOutFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/" + histOutFileName


# figure out which vars to load from analysis script
start = 'vars_arr["'
end = '"]'
vars = []
for line in open("/lustre/fs22/group/atlas/freder/hh/hh-analysis/Analysis.py", "r"):
    if "vars_arr[" in line:
        if "#" not in line:
            vars.append((line.split(start))[1].split(end)[0])


# define hists
accEffBinning = {"binrange": (0, 3_000_000), "bins": 75}
m_hBinning = {"binrange": (0, 300_000), "bins": 100}
pt_hBinning = {"binrange": (0.2e6, 1e6), "bins": 100}
TriggerEffpT = {"binrange": (0, 3_000_000), "bins": 150}
TriggerEffm = {"binrange": (0, 300_000), "bins": 150}
dRbins = {"binrange": (0, 1.2), "bins": 75}

hists = [
    FloatHistogram(
        name="truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    # needs to be the same binning as accEff for plot
    FloatHistogram(
        name="mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="mh1",
        binrange=m_hBinning["binrange"],
        bins=m_hBinning["bins"],
    ),
    FloatHistogram(
        name="mh2",
        binrange=m_hBinning["binrange"],
        bins=m_hBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h1",
        binrange=pt_hBinning["binrange"],
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h2",
        binrange=pt_hBinning["binrange"],
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="pt_hh",
        binrange=(0, 1e6),
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="pt_hh_scalar",
        binrange=(0.4e6, 1.5e6),
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="nTriggerPass_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTwoLargeR_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTwoSelLargeR_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTotalSelLargeR",
        binrange=(0, 2_500_000),
        bins=100,
    ),
    FloatHistogram(
        name="leadingLargeRpT",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="leadingLargeRpT_trigger",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="triggerRef_leadingLargeRpT",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="trigger_leadingLargeRpT",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="triggerRef_leadingLargeRm",
        binrange=TriggerEffm["binrange"],
        bins=TriggerEffm["bins"],
    ),
    FloatHistogram(
        name="trigger_leadingLargeRm",
        binrange=TriggerEffm["binrange"],
        bins=TriggerEffm["bins"],
    ),
    FloatHistogram(
        name="btagLow_1b1j_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagLow_2b1j_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagLow_2b2j_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagHigh_1b1b_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagHigh_2b1b_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagHigh_2b2b_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="dR_h1",
        binrange=dRbins["binrange"],
        bins=dRbins["bins"],
    ),
    FloatHistogram(
        name="dR_h2",
        binrange=dRbins["binrange"],
        bins=dRbins["bins"],
    ),
    FloatHistogram2D(
        name="massplane_77",
        binrange1=(50_000, 250_000),
        binrange2=(50_000, 250_000),
        bins=100,
    ),
    # "vrJetEfficiencyBoosted": IntHistogram(
    #     name="vrJetEfficiencyBoosted",
    #     binrange=(0, 3),
    # ),
]

# the filling is executed each time an Analysis.Run job finishes
def filling_callback(results):
    # update bin heights per iteration
    for hist in hists:
        if hist._name not in results.keys():
            print(f"{hist._name} defined but not in results")
        res = results[hist._name]
        hist.fill(values=res[0], weights=res[1])
    pbar.update(batchSize)


def error_handler(e):
    print("\n\n---error_start---{}\n---error_end---\n".format(e.__cause__))
    pool.terminate()


# debugging settings
if args.debug:
    filelist = filelist[:2]
    histOutFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-debug.h5"


with File(histOutFile, "w") as outfile:
    # loop over input files
    for i, file_ in enumerate(filelist):
        print("\nProcessing file " + str(i + 1) + "/" + str(len(filelist)))
        with uproot.open(file_) as file:
            # access the tree
            tree = file["AnalysisMiniTree"]
            # take only vars that exist
            varsExist = set(tree.keys()).intersection(vars)
            # the auto batchSize setup could crash if you don't have enough
            # memory
            if args.debug:
                nEvents = 100
                cpus = 1
                batchSize = int(tree.num_entries / cpus)
                metaData = {}
                metaData["initial_sum_of_weights"] = 1e10
                metaData["crossSection"] = 1e-6
                metaData["dataYears"] = ["2017"]
                metaData["genFiltEff"] = 1.0

            else:
                nEvents = None
                cpus = multiprocessing.cpu_count() - 4
                if cpus > 32:
                    cpus = 32
                batchSize = int(tree.num_entries / cpus)
                metaData = {}
                if "data" not in file_:
                    metaData = tools.getMetaData(file)
                if args.cpus:
                    cpus = args.cpus
                    batchSize = 10_000
            eventBatches = Loader.EventRanges(
                tree, batch_size=batchSize, nEvents=nEvents
            )
            # a pool objects can start child processes on different cpus
            pool = multiprocessing.Pool(cpus)
            # progressbar
            pbar = tqdm(total=tree.num_entries, position=0, leave=True)
            for batch in eventBatches:
                pool.apply_async(
                    Analysis.Run,
                    (batch, metaData, tree, varsExist),
                    callback=filling_callback,
                    error_callback=error_handler,
                )
            pool.close()
            pool.join()
            pbar.close()
            print("Done")

    # write histograms to file
    for hist in hists:
        hist.write(outfile)

# # if to plot directly
# if not args.debug:
#     subprocess.call(
#         "python3 /lustre/fs22/group/atlas/freder/hh/hh-analysis/Plotter.py --histFile"
#         f" {histOutFile}",
#         shell=True,
#     )
