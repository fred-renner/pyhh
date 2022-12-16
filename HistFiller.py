#!/usr/bin/env python3
from tqdm.auto import tqdm
import numpy as np
import uproot
import numpy as np
import Loader
from Histograms import FloatHistogram, IntHistogram, FloatHistogram2D
from h5py import File
import Analysis
import yaml
import os
import multiprocessing
import argparse
import glob


# TODO
# make yaml config

parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, default=None)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

# import time
# t0 = time.time()
# print(time.time() - t0)
# yaml.safe_load(file)

# files to load
pattern = "*"

# mc20 testfiles
# topPath = "/lustre/fs22/group/atlas/freder/hh/run/signal-test/"
# topPath = "/lustre/fs22/group/atlas/freder/hh/run/bkg-test"

# mc21 signal
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE"
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_25_.601480.PhPy8EG_HH4b_cHHH10d0.e8472_s3873_r13829_p5440_TREE"
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_30.801172.Py8EG_A14NNPDF23LO_jj_JZ7.e8453_s3873_r13829_p5278_TREE"

# mc20 signal
# 1cvv1cv1
topPath = "/lustre/fs22/group/atlas/freder/hh/samples/"
pattern = "user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r*/*"
histOutFileName = "hists-MC20-signal-1cvv1cv1.h5"

# mc20 bkg
#ttbar
topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
pattern = "*ttbar*/*"
histOutFileName = "hists-MC20-bkg-ttbar.h5"


# get all files also from subdirectories with wildcard
filelist = []
for file in glob.iglob(topPath + "/" + pattern):
    filelist += [file]

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
TriggerEffpT = {"binrange": (0, 3_000_000), "bins": 100}
TriggerEffm = {"binrange": (0, 300_000), "bins": 100}

hists = [
    FloatHistogram(
        name="truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="mh1",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="mh2",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
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
    for hist in hists:
        # update bin heights per iteration
        values = results[hist._name]
        hist.fill(values)
    pbar.update(batchSize)


def error_handler(e):
    print("\n\n---error_start---{}\n---error_end---\n".format(e.__cause__))
    pool.terminate()


# debugging settings
if args.debug:
    nEvents = 100
    cpus = 1
    filelist = filelist[:2]
else:
    nEvents = None

with File(histOutFile, "w") as outfile:
    # loop over input files
    for i, file in enumerate(filelist):
        print("Making hists for " + file)
        print("Processing file " + str(i + 1) + "/" + str(len(filelist)))
        with uproot.open(file) as file_:
            # access the tree
            tree = file_["AnalysisMiniTree"]
            # take only vars that exist
            varsExist = set(tree.keys()).intersection(vars)
            # progressbar
            pbar = tqdm(total=tree.num_entries, position=0, leave=True)
            # the auto batchSize setup could crash if you don't have enough
            # memory
            cpus = multiprocessing.cpu_count() - 8
            batchSize = int(tree.num_entries / cpus)
            if args.cpus:
                cpus = args.cpus
                batchSize = 10_0000

            eventBatches = Loader.EventRanges(
                tree, batch_size=batchSize, nEvents=nEvents
            )
            # a pool objects can start child processes on different cpus
            pool = multiprocessing.Pool(cpus)
            for batch in eventBatches:
                pool.apply_async(
                    Analysis.Run,
                    (batch, tree, varsExist),
                    callback=filling_callback,
                    error_callback=error_handler,
                )
            pool.close()
            pool.join()
            pbar.close()

    # write histograms to file
    for hist in hists:
        hist.write(outfile)

# if you want to plot directly
if not args.debug:
    import subprocess

    subprocess.call(
        "python3 /lustre/fs22/group/atlas/freder/hh/hh-analysis/Plotter.py --histFile"
        f" {histOutFile}",
        shell=True,
    )
