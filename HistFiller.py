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


parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, default=None)
args = parser.parse_args()


# import time
# t0 = time.time()
# print(time.time() - t0)
# yaml.safe_load(file)

# files to load
path = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE"
# path = "/lustre/fs22/group/atlas/freder/hh/run/testfiles"
filenames = os.listdir(path)
filelist = [path + "/" + file for file in filenames]


# make hist out file name from filename
dataset = path.split("/")
histOutFile = (
    "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-" + dataset[-1] + ".h5"
)

# figure out which vars to load from analysis script
start = 'vars_arr["'
end = '"]'
vars = []
for line in open("/lustre/fs22/group/atlas/freder/hh/hh-analysis/Analysis.py", "r"):
    if "vars_arr[" in line:
        vars.append((line.split(start))[1].split(end)[0])

# TODO
# could think of having btag wp configurable for everything
# make yaml config
# could think of remove defaults before sending into analysis

# define hists
accEffBinning = {"binrange": (0, 3_000_000), "bins": 150}
TriggerEff = {"binrange": (0, 3_000_000), "bins": 100}

hists = [
    FloatHistogram(
        name="truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTriggerPass_truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTwoSelLargeR_truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTotalSelLargeR",
        binrange=(0, 2_500_000),
        bins=100,
    ),
    FloatHistogram(
        name="triggerRef_leadingLargeRpT",
        binrange=TriggerEff["binrange"],
        bins=TriggerEff["bins"],
    ),
    FloatHistogram(
        name="trigger_leadingLargeRpT",
        binrange=TriggerEff["binrange"],
        bins=TriggerEff["bins"],
    ),
    FloatHistogram(
        name="leadingLargeRpT",
        binrange=TriggerEff["binrange"],
        bins=TriggerEff["bins"],
    ),
    FloatHistogram(
        name="hh_m_85",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    # "pairingEfficiencyResolved": IntHistogram(
    #     name="pairingEfficiencyResolved",
    #     binrange=(0, 3),
    # ),
    # "vrJetEfficiencyBoosted": IntHistogram(
    #     name="vrJetEfficiencyBoosted",
    #     binrange=(0, 3),
    # ),
    FloatHistogram2D(
        name="massplane_85",
        binrange1=(50_000, 250_000),
        binrange2=(50_000, 250_000),
        bins=100,
    ),
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


with File(histOutFile, "w") as outfile:
    # loop over input files
    for file in filelist:
        print("Making hists for " + file)
        with uproot.open(file) as file_:
            # access the tree
            tree = file_["AnalysisMiniTree"]
            # progressbar
            pbar = tqdm(total=tree.num_entries, position=0, leave=True)
            # with batchsize=1000 you would load events incrementally
            # [[0, 999], [1000, 1999], [2000, 2999],...]
            # the auto batchSize setup could crash if you don't have enough
            # memory
            cpus = multiprocessing.cpu_count()
            batchSize = int(tree.num_entries / cpus)

            if args.cpus:
                cpus = args.cpus
                batchSize = 30_0000

            eventBatches = Loader.EventRanges(tree, batch_size=batchSize, nEvents=None)
            # a pool objects can start child processes on different cpus
            pool = multiprocessing.Pool(cpus)
            for batch in eventBatches:
                pool.apply_async(
                    Analysis.Run,
                    (batch, tree, vars),
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
import subprocess

subprocess.call(
    "python3 /lustre/fs22/group/atlas/freder/hh/hh-analysis/Plotter.py --histFile"
    f" {histOutFile}",
    shell=True,
)
