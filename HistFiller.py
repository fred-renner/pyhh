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
accEffBinning = {"binrange": (0, 3_000_000), "bins": 100}
hists = {
    "events_truth_mhh": FloatHistogram(
        name="events_truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    "nTriggerPass_truth_mhh": FloatHistogram(
        name="nTriggerPass_truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    "nTwoSelLargeR_truth_mhh": FloatHistogram(
        name="nTwoSelLargeR_truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    # "nTotalSelLargeR": FloatHistogram(
    #     name="nTotalSelLargeR",
    #     binrange=(0, 1_000_000),
    #     bins=150,
    # ),
    # "triggerEff": FloatHistogram(
    #     name="triggerEff",
    #     binrange=(0, 1_000_000),
    #     bins=150,
    # ),
    "hh_m_85": FloatHistogram(
        name="hh_m_85",
        binrange=(0, 900_000),
        bins=150,
    ),
    # "pairingEfficiencyResolved": IntHistogram(
    #     name="pairingEfficiencyResolved",
    #     binrange=(0, 3),
    # ),
    # "vrJetEfficiencyBoosted": IntHistogram(
    #     name="vrJetEfficiencyBoosted",
    #     binrange=(0, 3),
    # ),
    "massplane_85": FloatHistogram2D(
        name="massplane_85",
        binrange1=(50_000, 300_000),
        binrange2=(50_000, 300_000),
        bins=100,
    ),
}

# the filling is executed each time an Analysis.Run job finishes
def filling_callback(results):
    for hist in hists:
        # update bin heights per iteration
        values = results[hist]
        hists[hist].fill(values)
    pbar.update(batchSize)


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
            # the auto batchSize setup could crash if you don't have enough memory
            cpus = multiprocessing.cpu_count()
            batchSize = int(tree.num_entries / cpus)
            eventBatches = Loader.EventRanges(tree, batch_size=batchSize, nEvents=-1)
            # a pool objects can start child processes on different cpus
            pool = multiprocessing.Pool(cpus)
            for batch in eventBatches:
                pool.apply_async(
                    Analysis.Run, (batch, tree, vars), callback=filling_callback
                )
            pool.close()
            pool.join()
            pbar.close()

    # write histograms to file
    for hist in hists:
        hists[hist].write(outfile, hist)

# if you want to plot directly
import subprocess

subprocess.call(
    "python3 /lustre/fs22/group/atlas/freder/hh/hh-analysis/Plotter.py --histFile"
    f" {histOutFile}",
    shell=True,
)
