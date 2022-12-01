#!/usr/bin/env python3
from tqdm.auto import tqdm
import numpy as np
import uproot
import numpy as np
import Loader
from Histograms import FloatHistogram, IntHistogram, FloatHistogram2D
from h5py import File, Group, Dataset
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

if filenames:
    filelist = [path + "/" + file for file in filenames]


# filelist = [
#     # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801577.Py8EG_A14NNPDF23LO_XHS_X200_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
#     # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
#     # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801619.Py8EG_A14NNPDF23LO_XHS_X2000_S400_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.root"
#     # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc21_13p6TeV.601479.PhPy8EG_HH4b_cHHH01d0.deriv.DAOD_PHYS.e8472_s3873_r13829_p5440.root"
#     # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables.root"
#     "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE/user.frenner.31380623._000001.output-hh4b.root"
# ]

# make hist out file name from filename
filename = filelist[0].split("/")
filename = str(filename[-1]).replace(".root", "")
histOutFile = (
    "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-" + filename + ".h5"
)


start = 'vars_arr["'
end = '"]'

vars = []
for line in open("/lustre/fs22/group/atlas/freder/hh/hh-analysis/Analysis.py", "r"):
    if "vars_arr[" in line:
        vars.append((line.split(start))[1].split(end)[0])


# vars to load
# vars = [
#     # "resolved_DL1dv00_FixedCutBEff_85_hh_m",
#     # "resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle",
#     # "resolved_DL1dv00_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle",
#     # "resolved_DL1dv00_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB",
#     # "resolved_DL1dv00_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB",
#     # "resolved_DL1dv00_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB",
#     # "resolved_DL1dv00_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB",
#     # "recojet_antikt4_NOSYS_pt",
#     # "boosted_DL1r_FixedCutBEff_85_h1_parentPdgId_leadingJet_closestTruthB",
# ]

# TODO
# could think of having btag wp configurable for everything
# make yaml config
# could think of remove defaults before sending into analysis

# define hists
accEffBinning = {"binrange": (0, 1_500_000), "bins": 100}
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


# def filling_callback(results):
#     print("did")
#     for hist in hists:
#         # update bin heights per iteration
#         values = results[hist]
#         print(values.shape)
#         hists[hist].fill(values)
#     #     print(hists[hist]._hist[100])
#     # pbar.update(1000)



with File(histOutFile, "w") as outfile:
    # loop over input files
    for file in filelist:
        print("Making hists for " + file)
        with uproot.open(file) as file_:
            # access the tree
            tree = file_["AnalysisMiniTree"]
            # progressbar
            pbar = tqdm(total=tree.num_entries, position=0, leave=True)
            # load only a certain amount of events in batches likes
            # [[0, 999], [1000, 1999], [2000, 2999],...]
            eventBatches = Loader.EventRanges(tree, batch_size=100000)
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                for batch in eventBatches:
                    results=pool.apply(Analysis.Run, args=(batch, tree, vars))
                    for hist in hists:
                        # update bin heights per iteration
                        values = results[hist]
                        hists[hist].fill(values)
                    pbar.update(batch[1])

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
