#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import uproot
import numpy as np
import re
import Loader
from Histograms import FloatHistogram
from h5py import File, Group, Dataset


# files to load
filelist = [
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801578.Py8EG_A14NNPDF23LO_XHS_X300_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801584.Py8EG_A14NNPDF23LO_XHS_X400_S200_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.root",
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
]


# vars to load
vars = [
    "resolved_DL1dv00_FixedCutBEff_85_hh_m",
    "resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle",
    "resolved_DL1dv00_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle",
    "resolved_DL1dv00_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB",
    "resolved_DL1dv00_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB",
    "resolved_DL1dv00_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB",
    "resolved_DL1dv00_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB",
]

with File("/lustre/fs22/group/atlas/freder/hh/run/histograms/hists.h5", "w") as outfile:
    for file in filelist:
        with uproot.open(file) as file_:
            tree = file_["AnalysisMiniTree"]
            plotname = file.split("/")
            plotname = str(plotname[-1]).replace(".root", "")
            print("Making hists for " + plotname)
            # # create empty hist
            hist = FloatHistogram(
                name=plotname, binrange=(0, 900_000), bins=150, compress=True
            )

            # make generators to load only a certain amount
            generators = Loader.GetGenerators(tree, vars)
            for vars_arr in generators:

                # cutting
                vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"] = vars_arr[
                    "resolved_DL1dv00_FixedCutBEff_85_hh_m"
                ][vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"] > 0]

                # update bin heights per iteration
                hist.fill(vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"])

            # write histogram to file
            hist.write(outfile, plotname)
