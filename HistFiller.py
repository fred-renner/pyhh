#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import uproot
import numpy as np
import Loader
from Histograms import FloatHistogram, IntHistogram
from h5py import File, Group, Dataset
import Analysis
import yaml

# yaml.safe_load(file)

# files to load
filelist = [
    # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801577.Py8EG_A14NNPDF23LO_XHS_X200_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
    # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801584.Py8EG_A14NNPDF23LO_XHS_X400_S200_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.root",
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
    # "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801619.Py8EG_A14NNPDF23LO_XHS_X2000_S400_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.root"
]

# make hist out file name from filename
filename = filelist[0].split("/")
filename = str(filename[-1]).replace(".root", "")
histOutFile = (
    "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-" + filename + ".h5"
)


# vars to load
vars = [
    # "resolved_DL1dv00_FixedCutBEff_85_hh_m",
    # "resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle",
    # "resolved_DL1dv00_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle",
    # "resolved_DL1dv00_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB",
    # "resolved_DL1dv00_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB",
    # "resolved_DL1dv00_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB",
    # "resolved_DL1dv00_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB",
    # "recojet_antikt4_NOSYS_pt",
    # "boosted_DL1r_FixedCutBEff_85_h1_parentPdgId_leadingJet_closestTruthB",
]

# TODO
# could think of having btag wp configurable for everything
# make yaml config

# define hists
hists = {
    "hh_m_85": FloatHistogram(
        name="hh_m_85",
        binrange=(0, 900_000),
        bins=150,
    ),
    "pairingEfficiencyResolved": IntHistogram(
        name="pairingEfficiencyResolved",
        binrange=(0, 3),
    ),
    "vrJetEfficiencyBoosted": IntHistogram(
        name="vrJetEfficiencyBoosted",
        binrange=(0, 3),
    ),
}

# loop over input files
with File(histOutFile, "w") as outfile:
    for file in filelist:
        print("Making hists for " + filename)
        with uproot.open(file) as file_:
            # access the tree
            tree = file_["AnalysisMiniTree"]
            # make generators to load only a certain amount of events
            generators = Loader.GetGenerators(tree, vars, nEvents=-1)
            for vars_arr in generators:
                for hist in hists:
                    # do analysis on a defined hist
                    values = Analysis.do(hist, vars_arr)
                    # update bin heights per iteration
                    hists[hist].fill(values)

    # write histograms to file
    for hist in hists:
        hists[hist].write(outfile, hist)
