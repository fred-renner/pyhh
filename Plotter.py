#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import ticker
import mplhep as hep
from h5py import File, Group, Dataset
import os
import logging
import argparse


# quick and dirty color log
logging.basicConfig(level=logging.INFO)
logging.addLevelName(
    logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO)
)

matplotlib.font_manager._rebuild()
plt.style.use(hep.style.ATLAS)

# plt.rc("text", usetex=True)

# for debug
# print(file[hist]["histogram"][1:-1])
# print(file[hist]["edges"])[:]

parser = argparse.ArgumentParser()
parser.add_argument("--histFile", type=str, default=None)
args = parser.parse_args()

# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801577.Py8EG_A14NNPDF23LO_XHS_X200_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.h5"
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801619.Py8EG_A14NNPDF23LO_XHS_X2000_S400_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.h5"
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.h5"
histFile = (
    "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables.h5"
)


if args.histFile:
    histFile = args.histFile

# make plot directory
filename = histFile.split("/")
filename = str(filename[-1]).replace(".h5", "")
logging.info("make plots for " + filename)
plotPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/plots-" + filename + "/"
if not os.path.isdir(plotPath):
    os.makedirs(plotPath)

with File(histFile, "r") as file:
    for hist in file.keys():

        # access [1:-1] to remove underflow and overflow bins

        if "events_truth_mhh" in hist:
            plt.figure()
            nTruthEvents = file[hist]["histogram"][1:-1]
            nTwoSelLargeR_truth_mhh = file["nTwoSelLargeR_truth_mhh"]["histogram"][1:-1]
            hep.histplot(
                nTwoSelLargeR_truth_mhh / nTruthEvents,
                file[hist]["edges"],
                label=">2 nLargeR ",
                yerr=True,
                density=False,
                # alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Acc x Efficiency")
            hep.atlas.set_xlabel("$m_{hh}$ $[GeV]$ ")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            ax.containers[0].fmt = "o"
            ax.containers[0].linewidth = 2
            ax.containers[0].capsize = 6

            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig(plotPath + "accEff_truth_mhh.pdf")
            plt.close()

        if "hh_m" in hist:
            plt.figure()
            hep.histplot(
                file[hist]["histogram"][1:-1],
                file[hist]["edges"],
                label="HH mass",
                density=False,
                alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Events")
            hep.atlas.set_xlabel("$m_{hh}$ $[GeV]$ ")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            plt.tight_layout()
            # plt.legend(loc="upper right")
            plt.savefig(plotPath + "m_hh.pdf")
            plt.close()

        if "pairingEfficiencyResolved" in hist:
            plt.figure()
            vals = file[hist]["histogram"][1:-1]
            hep.histplot(
                [vals[1] / (vals[0] + vals[1]), vals[3] / (vals[2] + vals[3])],
                file[hist]["edges"][:3],
                label=hist,
                density=False,
                alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            nEvents = vals[0] + vals[1]
            hep.atlas.set_ylabel("Pairing efficiency")
            hep.atlas.set_xlabel("leading H (bin 0), subleading H (bin 1)")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            # plt.legend(loc="upper right")
            plt.savefig(plotPath + "pairingEfficiencyResolved_0p2.pdf")
            plt.close()

        if "vrJetEfficiencyBoosted" in hist:
            plt.figure()
            vals = file[hist]["histogram"][1:-1]
            hep.histplot(
                [vals[1] / (vals[0] + vals[1]), vals[3] / (vals[2] + vals[3])],
                file[hist]["edges"][:3],
                label=hist,
                density=False,
                alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            nEvents = vals[2] + vals[3]
            hep.atlas.set_ylabel("VR jets efficiency")
            hep.atlas.set_xlabel("leading H (bin 0), subleading H (bin 1)")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            # plt.legend(loc="upper right")
            plt.savefig(plotPath + "vrJetEfficiencyBoosted_0p2.pdf")
            plt.close()

        if "massplane_85" in hist:
            plt.figure()
            histValues = file[hist]["histogram"][1:-1, 1:-1]
            hep.hist2dplot(
                histValues,
                xbins=file[hist]["edges"][0][1:-1],
                ybins=file[hist]["edges"][1][1:-1],
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("$m_{h2}$ $[GeV]$ ")
            hep.atlas.set_xlabel("$m_{h1}$ $[GeV]$ ")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            plt.tight_layout()
            # plt.legend(loc="upper right")
            plt.savefig(plotPath + "massplane.pdf")
            plt.close()
