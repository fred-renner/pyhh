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

# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801577.Py8EG_A14NNPDF23LO_XHS_X200_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.h5"
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801619.Py8EG_A14NNPDF23LO_XHS_X2000_S400_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.h5"
histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.h5"
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
            print(vals[0] + vals[1])
            hep.atlas.text(" Simulation", loc=1)
            nEvents = vals[0] + vals[1]
            print(nEvents)
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
            print(nEvents)
            hep.atlas.set_ylabel("VR jets efficiency")
            hep.atlas.set_xlabel("leading H (bin 0), subleading H (bin 1)")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            # plt.legend(loc="upper right")
            plt.savefig(plotPath + "vrJetEfficiencyBoosted_0p2.pdf")
            plt.close()
