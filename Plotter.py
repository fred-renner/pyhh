#!/usr/bin/env python3
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import ticker
import mplhep as hep
from h5py import File, Group, Dataset
import os
import logging
import argparse
import PlottingTools as tools

# quick and dirty color log
logging.basicConfig(level=logging.INFO)
logging.addLevelName(
    logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO)
)

matplotlib.font_manager._rebuild()
plt.style.use(hep.style.ATLAS)


parser = argparse.ArgumentParser()
parser.add_argument("--histFile", type=str, default=None)
args = parser.parse_args()


# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801577.Py8EG_A14NNPDF23LO_XHS_X200_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.h5"
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801619.Py8EG_A14NNPDF23LO_XHS_X2000_S400_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.h5"
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.h5"
histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc21_13p6TeV.601479.PhPy8EG_HH4b_cHHH01d0.deriv.DAOD_PHYS.e8472_s3873_r13829_p5440_.h5"
# histFile = (
#     "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables.h5"
# )
histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE.h5"

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

        if "trigger_leadingLargeRpT" in hist:
            # normalize + cumulative
            edges = file["triggerRef_leadingLargeRpT"]["edges"]
            counts = file["triggerRef_leadingLargeRpT"]["histogram"][1:-1].astype(int)
            values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            triggerRef_leadingLargeRpT = np.array(
                plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            )

            edges = file["trigger_leadingLargeRpT"]["edges"]
            counts = file["trigger_leadingLargeRpT"]["histogram"][1:-1].astype(int)
            values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            trigger_leadingLargeRpT = np.array(
                plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            )

            # print(trigger_leadingLargeRpT)
            # triggerRef_leadingLargeRpT = file["triggerRef_leadingLargeRpT"][
            #     "histogram"
            # ][1:-1]
            # trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]

            # trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
            # triggerRef_leadingLargeRpT_err = tools.getEfficiencyErrors(
            #     passed=trigger_leadingLargeRpT, total=triggerRef_leadingLargeRpT
            # )
            # trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
            # triggerRef_leadingLargeRpT_err = tools.getEfficiencyErrors(
            #     passed=trigger_leadingLargeRpT, total=triggerRef_leadingLargeRpT
            # )
            plt.figure()
            hep.histplot(
                trigger_leadingLargeRpT / triggerRef_leadingLargeRpT,
                file[hist]["edges"],
                histtype="errorbar",
                yerr=False,
                # density=True,
                # alpha=0.75,
            )

            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Cumulative Trigger eff.")
            hep.atlas.set_xlabel("Leading Large R Jet p$_T$ $[GeV]$ ")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))

            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            ax.set_ylim([0.8, 1.05])
            ax.set_xlim([0.8, 2500_000])
            plt.grid()
            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig(plotPath + "trigger_leadingLargeRpT.pdf")
            plt.close()

        if "trigger_leadingLargeRm" in hist:
            # normalize + cumulative
            edges = file["triggerRef_leadingLargeRm"]["edges"]
            counts = file["triggerRef_leadingLargeRm"]["histogram"][1:-1].astype(int)
            values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            triggerRef_leadingLargeRm = np.array(
                plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            )

            edges = file["trigger_leadingLargeRm"]["edges"]
            counts = file["trigger_leadingLargeRm"]["histogram"][1:-1].astype(int)
            values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            trigger_leadingLargeRm = np.array(
                plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            )

            # print(trigger_leadingLargeRpT)
            # triggerRef_leadingLargeRpT = file["triggerRef_leadingLargeRpT"][
            #     "histogram"
            # ][1:-1]
            # trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]

            # trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
            # triggerRef_leadingLargeRpT_err = tools.getEfficiencyErrors(
            #     passed=trigger_leadingLargeRpT, total=triggerRef_leadingLargeRpT
            # )
            # trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
            # triggerRef_leadingLargeRpT_err = tools.getEfficiencyErrors(
            #     passed=trigger_leadingLargeRpT, total=triggerRef_leadingLargeRpT
            # )
            plt.figure()
            hep.histplot(
                trigger_leadingLargeRm / triggerRef_leadingLargeRm,
                file[hist]["edges"],
                histtype="errorbar",
                yerr=False,
                # density=True,
                # alpha=0.75,
            )

            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Cumulative Trigger eff.")
            hep.atlas.set_xlabel("Leading Large R Jet Mass $[GeV]$ ")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            plt.grid()

            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig(plotPath + "trigger_leadingLargeRm.pdf")
            plt.close()

        if "nTriggerPass_truth_mhh" in hist:
            nTruthEvents = file["truth_mhh"]["histogram"][1:-1]
            nTriggerPass_truth_mhh = file["nTriggerPass_truth_mhh"]["histogram"][1:-1]

            nTwoSelLargeR_truth_mhh = file["nTwoSelLargeR_truth_mhh"]["histogram"][1:-1]
            nTriggerPass_truth_mhh_err = tools.getEfficiencyErrors(
                passed=nTriggerPass_truth_mhh, total=nTruthEvents
            )
            nTwoSelLargeR_truth_mhh_err = tools.getEfficiencyErrors(
                passed=nTwoSelLargeR_truth_mhh, total=nTruthEvents
            )
            triggerPass = nTriggerPass_truth_mhh / nTruthEvents
            twoLargeR = triggerPass * nTwoSelLargeR_truth_mhh / nTruthEvents

            hep.histplot(
                [triggerPass, twoLargeR],
                file[hist]["edges"],
                histtype="errorbar",
                label=["passed Trigger", "≥2 LargeR"],
                yerr=False,
                # density=False,
                # w2=np.ones(triggerPass.shape[0])*0.001,
                alpha=0.75,
            )
            # plt.errorbar(file[hist]["edges"][:-1], triggerPass, nTriggerPass_truth_mhh_err)

            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Acc x Efficiency")
            hep.atlas.set_xlabel("Truth $m_{hh}$ $[GeV]$ ")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))

            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            # ax.set_ylim([0, 1.2])

            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            plt.legend(loc="lower right")
            plt.savefig(plotPath + "accEff_truth_mhh.pdf")
            plt.close()

        if "leadingLargeRpT" in hist:
            plt.figure()
            leadingLargeRpT = file["leadingLargeRpT"]["histogram"][1:-1]
            triggerRef_leadingLargeRpT = file["triggerRef_leadingLargeRpT"][
                "histogram"
            ][1:-1]
            trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
            hep.histplot(
                [triggerRef_leadingLargeRpT, trigger_leadingLargeRpT],
                file[hist]["edges"],
                histtype="errorbar",
                yerr=False,
                density=False,
                # alpha=0.75,
                label=["triggerRef_leadingLargeRpT", "trigger_leadingLargeRpT"],
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Events")
            hep.atlas.set_xlabel("Leading Large R Jet p$_T$ $[GeV]$ ")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig(plotPath + "leadingLargeRpT.pdf")
            plt.close()

        if "truth_mhh" in hist:
            plt.figure()
            hh_m_85 = file["hh_m_85"]["histogram"][1:-1]
            nTruthEvents = file["truth_mhh"]["histogram"][1:-1]
            hep.histplot(
                [nTruthEvents, hh_m_85],
                file[hist]["edges"],
                histtype="errorbar",
                yerr=True,
                density=False,
                label=["truth", "reco"]
                # alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Events")
            hep.atlas.set_xlabel("$m_{hh}$ $[GeV]$ ")
            ax = plt.gca()
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            plt.legend(loc="upper right")
            hep.yscale_legend()
            plt.savefig(plotPath + "truth_mhh.pdf")
            plt.close()

        if "truth_mhh" in hist:
            plt.figure()
            hh_m_85 = file["hh_m_85"]["histogram"][1:-1]
            nTruthEvents = file["truth_mhh"]["histogram"][1:-1]
            hep.histplot(
                hh_m_85 / nTruthEvents,
                file[hist]["edges"],
                histtype="errorbar",
                yerr=False,
                density=False,
                # label=["truth", "reco"]
                # alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Events (reco / truth)")
            hep.atlas.set_xlabel("$m_{hh}$ $[GeV]$ ")
            ax = plt.gca()
            # ax.set_yscale("log")
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            plt.legend(loc="upper right")
            # hep.yscale_legend()
            plt.tight_layout()
            plt.savefig(plotPath + "truth_reco_ratio_mhh.pdf")
            plt.close()

        if "hh_m_85" in hist:
            plt.figure()
            hh_m_85 = file["hh_m_85"]["histogram"][1:-1]
            hep.histplot(
                hh_m_85,
                file[hist]["edges"],
                histtype="errorbar",
                yerr=False,
                density=False,
                # alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Events")
            hep.atlas.set_xlabel("$m_{hh}$ $[GeV]$ ")
            ax = plt.gca()
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig(plotPath + "reco_mhh.pdf")
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
            txt = hep.atlas.text(" Simulation", loc=1)
            txt[0]._color = "white"
            txt[1]._color = "white"
            hep.atlas.set_ylabel("$m_{h2}$ $[GeV]$ ")
            hep.atlas.set_xlabel("$m_{h1}$ $[GeV]$ ")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            ax.yaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            ax.set_aspect('equal')
            plt.tight_layout()
            # plt.legend(loc="upper right")
            plt.savefig(plotPath + "massplane.pdf")
            plt.close()
