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
            # edges = file["triggerRef_leadingLargeRpT"]["edges"]
            # counts = file["triggerRef_leadingLargeRpT"]["histogram"][1:-1].astype(int)
            # values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            # triggerRef_leadingLargeRpT = np.array(
            #     plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            # )

            # edges = file["trigger_leadingLargeRpT"]["edges"]
            # counts = file["trigger_leadingLargeRpT"]["histogram"][1:-1].astype(int)
            # values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            # trigger_leadingLargeRpT = np.array(
            #     plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            # )

            triggerRef_leadingLargeRpT = file["triggerRef_leadingLargeRpT"][
                "histogram"
            ][1:-1]
            trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]

            trigger_leadingLargeRpT_err = tools.getEfficiencyErrors(
                passed=trigger_leadingLargeRpT, total=triggerRef_leadingLargeRpT
            )
            plt.figure()
            plt.grid()

            hep.histplot(
                trigger_leadingLargeRpT / triggerRef_leadingLargeRpT,
                file[hist]["edges"],
                histtype="errorbar",
                yerr=trigger_leadingLargeRpT_err,
                solid_capstyle='projecting',
                capsize=3,
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
            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig(plotPath + "trigger_leadingLargeRpT.pdf")
            plt.close()

        if "trigger_leadingLargeRm" in hist:
            # normalize + cumulative
            # edges = file["triggerRef_leadingLargeRm"]["edges"]
            # counts = file["triggerRef_leadingLargeRm"]["histogram"][1:-1].astype(int)
            # values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            # triggerRef_leadingLargeRm = np.array(
            #     plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            # )

            # edges = file["trigger_leadingLargeRm"]["edges"]
            # counts = file["trigger_leadingLargeRm"]["histogram"][1:-1].astype(int)
            # values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
            # trigger_leadingLargeRm = np.array(
            #     plt.hist(values, edges, density=True, cumulative=True)[0], dtype=float
            # )

            triggerRef_leadingLargeRm = file["triggerRef_leadingLargeRm"]["histogram"][
                1:-1
            ]
            trigger_leadingLargeRm = file["trigger_leadingLargeRm"]["histogram"][1:-1]
            trigger_leadingLargeRm_err = tools.getEfficiencyErrors(
                passed=trigger_leadingLargeRm, total=triggerRef_leadingLargeRm
            )
            plt.figure()
            hep.histplot(
                trigger_leadingLargeRm / triggerRef_leadingLargeRm,
                file[hist]["edges"],
                histtype="errorbar",
                yerr=trigger_leadingLargeRm_err,
                solid_capstyle='projecting',
                capsize=3,
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
            nTwoLargeR_truth_mhh = file["nTwoLargeR_truth_mhh"]["histogram"][1:-1]
            nTwoSelLargeR_truth_mhh = file["nTwoSelLargeR_truth_mhh"]["histogram"][1:-1]
            btagHigh_2b2b_truth_mhh = file["btagHigh_2b2b_truth_mhh"][
                "histogram"
            ][1:-1]
            triggerPass = nTriggerPass_truth_mhh / nTruthEvents
            twoLargeR = triggerPass * nTwoLargeR_truth_mhh / nTruthEvents
            twoSelLargeR = twoLargeR * nTwoSelLargeR_truth_mhh / nTruthEvents
            twoSelLargeRhave2b = (
                twoSelLargeR * btagHigh_2b2b_truth_mhh / nTruthEvents
            )
            
            btagLow_1b1j=file["btagLow_1b1j_truth_mhh"]["histogram"][1:-1]
            btagLow_2b1j=file["btagLow_2b1j_truth_mhh"]["histogram"][1:-1]
            btagLow_2b2j=file["btagLow_2b2j_truth_mhh"]["histogram"][1:-1]
            btagHigh_1b1b=file["btagHigh_1b1b_truth_mhh"]["histogram"][1:-1]
            btagHigh_2b1b=file["btagHigh_2b1b_truth_mhh"]["histogram"][1:-1]
            btagHigh_2b2b=file["btagHigh_2b2b_truth_mhh"]["histogram"][1:-1]
            
            # errors
            nTriggerPass_err = tools.getEfficiencyErrors(
                passed=nTriggerPass_truth_mhh, total=nTruthEvents
            )
            nTwoLargeR_err = tools.getEfficiencyErrors(
                passed=nTwoLargeR_truth_mhh, total=nTruthEvents
            )
            nTwoSelLargeR_err = tools.getEfficiencyErrors(
                passed=nTwoSelLargeR_truth_mhh, total=nTruthEvents
            )
            nTwoLargeRHave2BtagVR_err = tools.getEfficiencyErrors(
                passed=btagHigh_2b2b_truth_mhh, total=nTruthEvents
            )
            # error propagation
            twoLargeR_err = twoLargeR * np.sqrt(
                np.power(nTriggerPass_err / triggerPass, 2)
                + np.power(nTwoLargeR_err / twoLargeR, 2)
            )
            twoSelLargeR_err = twoSelLargeR * np.sqrt(
                np.power(nTriggerPass_err / triggerPass, 2)
                + np.power(nTwoLargeR_err / twoLargeR, 2)
                + np.power(nTwoSelLargeR_err / twoSelLargeR, 2)
            )
            twoSelLargeRhave2b_err = twoSelLargeR * np.sqrt(
                np.power(nTriggerPass_err / triggerPass, 2)
                + np.power(nTwoLargeR_err / twoLargeR, 2)
                + np.power(nTwoSelLargeR_err / twoSelLargeR, 2)
                + np.power(nTwoLargeRHave2BtagVR_err / twoSelLargeRhave2b, 2)
            )

            plt.figure()
            hep.histplot(
                triggerPass,
                file[hist]["edges"],
                histtype="errorbar",
                label="passed Trigger",
                yerr=nTriggerPass_err,
                alpha=0.75,
                solid_capstyle='projecting',
                capsize=3,
            )
            hep.histplot(
                twoLargeR,
                file[hist]["edges"],
                histtype="errorbar",
                label="≥2 LargeR",
                yerr=twoLargeR_err,
                alpha=0.75,
                solid_capstyle='projecting',
                capsize=3,
            )
            hep.histplot(
                twoSelLargeR,
                file[hist]["edges"],
                histtype="errorbar",
                label="$p_T$>250 GeV, $|\eta\| < 2.0$",
                yerr=twoSelLargeR_err,
                alpha=0.75,
                solid_capstyle='projecting',
                capsize=3,
            )
            hep.histplot(
                twoSelLargeRhave2b,
                file[hist]["edges"],
                histtype="errorbar",
                label="2 b-tagged VR per Large R",
                yerr=twoSelLargeRhave2b_err,
                alpha=0.75,
                solid_capstyle='projecting',
                capsize=3,
            )

            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Acc x Efficiency")
            hep.atlas.set_xlabel("Truth $m_{hh}$ $[GeV]$ ")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            ax.set_ylim([0, 1.4])
            plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.9))
            hep.rescale_to_axessize
            plt.tight_layout()
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
                yerr=True,
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

        if "hh_m_77" in hist:
            plt.figure()
            hh_m_77 = file["hh_m_77"]["histogram"][1:-1]
            nTruthEvents = file["truth_mhh"]["histogram"][1:-1]
            hep.histplot(
                hh_m_77 / nTruthEvents,
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

        if "massplane_77" in hist:
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
            ax.set_aspect("equal")
            plt.tight_layout()
            # plt.legend(loc="upper right")
            plt.savefig(plotPath + "massplane.pdf")
            plt.close()
