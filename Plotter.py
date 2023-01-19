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
import colors

# quick and dirty color log
# logging.basicConfig(level=logging.INFO)
# logging.addLevelName(
#     logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO)
# )

matplotlib.font_manager._rebuild()
plt.style.use(hep.style.ATLAS)


parser = argparse.ArgumentParser()
parser.add_argument("--histFile", type=str, default=None)
args = parser.parse_args()

# fmt: off
withBackground=True
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE.h5"
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-user.frenner.HH4b.2022_11_25_.601480.PhPy8EG_HH4b_cHHH10d0.e8472_s3873_r13829_p5440_TREE.h5"
histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r13144_p5440_TREE.h5"
# histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-MC20-signal-1cvv1cv1.h5"
ttbarHists = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-MC20-bkg-ttbar.h5"
dijetHists = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-MC20-bkg-dijet.h5"
# fmt: on

if args.histFile:
    histFile = args.histFile

# make plot directory
filename = histFile.split("/")
filename = str(filename[-1]).replace(".h5", "")
logging.info("make plots for " + filename)
plotPath = "/lustre/fs22/group/atlas/freder/hh/run/plots/" + filename + "/"
if not os.path.isdir(plotPath):
    os.makedirs(plotPath)


def getHist(file, name):
    # access [1:-1] to remove underflow and overflow bins
    h = file[name]["histogram"][1:-1]
    hRaw = file[name]["histogramRaw"][1:-1]
    edges = file[name]["edges"][:]
    err = np.sqrt(file[name]["w2sum"][1:-1])
    return {"h": h, "hRaw": hRaw, "edges": edges, "err": err}


def CountBackground():
    print("N_CR_4b", hists["N_CR_4b"]["hRaw"])
    w_CR = hists["N_CR_4b"]["hRaw"] / hists["N_CR_2b"]["hRaw"]
    w_VR = hists["N_VR_4b"]["hRaw"] / hists["N_VR_2b"]["hRaw"]
    print(locals())


def triggerRef_leadingLargeRpT():
    triggerRef_leadingLargeRpT = file["triggerRef_leadingLargeRpT"]["histogram"][1:-1]
    trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
    trigger_leadingLargeRpT_err = tools.getEfficiencyErrors(
        passed=trigger_leadingLargeRpT, total=triggerRef_leadingLargeRpT
    )
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

    plt.figure()
    plt.grid()

    hep.histplot(
        trigger_leadingLargeRpT / triggerRef_leadingLargeRpT,
        file["triggerRef_leadingLargeRpT"]["edges"],
        histtype="errorbar",
        yerr=trigger_leadingLargeRpT_err,
        solid_capstyle="projecting",
        capsize=3,
        # density=True,
        alpha=0.75,
    )

    hep.atlas.text(" Simulation", loc=1)
    # hep.atlas.set_ylabel("Cumulative Trigger eff.")
    hep.atlas.set_ylabel("Trigger eff.")
    hep.atlas.set_xlabel("Leading Large R Jet p$_T$ $[GeV]$ ")
    ax = plt.gca()
    # ax.set_ylim([0.8, 1.05])
    # ax.set_xlim([0.8, 2500_000])
    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    plt.savefig(plotPath + "triggerRef_leadingLargeRpT.pdf")
    plt.close()


def triggerRef_leadingLargeRm():
    # normalize + cumulative
    triggerRef_leadingLargeRm = file["triggerRef_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm = file["trigger_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm_err = tools.getEfficiencyErrors(
        passed=trigger_leadingLargeRm, total=triggerRef_leadingLargeRm
    )
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

    plt.figure()
    hep.histplot(
        trigger_leadingLargeRm / triggerRef_leadingLargeRm,
        file["trigger_leadingLargeRm"]["edges"],
        histtype="errorbar",
        yerr=trigger_leadingLargeRm_err,
        solid_capstyle="projecting",
        capsize=3,
        # density=True,
        alpha=0.75,
    )

    hep.atlas.text(" Simulation", loc=1)
    # hep.atlas.set_ylabel("Cumulative Trigger eff.")
    hep.atlas.set_ylabel("Trigger eff.")
    hep.atlas.set_xlabel("Leading Large R Jet Mass $[GeV]$ ")
    ax = plt.gca()
    ax.set_ylim([0, 2])

    plt.grid()

    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    plt.savefig(plotPath + "triggerRef_leadingLargeRm.pdf")
    plt.close()


def accEff_mhh():
    keys = [
        "nTriggerPass_mhh",
        "nTwoLargeR_mhh",
        "nTwoSelLargeR_mhh",
        # "btagLow_1b1j_mhh",
        # "btagLow_2b1j_mhh",
        # "btagLow_2b2j_mhh",
        # "btagHigh_1b1b_mhh",
        # "btagHigh_2b1b_mhh",
        # "btagHigh_2b2b_mhh",
    ]
    hists_ = []
    for key in keys:
        print(key)
        hists_.append(hists[key]["hRaw"])
        print(hists[key]["hRaw"])
    print(hists["mhh"]["hRaw"])
    hists_cumulative, hists_cumulative_err = tools.CumulativeEfficiencies(
        hists_, baseline=hists["mhh_twoLargeR"]["hRaw"], stopCumulativeFrom=4
    )
    labels = [
        "passed Trigger",
        "≥2 LargeR",
        "Kinematic Selection"
        # "btagLow 1b1j",
        # "btagLow 2b1j",
        # "btagLow 2b2j",
        # "btagHigh 1b1b",
        # "btagHigh 2b1b",
        # "btagHigh 2b2b",
    ]
    #         "$p_T(H1)$>450 GeV, $p_T(H2)$>250 GeV, $|\eta\| < 2.0$",

    plt.figure()

    for i, (h, err, label) in enumerate(
        zip(hists_cumulative, hists_cumulative_err, labels)
    ):
        hep.histplot(
            h,
            file["nTriggerPass_mhh"]["edges"],
            histtype="errorbar",
            label=label,
            yerr=err,
            alpha=0.7,
            solid_capstyle="projecting",
            capsize=3,
            color="C{}".format(i),
        )

    hep.atlas.text(" Simulation", loc=1)
    hep.atlas.set_ylabel("Acc x Efficiency")
    hep.atlas.set_xlabel("$m_{hh}$ $[GeV]$ ")
    ax = plt.gca()

    # ax.set_ylim([0, 1.2])
    plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.9))
    hep.rescale_to_axessize
    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    plt.savefig(plotPath + "accEff_mhh.pdf")
    plt.close()


def trigger_leadingLargeRpT():
    plt.figure()
    err = tools.getEfficiencyErrors(
        passed=hists["leadingLargeRpT_trigger"]["hRaw"],
        total=hists["leadingLargeRpT"]["hRaw"],
    )
    hep.histplot(
        hists["leadingLargeRpT_trigger"]["h"] / hists["leadingLargeRpT"]["h"],
        file["leadingLargeRpT"]["edges"],
        histtype="errorbar",
        yerr=err,
        density=False,
        # alpha=0.75,
        solid_capstyle="projecting",
        capsize=3,
        label="trigPassed_HLT_j420_a10_lcw_L1J100",
    )
    hep.atlas.text(" Simulation", loc=1)
    hep.atlas.set_ylabel("Trigger efficiency")
    hep.atlas.set_xlabel("Leading Large R Jet p$_T$ $[GeV]$ ")
    ax = plt.gca()
    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="lower right")
    plt.savefig(plotPath + "trigger_leadingLargeRpT.pdf")
    plt.close()


def mhh():
    plt.figure()
    # truth_mhh = file["truth_mhh"]["histogram"][1:-1]
    # trigger_leadingLargeRm_err = tools.getEfficiencyErrors(
    #     passed=mhh, total=truth_mhh
    # )
    hep.histplot(
        hists["mhh"]["h"],  # / truth_mhh,
        hists["mhh"]["edges"],  # / truth_mhh,
        histtype="errorbar",
        yerr=hists["mhh"]["err"],  # / truth_mhh,
        solid_capstyle="projecting",
        capsize=3,
        # label=["truth", "reco"]
        # alpha=0.75,
    )
    # hep.atlas.text(" Simulation", loc=1)
    hep.atlas.set_ylabel("Events")
    hep.atlas.set_xlabel("$m_{hh}$ $[GeV]$ ")
    ax = plt.gca()
    # ax.set_yscale("log")
    ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    # hep.yscale_legend()
    hep.atlas.label(data=False, lumi="140????", year=None, loc=0)

    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))

    plt.savefig(plotPath + "mhh.pdf")
    plt.close()


def pts(name):
    plt.figure()
    hep.histplot(
        hists[name]["h"],  # / truth_mhh,
        hists[name]["edges"],  # / truth_mhh,
        histtype="errorbar",
        yerr=hists[name]["err"],  # / truth_mhh,
        solid_capstyle="projecting",
        capsize=3,
        # label=["truth", "reco"]
        # alpha=0.75,
    )
    # hep.atlas.text(" Simulation", loc=1)
    hep.atlas.set_ylabel("Events")
    hep.atlas.set_xlabel(f"{name} $[GeV]$ ")
    ax = plt.gca()
    # ax.set_yscale("log")
    ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    # hep.yscale_legend()
    hep.atlas.label(data=False, lumi="140????", year=None, loc=0)

    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))

    plt.savefig(plotPath + f"{name}.pdf")
    plt.close()


def dRs():
    plt.figure()
    dR_h1 = getHist(file, "dR_h1")
    dR_h2 = getHist(file, "dR_h2")
    # truth_mhh = file["truth_mhh"]["histogram"][1:-1]
    # trigger_leadingLargeRm_err = tools.getEfficiencyErrors(
    #     passed=mhh, total=truth_mhh
    # )
    hep.histplot(
        [dR_h1["h"], dR_h2["h"]],
        dR_h1["edges"],
        histtype="errorbar",
        yerr=[dR_h1["err"], dR_h2["err"]],
        solid_capstyle="projecting",
        capsize=3,
        label=["H1", "H2"],
        alpha=0.75,
    )
    # hep.atlas.text(" Simulation", loc=1)
    hep.atlas.set_ylabel("Events")
    hep.atlas.set_xlabel("DeltaR leading VR jets")
    ax = plt.gca()
    # ax.set_yscale("log")
    # ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    # hep.yscale_legend()
    hep.atlas.label(data=False, lumi="140????", year=None, loc=0)

    plt.tight_layout()
    # ax.get_xaxis().get_offset_text().set_position((2, 0))

    plt.savefig(plotPath + "VR_dR.pdf")
    plt.close()


def massplane_77():
    plt.figure()
    xbins = file["massplane_77"]["edges"][0][1:-1]
    ybins = file["massplane_77"]["edges"][1][1:-1]
    histValues = file["massplane_77"]["histogram"][1:-1, 1:-1]
    hep.hist2dplot(
        histValues,
        xbins=xbins,
        ybins=ybins,
    )
    txt = hep.atlas.text(" Simulation", loc=1)
    txt[0]._color = "white"
    txt[1]._color = "white"
    hep.atlas.set_ylabel("$m_{H2}$ $[GeV]$ ")
    hep.atlas.set_xlabel("$m_{H1}$ $[GeV]$ ")
    ax = plt.gca()

    X, Y = np.meshgrid(xbins, ybins)
    CS1 = plt.contour(X, Y, tools.Xhh(X, Y), [1.6], colors="white", linewidths=1)
    fmt = {}
    strs = ["SR"]
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)

    CS1 = plt.contour(X, Y, tools.CR_hh(X, Y), [100e3], colors="white", linewidths=1)
    fmt = {}
    strs = ["VR"]
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)

    CS1 = plt.contour(X, Y, X + Y, [130e3], colors="white", linewidths=1)
    fmt = {}
    strs = ["CR"]
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)

    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.get_yaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    ax.yaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i"))
    ax.set_aspect("equal")

    # plt.legend(loc="upper right")
    plt.savefig(plotPath + "massplane.pdf")
    plt.close()


def vrJetEfficiencyBoosted():
    plt.figure()
    vals = file["vrJetEfficiencyBoosted"]["histogram"][1:-1]
    hep.histplot(
        [vals[1] / (vals[0] + vals[1]), vals[3] / (vals[2] + vals[3])],
        file["vrJetEfficiencyBoosted"]["edges"][:3],
        label="vrJetEfficiencyBoosted",
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


def mh_ratio(histKey):
    signal, edges, signal_err = getHist(file, histKey)
    ttbar, edges2, ttbar_err = getHist(ttbarFile, histKey)
    dijet, edges2, dijet_err = getHist(dijetFile, histKey)

    bkg = np.array([ttbar, dijet])
    bkg_tot = np.sum(bkg, axis=0)
    bkg_tot_err = np.sqrt(bkg_tot)
    ratio = signal / np.sqrt(bkg_tot)

    # B = Q + T
    # Berr = sqrt(Qerr^2 + Terr^2)

    # values_signal = np.repeat((edges[:-1] + edges[1:]) / 2.0, signal.astype(int))
    # values_bkg = np.repeat((edges[:-1] + edges[1:]) / 2.0, bkg_tot.astype(int))
    # cumulative_signal = np.array(
    #     plt.hist(
    #         values_signal,
    #         edges,
    #         # density=True,
    #         cumulative=True,
    #     )[0],
    #     dtype=float,
    # )
    # cumulative_bkg = np.array(
    #     plt.hist(
    #         values_bkg,
    #         edges,
    #         # density=True,
    #         cumulative=True,
    #     )[0],
    #     dtype=float,
    # )

    plt.figure()

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True,
    )

    # stack plot
    hep.histplot(
        [ttbar, dijet],
        edges,
        stack=True,
        histtype="fill",
        # yerr=True,
        label=["$t\overline{t}$", "Multijet"],
        ax=ax,
        color=["hh:darkpink", "hh:medturquoise"],
        edgecolor="black",
        linewidth=0.5,
    )
    # error stackplot
    ax.fill_between(
        edges,
        np.append(bkg_tot - bkg_tot_err, 0),
        np.append(bkg_tot + bkg_tot_err, 0),
        hatch="\\\\\\\\",
        facecolor="None",
        edgecolor="dimgrey",
        linewidth=0,
        step="post",
        zorder=1,
        label="stat. uncertainty",
    )
    # signal
    hep.histplot(
        signal * 10000,
        edges,
        histtype="step",
        # yerr=True,
        label="SM Signal x 10000",
        ax=ax,
        color="hh:darkyellow",  # "orangered",
        linewidth=1.25,
    )

    # ratio plot
    hep.histplot(
        # cumulative_signal/cumulative_bkg,
        ratio,
        edges,
        histtype="errorbar",
        yerr=True,
        ax=rax,
        color="Black",
    )
    fig.subplots_adjust(hspace=0.07)
    ax.set_ylabel("Events")
    rax.set_ylabel("$S/\sqrt{B}$")
    rax.set_ylim([0, 0.0001])
    ax.set_yscale("log")
    ax.set_ylim([0, 1_000_000])

    hep.atlas.label(data=False, lumi="140.0", loc=0, ax=ax)
    if "mh1" in histKey:
        whichHiggs = "H1"
    if "mh2" in histKey:
        whichHiggs = "H2"
    if "hh" in histKey:
        whichHiggs = "HH"
    hep.atlas.set_xlabel(f"$m_{{{whichHiggs}}}$ $[GeV]$ ")
    plt.tight_layout()
    rax.get_xaxis().get_offset_text().set_position((2, 0))
    rax.xaxis.set_major_formatter(tools.OOMFormatter(3, "%1.1i", offset=False))
    ax.legend(loc="upper right")
    plt.savefig(plotPath + f"{histKey}_ratio.pdf")
    plt.close()


with File(histFile, "r") as file:
    hists = {}
    for key in file.keys():
        hists[key] = getHist(file, key)
    # trigger_leadingLargeRpT()
    # triggerRef_leadingLargeRpT()
    # triggerRef_leadingLargeRm()
    accEff_mhh()
    # mhh()
    # for name in ["pt_h1", "pt_h2", "pt_hh", "pt_hh_scalar"]:
    #     pts(name)
    # dRs()
    CountBackground()
    massplane_77()
    # if withBackground:
    #     with File(ttbarHists, "r") as ttbarFile, File(dijetHists, "r") as dijetFile:
    #         mh_ratio("mh1")
    #         mh_ratio("mh2")
    #         mh_ratio("btagHigh_2b2b_mhh")
