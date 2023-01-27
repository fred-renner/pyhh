#!/usr/bin/env python3
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import mplhep as hep
from h5py import File
import os
import logging
import argparse
import colors
import utils


matplotlib.font_manager._rebuild()
plt.style.use(hep.style.ATLAS)


parser = argparse.ArgumentParser()
parser.add_argument("--histFile", type=str, default=None)
args = parser.parse_args()

# fmt: off
SMsignalFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r13144_p5440_TREE.h5"
SMsignalFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_l1cvv1cv1.h5"
ttbarFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_ttbar.h5"
dijetFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_dijet.h5"
run2File = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-run2.h5"

# fmt: on

if args.histFile:
    histFile = args.histFile


# make plot directory
if "histFile" in locals():
    filename = histFile.split("/")
    filename = str(filename[-1]).replace(".h5", "")
else:
    filename = "run2"

logging.info("make plots for " + filename)
plotPath = "/lustre/fs22/group/atlas/freder/hh/run/plots/" + filename + "/"

if not os.path.isdir(plotPath):
    os.makedirs(plotPath)


def getHist(file, name):
    # access [1:-1] to remove underflow and overflow bins
    h = np.array(file[name]["histogram"][1:-1])
    hRaw = np.array(file[name]["histogramRaw"][1:-1])
    edges = np.array(file[name]["edges"][:])
    err = np.sqrt(hRaw)
    return {"h": h, "hRaw": hRaw, "edges": edges, "err": err}


def load(file, blind=True):
    hists = {}
    for key in file.keys():
        if blind and "SR_4b" in key:
            continue
        hists[key] = getHist(file, key)
    return hists


def triggerRef_leadingLargeRpT():
    triggerRef_leadingLargeRpT = file["triggerRef_leadingLargeRpT"]["histogram"][1:-1]
    trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
    trigger_leadingLargeRpT_err = utils.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    plt.savefig(plotPath + "triggerRef_leadingLargeRpT.pdf")
    plt.close()


def triggerRef_leadingLargeRm():
    # normalize + cumulative
    triggerRef_leadingLargeRm = file["triggerRef_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm = file["trigger_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm_err = utils.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
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
    hists_cumulative, hists_cumulative_err = utils.CumulativeEfficiencies(
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
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
    plt.savefig(plotPath + "accEff_mhh.pdf")
    plt.close()


def trigger_leadingLargeRpT():
    plt.figure()
    err = utils.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="lower right")
    plt.savefig(plotPath + "trigger_leadingLargeRpT.pdf")
    plt.close()


def mhh():
    plt.figure()
    # truth_mhh = file["truth_mhh"]["histogram"][1:-1]
    # trigger_leadingLargeRm_err = utils.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
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
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
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
    # trigger_leadingLargeRm_err = utils.getEfficiencyErrors(
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
    # ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    # hep.yscale_legend()
    hep.atlas.label(data=False, lumi="140????", year=None, loc=0)

    plt.tight_layout()
    # ax.get_xaxis().get_offset_text().set_position((2, 0))

    plt.savefig(plotPath + "VR_dR.pdf")
    plt.close()


def massplane(dataType, blind=True):
    plt.figure()
    h_file = globals()[dataType]
    print(h_file)
    xbins = h_file["massplane"]["edges"][0][1:-1]
    ybins = h_file["massplane"]["edges"][1][1:-1]
    histValues = h_file["massplane"]["histogram"][1:-1, 1:-1]
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
    print(h_file["massplane"]["edges"][0][1:-1])
    X, Y = np.meshgrid(xbins, ybins)
    CS1 = plt.contour(X, Y, utils.Xhh(X, Y), [1.6], colors="white", linewidths=1)
    fmt = {}
    strs = ["SR"]
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)

    CS1 = plt.contour(X, Y, utils.CR_hh(X, Y), [100e3], colors="white", linewidths=1)
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
    if blind:
        CS1 = plt.contourf(X, Y, utils.Xhh(X, Y), [0, 1.6], colors="black")

    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.get_yaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
    ax.yaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
    ax.set_aspect("equal")

    # plt.legend(loc="upper right")
    plt.savefig(plotPath + "massplane_" + dataType + ".pdf")
    plt.close()


def mh_SB_ratio(histKey):

    # make both ratios, s/sqrt(B) and over Data
    signal = SMsignal[histKey]["h"]
    tt = ttbar[histKey]["h"]
    jj = dijet[histKey]["h"]
    edges = SMsignal[histKey]["edges"]
    # need to correct error

    bkg_tot = tt + jj
    bkg_tot_err = np.sqrt(ttbar[histKey]["err"] ** 2 + dijet[histKey]["err"] ** 2)
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
    # print(tt.shape)
    # print(jj.shape)
    # print(edges.shape)
    # # stack plot
    hep.histplot(
        [tt, jj],
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
        color="dimgrey",
        linewidth=0,
        alpha=0.5,
        step="post",
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
    # rax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i", offset=False))
    ax.legend(loc="upper right")
    plt.savefig(plotPath + f"SB_{histKey}_ratio.pdf")
    plt.close()


def mh_data_ratio(histKey):

    # wollen VR_4b ansehsen

    # wir wollen verschieden regionen, mh1 mh2 mhh,

    # would like datapoints in VR_4b für m_hh
    # bkg estimate dazu: data_VR_2b_weights-ttbar_VR_4b, ttbar_VR_4b

    # # sm = SMsignal[histKey + "VR_4b"]["h"]
    # tt = ttbar[histKey + "VR_4b"]["h"]
    # jj = (run2[histKey + "VR_2b"]["h"] - ttbar[histKey + "VR_2b"]["h"]) * 0.000561456456456456
    # r2= jj+tt
    # # pred = jj + tt == r2
    # ratio = (run2[histKey + "VR_4b"]["h"] - r2) / r2
    # # need to correct error
    # edges = ttbar[histKey + "VR_2b"]["edges"]
    # # bkg = np.array([ttbar[histKey]["hRaw"], bkgEstimate])
    # # bkg_err = np.sqrt(bkg_tot)

    # B = Q + T
    # Berr = sqrt(Qerr^2 + Terr^2)

    tt = ttbar[histKey]["h"]
    jj = dijet[histKey]["h"]
    data = run2[histKey]["h"]
    edges = dijet[histKey]["edges"]

    plt.figure()

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True,
    )

    #  stack plot
    hep.histplot(
        [tt, jj],
        edges,
        stack=True,
        histtype="fill",
        # yerr=True,
        label=["$t\overline{t}$", "Multijet"],  # "run2_VR_2b_weighted"],
        ax=ax,
        color=["hh:darkpink", "hh:medturquoise"],
        edgecolor="black",
        linewidth=0.5,
    )
    # error stackplot
    # ax.fill_between(
    #     edges,
    #     np.append(bkg_tot - bkg_tot_err, 0),
    #     np.append(bkg_tot + bkg_tot_err, 0),
    #     color="dimgrey",
    #     linewidth=0,
    #     alpha=0.5,
    #     step="post",
    #     label="stat. uncertainty",
    # )
    # data
    hep.histplot(
        data,
        edges,
        histtype="errorbar",
        yerr=True,
        color="Black",
        label="data",
        ax=ax,
    )
    # ratio plot
    hep.histplot(
        data / (tt + jj),
        edges,
        histtype="errorbar",
        yerr=True,
        ax=rax,
        color="Black",
    )
    fig.subplots_adjust(hspace=0.07)
    ax.set_ylabel("Events")
    rax.set_ylabel(
        r"$ \frac{\mathrm{Data}}{\mathrm{Pred}}$", horizontalalignment="center"
    )
    # rax.set_ylim([0, 0.0001])
    ax.set_yscale("log")
    # ax.set_ylim([0, 1_000_000])

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
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))

    plt.savefig(plotPath + f"{histKey}_ratio.pdf")
    plt.close()


def compareABCD(histKey):
    r2 = run2[histKey + "_VR_4b"]["h"]
    tt = ttbar[histKey + "_VR_2b"]["h"]
    jj = dijet[histKey + "_VR_4b"]["h"]
    bkgEstimate = run2[histKey + "_VR_2b"]["h"] - tt
    edges = run2[histKey + "_VR_4b"]["edges"]

    bkg_tot = bkgEstimate + tt
    bkg_tot_err = np.sqrt(run2[histKey + "_VR_2b_weights"]["h"])
    plt.figure()
    hep.histplot(
        [tt, bkgEstimate],
        edges,
        stack=True,
        histtype="fill",
        # yerr=True,
        label=["$t\overline{t}$", "Bkg estimate"],  # , "Multijet"],
        color=["hh:darkpink", "hh:medturquoise"],
        edgecolor="black",
        linewidth=0.5,
    )
    hep.histplot(
        # cumulative_signal/cumulative_bkg,
        r2,
        edges,
        histtype="errorbar",
        yerr=True,
        color="Black",
        label="data",
    )
    ax = plt.gca()
    ax.fill_between(
        edges,
        np.append(bkg_tot + bkg_tot_err, 0),
        np.append(bkg_tot - bkg_tot_err, 0),
        hatch="\\\\\\\\",
        facecolor="None",
        edgecolor="dimgrey",
        linewidth=0,
        step="post",
        zorder=1,
        label="stat. uncertainty",
    )
    # hep.atlas.text(" Simulation", loc=1)
    hep.atlas.set_ylabel("Events")
    hep.atlas.set_xlabel(f"{histKey}")
    ax = plt.gca()
    # ax.set_yscale("log")
    # ax.xaxis.set_major_formatter(utils.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    # hep.yscale_legend()
    hep.atlas.label(data=False, lumi="140", year=None, loc=0)

    plt.tight_layout()
    # ax.get_xaxis().get_offset_text().set_position((2, 0))

    plt.savefig(plotPath + histKey + "_BkgEstimate.pdf")
    plt.close()

blind = True

with File(SMsignalFile, "r") as f_SMsignal, File(run2File, "r") as f_run2, File(
    ttbarFile, "r"
) as f_ttbar, File(dijetFile, "r") as f_dijet:
    SMsignal = load(f_SMsignal, blind=False)
    run2 = load(f_run2, blind=blind)
    ttbar = load(f_ttbar, blind=False)
    dijet = load(f_dijet, blind=False)
    # trigger_leadingLargeRpT()
    # triggerRef_leadingLargeRpT()
    # triggerRef_leadingLargeRm()
    # accEff_mhh()
    # mhh()
    # for name in ["pt_h1", "pt_h2", "pt_hh", "pt_hh_scalar"]:
    #     pts(name)
    # dRs()
    massplane("f_run2",blind=blind)
    massplane("f_SMsignal",blind=False)
    massplane("f_ttbar",blind=False)
    # mh_SB_ratio("mh1_VR_")
    # mh_SB_ratio("mh2")
    # mh_data_ratio("mhh_")

    # for region in ["SR_2b", "VR_4b", "VR_2b", "CR_4b", "CR_2b"]:
    #     # mh_SB_ratio("mh1_" + region)
    #     # mh_SB_ratio("mh2_" + region)
    #     mh_SB_ratio("mhh_" + region)
    mh_data_ratio("mhh_CR_2b")
    mh_data_ratio("mhh_CR_4b")
    # compareABCD("mhh")
