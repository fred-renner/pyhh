#!/usr/bin/env python3
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import mplhep as hep
import os
import logging
import argparse
import Plotting.loadHists
import Plotting.colors
import Plotting.tools
from Plotting.tools import ErrorPropagation as propagateError
from HistDefs import collectedKinVars, collectedKinVarsWithRegions
from matplotlib import ticker as mticker
from pdf2image import convert_from_path
from tools.logging import log

np.seterr(divide="ignore", invalid="ignore")

plt.style.use(hep.style.ATLAS)


parser = argparse.ArgumentParser()
parser.add_argument("--histFile", type=str, default=None)
args = parser.parse_args()


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


def savegrid(ims, plotName, rows=None, cols=None, fill=True, showax=False):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    if rows is None:
        rows = len(ims)
        cols = 1

    gridspec_kw = {"wspace": 0, "hspace": 0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, dpi=500, gridspec_kw=gridspec_kw)

    if fill:
        bleed = 0
        fig.subplots_adjust(
            left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
        )

    for ax, im in zip(axarr.ravel(), ims):
        ax.imshow(im)
        if not showax:
            ax.set_axis_off()

    kwargs = {"pad_inches": 0.01} if fill else {}
    fig.savefig(plotName, **kwargs)


def makeGrid():
    btags = ["2b2b", "2b2j"]
    regions = ["CR", "VR", "SR"]
    vbf = ["", "noVBF"]

    for var in collectedKinVars:
        log.info(f"making grid for variable {var}")
        plotsPerGrid = []
        for btag in btags:
            for reg in regions:
                if "massplane" in var:
                    plot = "_".join([var, reg, btag])
                else:
                    plot = "_".join([var, reg, btag, "ratio"])
                plot += ".pdf"
                plotsPerGrid += [plot]
        plotsPerGridWithPath = [plotPath + x for x in plotsPerGrid]
        y = len(regions)
        x = len(btags)

        ims = [
            np.array(convert_from_path(file, 500)[0]) for file in plotsPerGridWithPath
        ]

        savegrid(
            ims,
            f"/lustre/fs22/group/atlas/freder/hh/run/plots/grids/{var}.png",
            rows=x,
            cols=y,
        )

    # for bkg_estimate
    for var in collectedKinVars:
        log.info(f"making grid for variable {var} Background estimate")
        plotsPerGrid = []
        btag = "2b2b"
        for suffix in ["bkgEstimate_ratio", "ratio"]:
            for reg in ["CR", "VR"]:
                if "massplane" not in var:
                    plot = "_".join([var, reg, btag, suffix])
                    plot += ".pdf"
                    plotsPerGrid += [plot]
        plotsPerGridWithPath = [plotPath + x for x in plotsPerGrid]
        y = 2
        x = 2

        ims = [
            np.array(convert_from_path(file, 500)[0]) for file in plotsPerGridWithPath
        ]

        savegrid(
            ims,
            f"/lustre/fs22/group/atlas/freder/hh/run/plots/grids/{var}_bkgEstimate.png",
            rows=x,
            cols=y,
        )


def plotLabel(histKey, ax):
    if "_noVBF" in histKey:
        histKey = histKey[:-6]
    log.info(histKey)
    keyParts = histKey.split("_")
    labels = {}
    if "CR" in histKey:
        labels["region"] = "Control Region"
    elif "VR" in histKey:
        labels["region"] = "Validation Region"
    elif "SR" in histKey:
        labels["region"] = "Signal Region"
    else:
        labels["region"] = ""

    labels["btag"] = keyParts[-1]
    # if len(keyParts) == 3:
    #     labels["vbf"] = ""
    # else:
    labels["vbf"] = "VBF4b"

    # var
    labels["var"] = " ".join(keyParts[:-2])

    # labels["var"] = keyParts[0] + "$_\mathrm{" + ",".join(keyParts[1:]) + "}$"
    # print(keyParts)
    # print(",".join(keyParts[1:-2]))


    labels["plot"] = ("\n").join(
        ["Run 2, " + labels["vbf"], labels["region"] + ", " + labels["btag"]]
    )
    # hep.atlas.label(
    #         # data=False,
    #         lumi="140.0",
    #         loc=1,
    #         ax=ax,
    #         llabel="Internal",
    #     ),
    # print(ax.__dict__)
    # anchored_label = AnchoredText(
    #     s=hep.atlas.label(
    #         # data=False,
    #         lumi="140.0",
    #         loc=1,
    #         ax=ax,
    #         llabel="Internal",
    #     ),
    #     loc="upper left",
    #     frameon=False,
    # )

    # anchored_text = AnchoredText(
    #     s=labels["plot"],
    #     loc="upper left",
    #     frameon=False,
    # )
    # ax.add_artist(anchored_label)

    # ax.add_artist(anchored_text)

    hep.atlas.label(
        # data=False,
        lumi="140.0",
        loc=1,
        ax=ax,
        llabel="Internal",
    )
    ax.text(
        x=0.05,
        y=0.875,
        s=labels["plot"],
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=12,
    )

    return labels


# def draw_text(ax):
#     """
#     Draw two text-boxes, anchored by different corners to the upper-left
#     corner of the figure.
#     """
#     at = AnchoredText(
#         "Figure 1a",
#         loc="upper left",
#         prop=dict(size=8),
#         frameon=True,
#     )
#     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#     ax.add_artist(at)

#     at2 = AnchoredText(
#         "Figure 1(b)",
#         loc="lower left",
#         prop=dict(size=8),
#         frameon=True,
#         bbox_to_anchor=(0.0, 1.0),
#         bbox_transform=ax.transAxes,
#     )
#     at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#     ax.add_artist(at2)


def triggerRef_leadingLargeRpT():
    triggerRef_leadingLargeRpT = file["triggerRef_leadingLargeRpT"]["histogram"][1:-1]
    trigger_leadingLargeRpT = file["trigger_leadingLargeRpT"]["histogram"][1:-1]
    trigger_leadingLargeRpT_err = Plotting.tools.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    plt.savefig(plotPath + "triggerRef_leadingLargeRpT.pdf")
    plt.close()


def triggerRef_leadingLargeRm():
    # normalize + cumulative
    triggerRef_leadingLargeRm = file["triggerRef_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm = file["trigger_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm_err = Plotting.tools.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))
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
    hists_cumulative, hists_cumulative_err = Plotting.tools.CumulativeEfficiencies(
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
    ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))
    plt.savefig(plotPath + "accEff_mhh.pdf")
    plt.close()


def trigger_leadingLargeRpT():
    plt.figure()
    err = Plotting.tools.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="lower right")
    plt.savefig(plotPath + "trigger_leadingLargeRpT.pdf")
    plt.close()


def plotVar(histKey):
    signal = SMsignal[histKey]["h"]
    signal_err = SMsignal[histKey]["err"]
    data = run2[histKey]["h"]
    data_err = run2[histKey]["err"]
    tt = ttbar[histKey]["h"]
    tt_err = ttbar[histKey]["err"]
    edges = run2[histKey]["edges"]
    plt.figure()
    # truth_mhh = file["truth_mhh"]["histogram"][1:-1]
    # trigger_leadingLargeRm_err = Plotting.tools.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    # hep.yscale_legend()
    hep.atlas.label(data=False, lumi="140????", year=None, loc=0)

    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))

    plt.savefig(plotPath + "mhh.pdf")
    plt.close()


def massplane(histKey):
    # SMsignal
    # run2
    # ttbar
    # dijet
    plt.figure()
    xbins = run2[histKey]["xbins"]
    ybins = run2[histKey]["ybins"]
    histValues = run2[histKey]["h"]
    plane = hep.hist2dplot(
        histValues,
        xbins=xbins,
        ybins=ybins,
    )
    plane.pcolormesh.set_cmap("GnBu")

    hep.atlas.set_ylabel("m$_\mathrm{H2}$ [GeV]")
    hep.atlas.set_xlabel("m$_\mathrm{H1}$ [GeV]")
    ax = plt.gca()

    X, Y = np.meshgrid(xbins, ybins)
    CS1 = plt.contour(
        X, Y, Plotting.tools.Xhh(X, Y), [1.6], colors="tab:red", linewidths=1
    )
    fmt = {}
    strs = ["SR"]
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)

    # # to show opening of SR contour
    # for i in np.linspace(0, 5, 10):
    #     CS1 = plt.contour(X, Y, Plotting.tools.Xhh(X, Y), [i], linewidths=1)
    #     fmt = {}
    #     strs = [str(np.round(i, 2))]
    #     for l, s in zip(CS1.levels, strs):
    #         fmt[l] = s
    #     ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)
    # CS1 = plt.contour(
    #     X, Y, Plotting.tools.CR_hh(X, Y), [100e3], colors="tab:blue", linewidths=1
    # )
    fmt = {}
    strs = ["VR"]
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)

    CS1 = plt.contour(X, Y, X + Y, [130e3], colors="tab:blue", linewidths=1)
    fmt = {}
    strs = ["CR"]
    for l, s in zip(CS1.levels, strs):
        fmt[l] = s
    ax.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=12)
    # if blind:
    #     CS1 = plt.contourf(X, Y, Plotting.tools.Xhh(X, Y), [0, 1.6], colors="black")

    plt.tight_layout()
    ax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.get_yaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))
    ax.yaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))
    ax.set_aspect("equal")

    plotLabel(histKey, ax)
    # plt.text(f"VBF 4b, {region}, 2b2j")
    # plt.legend(loc="upper right")
    plt.savefig(plotPath + histKey + ".pdf")
    log.info(plotPath + histKey + ".pdf")
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
    # rax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i", offset=False))
    ax.legend(loc="upper right")
    plt.savefig(plotPath + f"SB_{histKey}_ratio.pdf")
    plt.close()


def kinVar_data_ratio(
    histKey,
    SoverB=False,
    bkgEstimate=False,
    rebinFactor=None,
):
    log.info("Plotting " + histKey)
    signal = SMsignal[histKey]["h"]
    signal_err = SMsignal[histKey]["err"]
    data = run2[histKey]["h"]
    data_err = run2[histKey]["err"]
    tt = ttbar[histKey]["h"]
    tt_err = ttbar[histKey]["err"]
    edges = run2[histKey]["edges"]

    if bkgEstimate:
        lowTagHistkey = histKey[:-1] + "j"
        dataLowTag = run2[lowTagHistkey]["h"]
        dataLowTag_err = run2[lowTagHistkey]["err"]
        ttLowTag = ttbar[lowTagHistkey]["h"]
        ttLowTag_err = ttbar[lowTagHistkey]["err"]
        w_CR = 0.008060635632402544
        err_w_CR = 0.0005150403753024878
        jj = (dataLowTag - ttLowTag) * w_CR
        jj_err = Plotting.tools.ErrorPropagation(
            sigmaA=Plotting.tools.ErrorPropagation(
                sigmaA=dataLowTag_err,
                sigmaB=ttLowTag_err,
                operation="-",
            ),
            sigmaB=np.ones(jj.shape) * err_w_CR,
            operation="*",
            A=dataLowTag - ttLowTag,
            B=np.ones(jj.shape) * w_CR,
        )
    else:
        jj = dijet[histKey]["h"]
        jj_err = dijet[histKey]["err"]

    if rebinFactor:
        signal, edges_, signal_err = Plotting.tools.factorRebin(
            h=signal,
            edges=edges,
            factor=rebinFactor,
            err=signal_err,
        )
        data, edges_, data_err = Plotting.tools.factorRebin(
            h=data,
            edges=edges,
            factor=rebinFactor,
            err=data_err,
        )
        tt, edges_, tt_err = Plotting.tools.factorRebin(
            h=tt,
            edges=edges,
            factor=rebinFactor,
            err=tt_err,
        )
        jj, edges_, jj_err = Plotting.tools.factorRebin(
            h=jj,
            edges=edges,
            factor=rebinFactor,
            err=jj_err,
        )
        edges = edges_

    # prediction
    pred = tt + jj
    pred_err = Plotting.tools.ErrorPropagation(tt_err, jj_err, operation="+")

    ratio = data / pred
    ratio_err = Plotting.tools.ErrorPropagation(
        data_err,
        pred_err,
        "/",
        data,
        pred,
    )

    plt.figure()
    if SoverB:
        fig, (ax, rax, rax2) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(8, 8 * 1.25),
            gridspec_kw={"height_ratios": (3, 1, 1)},
            sharex=True,
        )
    else:
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": (3, 1)},
            sharex=True,
        )
    fig.subplots_adjust(hspace=0.08)

    # stack plot
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
        np.append(pred - pred_err, 0),
        np.append(pred + pred_err, 0),
        color="dimgrey",
        linewidth=0,
        alpha=0.3,
        step="post",
        label="stat. uncertainty",
    )

    # data
    hep.histplot(
        data,
        edges,
        histtype="errorbar",
        yerr=data_err,
        color="Black",
        label="data",
        ax=ax,
    )
    # Signal
    hep.histplot(
        signal * 10000,
        edges,
        histtype="step",
        # yerr=True,
        label="SM Signal x $10^4$",
        ax=ax,
        color="hh:darkyellow",  # "orangered",
        linewidth=1.25,
    )
    ax.legend(loc="upper right")
    ax.autoscale()
    # ax.get_ylim()
    # if ymax:
    #     ax.set_ylim([1e-3, 1e6])
    # else:
    #     ax.set_ylim([1e-3, 1e6])
    ax.set_yscale("log")

    # ratio plot
    hep.histplot(
        ratio,
        edges,
        histtype="errorbar",
        yerr=ratio_err,
        ax=rax,
        color="Black",
    )
    normErrLow = (pred - pred_err) / pred
    normErrHigh = (pred + pred_err) / pred
    # error ratioplot
    rax.fill_between(
        edges,
        Plotting.tools.repeatLastValue(normErrLow),
        Plotting.tools.repeatLastValue(normErrHigh),
        color="dimgrey",
        linewidth=0,
        alpha=0.3,
        step="post",
        # label="stat. uncertainty",
    )
    rax.set_ylim([0.0, 2])
    # draw line at 1.0
    rax.axhline(y=1.0, color="tab:red", linestyle="-")

    plt.tight_layout()

    if SoverB:
        sqrt_pred = np.sqrt(pred)
        sqrt_pred_err = Plotting.tools.ErrorPropagation(
            A=pred, sigmaA=pred_err, operation="^", exp=0.5
        )
        ratio2 = signal / sqrt_pred
        ratio2_err = Plotting.tools.ErrorPropagation(
            signal_err,
            sqrt_pred_err,
            "/",
            signal,
            sqrt_pred,
        )
        hep.histplot(
            ratio2,
            edges,
            histtype="errorbar",
            yerr=ratio2_err,
            ax=rax2,
            color="Black",
        )
        # rax2.fill_between(
        #     edges,
        #     Plotting.tools.repeatLastValue(normErrLow),
        #     Plotting.tools.repeatLastValue(normErrHigh),
        #     color="dimgrey",
        #     linewidth=0,
        #     alpha=0.3,
        #     step="post",
        #     # label="stat. uncertainty",
        # )
        rax2.set_ylabel(
            "$\mathrm{S}/\sqrt{\mathrm{Pred.}}$", horizontalalignment="center"
        )
        # rax2.set_ylabel(
        #     r"$ \frac{\mathrm{S}}{\sqrt{\mathrm{Pred.}}}$", horizontalalignment="center"
        # )
        # rax2.set_yscale("log")
    ax.set_ylabel("Events")
    rax.set_ylabel(
        r"$ \frac{\mathrm{Data}}{\mathrm{Pred.}}$", horizontalalignment="center"
    )

    labels = plotLabel(histKey, ax)

    if "eta" in histKey or "phi" in histKey or "dR" in histKey:
        hep.atlas.set_xlabel(f"{labels['var']}")
    else:
        hep.atlas.set_xlabel(f"{labels['var']} [GeV]")
        ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))

    # hep.mpl_magic(ax=ax)
    newLim = list(ax.get_ylim())
    newLim[1] = newLim[1] * 10
    ax.set_ylim(newLim)
    plt.tight_layout()
    if SoverB:
        rax2.get_xaxis().get_offset_text().set_position((2, 0))
    else:
        rax.get_xaxis().get_offset_text().set_position((2, 0))

    # to show subticks of logplot
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    # rax2.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    # rax2.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    if bkgEstimate:
        plt.savefig(plotPath + f"{histKey}_bkgEstimate_ratio.pdf")
        log.info("saving to : " + plotPath + f"{histKey}_bkgEstimate_ratio.pdf")
    else:
        plt.savefig(plotPath + f"{histKey}_ratio.pdf")
        log.info("saving to : " + plotPath + f"{histKey}_ratio.pdf")

    plt.close(fig)


def compareABCD(histKey, factor=None):
    data = run2[histKey]["h"]
    data_err = run2[histKey]["err"]
    tt = ttbar[histKey]["h"]
    tt_err = ttbar[histKey]["err"]
    edges = run2[histKey]["edges"]

    # VR_2b2j*0.008078516356129706
    lowTagHistkey = histKey[:-1] + "j"
    data_2 = run2[lowTagHistkey]["h"]
    data_err_2 = run2[lowTagHistkey]["err"]
    tt_2 = ttbar[lowTagHistkey]["h"]
    tt_err_2 = ttbar[lowTagHistkey]["err"]
    if factor:
        data, edges_, data_err = Plotting.tools.factorRebin(
            h=data,
            edges=edges,
            factor=factor,
            err=data_err,
        )
        tt, edges_, tt_err = Plotting.tools.factorRebin(
            h=tt,
            edges=edges,
            factor=factor,
            err=tt_err,
        )

        data_2, edges_, data_err_2 = Plotting.tools.factorRebin(
            h=data_2,
            edges=edges,
            factor=factor,
            err=data_err_2,
        )
        tt_2, edges_, tt_err_2 = Plotting.tools.factorRebin(
            h=tt_2,
            edges=edges,
            factor=factor,
            err=tt_err_2,
        )
        edges = edges_
    w_CR = 0.008060635632402544
    err_w_CR = 0.0005150403753024878

    jj = data - tt
    jj_err = Plotting.tools.ErrorPropagation(
        sigmaA=data_err, sigmaB=tt_err, operation="-"
    )

    jj_2 = data_2 - tt_2
    jj_err_2 = Plotting.tools.ErrorPropagation(
        sigmaA=data_err_2, sigmaB=tt_err_2, operation="-"
    )

    plt.figure()
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True,
    )

    hep.histplot(
        jj,
        edges,
        histtype="errorbar",
        yerr=jj_err,
        label="data - tt, 2b2b",
        ax=ax,
    )

    bkgEstimateErr = (
        Plotting.tools.ErrorPropagation(
            sigmaA=jj_err_2,
            sigmaB=np.ones(jj_err_2.shape) * err_w_CR,
            operation="*",
            A=jj,
            B=w_CR,
        ),
    )

    hep.histplot(
        jj_2 * w_CR,
        edges,
        histtype="errorbar",
        yerr=bkgEstimateErr,
        label="w_CR*(data - tt), 2b2j",
        ax=ax,
    )

    hep.histplot(
        jj / (jj_2 * w_CR),
        edges,
        histtype="errorbar",
        yerr=Plotting.tools.ErrorPropagation(
            sigmaA=jj_err,
            sigmaB=bkgEstimateErr,
            operation="/",
            A=bkgEstimateErr,
            B=jj,
        ),
        color="Black",
        ax=rax,
    )

    rax.fill_between(
        edges,
        np.append((jj - jj_err) / jj, 0),
        np.append((jj + jj_err) / jj, 0),
        color="tab:blue",
        linewidth=0,
        alpha=0.3,
        step="post",
        label="2b2b stat. uncertainty",
    )

    # hep.atlas.text(" Simulation", loc=1)
    ax.legend()
    ax.set_ylabel("Events")
    rax.set_ylabel(
        r"$ \frac{\mathrm{2b2b}}{\mathrm{w_{CR}\times 2b2j}}$",
        horizontalalignment="center",
    )
    rax.set_ylim([0, 2])
    rax.axhline(y=1.0, color="tab:red", linestyle="-")
    rax.legend()

    hep.atlas.set_xlabel(f"{histKey} $[GeV]$ ")
    plt.tight_layout()
    rax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(Plotting.tools.OOMFormatter(3, "%1.1i"))

    plotLabel(histKey, ax)
    log.info(plotPath + histKey + "_compareABCD.pdf")

    plt.savefig(plotPath + histKey + "_compareABCD.pdf")
    plt.close()


hists = Plotting.loadHists.run()
SMsignal = hists["SMsignal"]
run2 = hists["run2"]
ttbar = hists["ttbar"]
dijet = hists["dijet"]
# trigger_leadingLargeRpT()
# triggerRef_leadingLargeRpT()
# triggerRef_leadingLargeRm()
# accEff_mhh()


# kinVar_data_ratio("m_h1_VR_2b2b", bkgEstimate=True)
# # kinVar_data_ratio("m_h1_VR_2b2b", bkgEstimate=False)
# massplane("massplane_CR_2b2b")
# kinVar_data_ratio("m_jjVBF_twoLargeR", rebinFactor=6)
# kinVar_data_ratio("m_h1_twoLargeR",rebinFactor=10)
# kinVar_data_ratio("lrj_phi_SR_2b2b", SoverB=True)
# kinVar_data_ratio("m_h1_test_VR_2b2b", SoverB=True)
# kinVar_data_ratio("m_jjVBF_twoLargeR_noVBF", rebinFactor=6)
# kinVar_data_ratio("m_hh_CR_2b2j", rebinFactor=2)
# kinVar_data_ratio("m_hh_VR_2b2b", rebinFactor=8)
# kinVar_data_ratio("m_hh_VR_2b2j", rebinFactor=8)
# kinVar_data_ratio("m_hh_VR_2b2b", rebinFactor=8, bkgEstimate=True)
# kinVar_data_ratio("m_h1_VR_2b2b", rebinFactor=8, bkgEstimate=True)


for var in collectedKinVarsWithRegions:
    if "massplane" in var:
        massplane(var)
    else:
        if "noVBF" not in var:
            kinVar_data_ratio(var, bkgEstimate=False, SoverB=True)
for var in collectedKinVarsWithRegions:
    if "2b2b" in var:
        if "noVBF" not in var:
            if "massplane" not in var:
                kinVar_data_ratio(var, bkgEstimate=True, SoverB=True)
makeGrid()

for var in collectedKinVarsWithRegions:
    if "mh" in var and "2b2b" in var and not "noVBF" in var and not "SR" in var:
        compareABCD(var, factor=6)
