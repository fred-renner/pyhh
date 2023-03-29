#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import plotter.colors
import plotter.loadHists
import plotter.tools
from selector.histdefs import collectedKinVars, collectedKinVarsWithRegions
from matplotlib import ticker as mticker
from matplotlib.offsetbox import AnchoredText
from pdf2image import convert_from_path
from plotter.tools import ErrorPropagation as propagateError
from tools.logging import log

np.seterr(divide="ignore", invalid="ignore")

plt.style.use(hep.style.ATLAS)

plotPath = "/lustre/fs22/group/atlas/freder/hh/run/plots/run2/"

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
    # vbf = ["", "noVBF"]

    for var in collectedKinVars:
        log.info(f"making grid for variable {var}")
        plotsPerGrid = []
        for btag in btags:
            for reg in regions:
                if "massplane" in var:
                    plot = var + "." + reg + "_" + btag
                else:
                    plot = var + "." + reg + "_" + btag + "_ratio"
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
            for reg in ["VR"]:
                if "massplane" not in var:
                    plot = var + "." + reg + "_" + btag + "_" + suffix
                    plot += ".pdf"
                    plotsPerGrid += [plot]
        plotsPerGridWithPath = [plotPath + x for x in plotsPerGrid]
        y = 2
        x = 1

        ims = [
            np.array(convert_from_path(file, 500)[0]) for file in plotsPerGridWithPath
        ]

        savegrid(
            ims,
            f"/lustre/fs22/group/atlas/freder/hh/run/plots/grids/{var}_bkgEstimate.png",
            rows=x,
            cols=y,
        )


def plotLabel(histkey, ax):
    if "_noVBF" in histkey:
        histkey = histkey[:-6]
    keyParts = histkey.split(".")
    var = keyParts[0]
    sel = keyParts[1]
    selParts = sel.split("_")
    varParts = var.split("_")
    varParts = [s for s in varParts if "lessBins" not in s]
    if "pt" in varParts[0]:
        varParts.pop(0)
        varParts.insert(0, "T")
        varParts.insert(0, "p")
    labels = {}
    if "CR" in histkey:
        labels["region"] = "Control Region"
    elif "VR" in histkey:
        labels["region"] = "Validation Region"
    elif "SR" in histkey:
        labels["region"] = "Signal Region"
    else:
        labels["region"] = ""

    if len(selParts) > 1:
        labels["btag"] = selParts[-1]
    else:
        labels["btag"] = ""

    labels["vbf"] = "VBF4b"

    # var
    labels["var"] = varParts[0] + "$_{\mathrm{" + ",".join(varParts[1:]) + "}}$"

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
    trigger_leadingLargeRpT_err = plotter.tools.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="upper right")
    plt.savefig(plotPath + "triggerRef_leadingLargeRpT.pdf")
    plt.close()


def triggerRef_leadingLargeRm():
    # normalize + cumulative
    triggerRef_leadingLargeRm = file["triggerRef_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm = file["trigger_leadingLargeRm"]["histogram"][1:-1]
    trigger_leadingLargeRm_err = plotter.tools.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))
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
    hists_cumulative, hists_cumulative_err = plotter.tools.CumulativeEfficiencies(
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
    ax.xaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))
    plt.savefig(plotPath + "accEff_mhh.pdf")
    plt.close()


def trigger_leadingLargeRpT():
    plt.figure()
    err = plotter.tools.getEfficiencyErrors(
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
    ax.xaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))
    plt.legend(loc="lower right")
    plt.savefig(plotPath + "trigger_leadingLargeRpT.pdf")
    plt.close()


def massplane(hists, histkey):
    plt.figure()
    xbins = hists["run2"][histkey]["xbins"]
    ybins = hists["run2"][histkey]["ybins"]
    histValues = hists["run2"][histkey]["h"]
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
        X, Y, plotter.tools.Xhh(X, Y), [1.6], colors="tab:red", linewidths=1
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
    CS1 = plt.contour(
        X, Y, plotter.tools.CR_hh(X, Y), [100e3], colors="tab:blue", linewidths=1
    )
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
    ax.xaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))
    ax.yaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))
    ax.set_aspect("equal")

    plotLabel(histkey, ax)
    # plt.text(f"VBF 4b, {region}, 2b2j")
    # plt.legend(loc="upper right")
    plt.savefig(plotPath + histkey + ".pdf")
    log.info(plotPath + histkey + ".pdf")
    plt.close()


def kinVar_data_ratio(
    hists,
    histkey,
    SoverB=False,
    bkgEstimate=False,
    rebinFactor=None,
):
    log.info("Plotting " + histkey)
    s = hists["SM"][histkey]["h"]
    s_err = hists["SM"][histkey]["err"]
    data = hists["run2"][histkey]["h"]
    data_err = hists["run2"][histkey]["err"]
    tt = hists["ttbar"][histkey]["h"]
    tt_err = hists["ttbar"][histkey]["err"]
    edges = hists["run2"][histkey]["edges"]

    if bkgEstimate:
        lowTaghistkey = histkey[:-1] + "j"
        dataLowTag = hists["run2"][lowTaghistkey]["h"]
        dataLowTag_err = hists["run2"][lowTaghistkey]["err"]
        ttLowTag = hists["ttbar"][lowTaghistkey]["h"]
        ttLowTag_err = hists["ttbar"][lowTaghistkey]["err"]
        w_CR = 0.0081093038933622
        err_w_CR = 0.0005316268235806789
        jj = (dataLowTag - ttLowTag) * w_CR
        jj_err = plotter.tools.ErrorPropagation(
            sigmaA=plotter.tools.ErrorPropagation(
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
        jj = hists["dijet"][histkey]["h"]
        jj_err = hists["dijet"][histkey]["err"]

    if rebinFactor:
        s, edges_, s_err = plotter.tools.factorRebin(
            h=s,
            edges=edges,
            factor=rebinFactor,
            err=s_err,
        )
        data, edges_, data_err = plotter.tools.factorRebin(
            h=data,
            edges=edges,
            factor=rebinFactor,
            err=data_err,
        )
        tt, edges_, tt_err = plotter.tools.factorRebin(
            h=tt,
            edges=edges,
            factor=rebinFactor,
            err=tt_err,
        )
        jj, edges_, jj_err = plotter.tools.factorRebin(
            h=jj,
            edges=edges,
            factor=rebinFactor,
            err=jj_err,
        )
        edges = edges_

    # prediction
    pred = tt + jj
    pred_err = plotter.tools.ErrorPropagation(tt_err, jj_err, operation="+")

    ratio = data / pred
    ratio_err = plotter.tools.ErrorPropagation(
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
        s * 10000,
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
        plotter.tools.fillStatHoles(normErrLow),
        plotter.tools.fillStatHoles(normErrHigh),
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
        sqrt_pred_err = plotter.tools.ErrorPropagation(
            A=pred, sigmaA=pred_err, operation="^", exp=0.5
        )
        ratio2 = s / sqrt_pred
        ratio2_err = plotter.tools.ErrorPropagation(
            s_err,
            sqrt_pred_err,
            "/",
            s,
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
        rax2.set_ylabel(
            "$\mathrm{S}/\sqrt{\mathrm{Pred.}}$", horizontalalignment="center"
        )
        # rax2.set_ylabel(
        #     r"$ \frac{\mathrm{S}}{\sqrt{\mathrm{Pred.}}}$", horizontalalignment="center"
        # )
    ax.set_ylabel("Events")
    rax.set_ylabel(
        r"$ \frac{\mathrm{Data}}{\mathrm{Pred.}}$", horizontalalignment="center"
    )

    labels = plotLabel(histkey, ax)

    if "eta" in histkey or "phi" in histkey or "dR" in histkey:
        hep.atlas.set_xlabel(f"{labels['var']}")
    else:
        hep.atlas.set_xlabel(f"{labels['var']} [GeV]")
        ax.xaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))

    # hep.mpl_magic()
    newLim = list(ax.get_ylim())
    newLim[1] = newLim[1] * 100
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
        plt.savefig(plotPath + f"{histkey}_bkgEstimate_ratio.pdf")
        log.info("saving to : " + plotPath + f"{histkey}_bkgEstimate_ratio.pdf")
    else:
        plt.savefig(plotPath + f"{histkey}_ratio.pdf")
        log.info("saving to : " + plotPath + f"{histkey}_ratio.pdf")

    plt.close(fig)


def compareABCD(hists, histkey, factor=None):
    data = hists["run2"][histkey]["h"]
    data_err = hists["run2"][histkey]["err"]
    tt = hists["ttbar"][histkey]["h"]
    tt_err = hists["ttbar"][histkey]["err"]
    edges = hists["run2"][histkey]["edges"]

    lowTaghistkey = histkey[:-1] + "j"
    data_2 = hists["run2"][lowTaghistkey]["h"]
    data_err_2 = hists["run2"][lowTaghistkey]["err"]
    tt_2 = hists["ttbar"][lowTaghistkey]["h"]
    tt_err_2 = hists["ttbar"][lowTaghistkey]["err"]
    if factor:
        data, edges_, data_err = plotter.tools.factorRebin(
            h=data,
            edges=edges,
            factor=factor,
            err=data_err,
        )
        tt, edges_, tt_err = plotter.tools.factorRebin(
            h=tt,
            edges=edges,
            factor=factor,
            err=tt_err,
        )

        data_2, edges_, data_err_2 = plotter.tools.factorRebin(
            h=data_2,
            edges=edges,
            factor=factor,
            err=data_err_2,
        )
        tt_2, edges_, tt_err_2 = plotter.tools.factorRebin(
            h=tt_2,
            edges=edges,
            factor=factor,
            err=tt_err_2,
        )
        edges = edges_
    w_CR = 0.007512067296747797
    err_w_CR = 0.00042915519038828545

    jj = data - tt
    jj_err = plotter.tools.ErrorPropagation(
        sigmaA=data_err, sigmaB=tt_err, operation="-"
    )

    jj_2 = data_2 - tt_2
    jj_err_2 = plotter.tools.ErrorPropagation(
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

    bkgEstimate = jj_2 * w_CR
    bkgEstimateErr = (
        plotter.tools.ErrorPropagation(
            sigmaA=jj_err_2,
            sigmaB=np.ones(jj_err_2.shape) * err_w_CR,
            operation="*",
            A=jj,
            B=np.repeat(w_CR, jj.shape[0]),
        ),
    )

    hep.histplot(
        bkgEstimate,
        edges,
        histtype="errorbar",
        yerr=bkgEstimateErr,
        label="w_CR*(data - tt), 2b2j",
        ax=ax,
    )

    hep.histplot(
        jj / bkgEstimate,
        edges,
        histtype="errorbar",
        yerr=plotter.tools.ErrorPropagation(
            sigmaA=jj_err,
            sigmaB=bkgEstimateErr,
            operation="/",
            A=jj,
            B=bkgEstimate,
        ),
        color="Black",
        ax=rax,
    )

    normErrLow = (jj - jj_err) / jj
    normErrHigh = (jj + jj_err) / jj
    rax.fill_between(
        edges,
        plotter.tools.fillStatHoles(normErrLow),
        plotter.tools.fillStatHoles(normErrHigh),
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

    labels = plotLabel(histkey, ax)
    hep.atlas.set_xlabel(f"{labels['var']} [GeV]")
    plt.tight_layout()
    rax.get_xaxis().get_offset_text().set_position((2, 0))
    ax.xaxis.set_major_formatter(plotter.tools.OOMFormatter(3, "%1.1i"))

    plotLabel(histkey, ax)
    log.info(plotPath + histkey + "_compareABCD.pdf")

    plt.savefig(plotPath + histkey + "_compareABCD.pdf")
    plt.close()


def limits():
    fitResults = json.load(
        open("/lustre/fs22/group/atlas/freder/hh/run/fitResults.json")
    )
    fig, ax = plt.subplots()
    ax.plot(
        fitResults["k2v"],
        fitResults["obs"],
        color="black",
    )
    ax.plot(
        fitResults["k2v"],
        fitResults["exp"],
        color="black",
        linestyle="dashed",
    )
    ax.fill_between(
        fitResults["k2v"],
        fitResults["-2s"],
        fitResults["2s"],
        color="hh:darkyellow",
        linewidth=0,
    )
    ax.fill_between(
        fitResults["k2v"],
        fitResults["-1s"],
        fitResults["1s"],
        color="hh:medturquoise",
        linewidth=0,
    )

    ax.set_ylabel(r"95% CL upper limit on $\sigma_{VBF,HH}$ (fb)")
    ax.set_xlabel(r"$\kappa_{\mathrm{2v}}$")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.legend(
        [
            "Observed",
            "Expected",
            "Expected Limit $\pm 1\sigma$",
            "Expected Limit $\pm 2\sigma$",
        ]
    )
    hep.atlas.label(
        # data=False,
        lumi="140.0",
        loc=1,
        ax=ax,
        llabel="Internal",
    )

    plt.tight_layout()
    log.info(plotPath + "limit.pdf")
    plt.savefig(plotPath + "limit.pdf")
    plt.close()


def run():
    hists = plotter.loadHists.run()

    # trigger_leadingLargeRpT()
    # triggerRef_leadingLargeRpT()
    # triggerRef_leadingLargeRm()
    # accEff_mhh()

    # massplane("massplane_CR_2b2b")
    kinVar_data_ratio(hists, histkey="m_hh_lessBins.VR_2b2b", bkgEstimate=True, SoverB=True)

    # limits()
    # for var in collectedKinVarsWithRegions:
    #     if "massplane" in var:
    #         massplane(hists,var)
    #     else:
    #         kinVar_data_ratio(hists, var, bkgEstimate=False, SoverB=True)

    # for var in collectedKinVarsWithRegions:
    #     if "2b2b" in var and not "massplane" in var:
    #         kinVar_data_ratio(hists, var, bkgEstimate=True, SoverB=True)

    # makeGrid()

    # for var in collectedKinVarsWithRegions:
    #     if "m_h" in var and "2b2b" in var and "VR" in var:
    #         compareABCD(hists, var)
    # compareABCD(hists, "m_hh_lessBins.VR_2b2b")