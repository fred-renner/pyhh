import copy
import math

from tools.Histograms import FloatHistogram, FloatHistogram2D, IntHistogram

# define hists
accEffBinning = {"binrange": (0, 5_000_000), "bins": 50}
m_hBinning = {"binrange": (0, 300_000), "bins": 50}
pt_hBinning = {"binrange": (0.2e6, 1e6), "bins": 50}
TriggerEffpT = {"binrange": (0, 3_000_000), "bins": 150}
TriggerEffm = {"binrange": (0, 300_000), "bins": 150}
dRbins = {"binrange": (0, 1.2), "bins": 50}
count = {"binrange": (0, 2), "bins": 2}
vrBinning = {"binrange": (0, 0.5e6), "bins": 50}

hists = [
    FloatHistogram(
        name="truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTriggerPass_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTwoLargeR_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="nTwoSelLargeR_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="leadingLargeRpT",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="leadingLargeRpT_trigger",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="triggerRef_leadingLargeRpT",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="trigger_leadingLargeRpT",
        binrange=TriggerEffpT["binrange"],
        bins=TriggerEffpT["bins"],
    ),
    FloatHistogram(
        name="triggerRef_leadingLargeRm",
        binrange=TriggerEffm["binrange"],
        bins=TriggerEffm["bins"],
    ),
    FloatHistogram(
        name="trigger_leadingLargeRm",
        binrange=TriggerEffm["binrange"],
        bins=TriggerEffm["bins"],
    ),
    FloatHistogram(
        name="btagLow_1b1j_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagLow_2b1j_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagLow_2b2j_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagHigh_1b1b_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="btagHigh_2b1b_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="pt_lrj",
        binrange=(0, 1.5e6),
        bins=50,
    ),
    FloatHistogram(
        name="eta_lrj",
        binrange=(-2, 2),
        bins=50,
    ),
    FloatHistogram(
        name="phi_lrj",
        binrange=(-math.pi, math.pi),
        bins=50,
    ),
    FloatHistogram(
        name="m_lrj",
        binrange=(0, 0.5e6),
        bins=50,
    ),
    FloatHistogram(
        name="pt_srj",
        binrange=(0, 1.5e6),
        bins=50,
    ),
    FloatHistogram(
        name="eta_srj",
        binrange=(-5, 5),
        bins=50,
    ),
    FloatHistogram(
        name="phi_srj",
        binrange=(-math.pi, math.pi),
        bins=50,
    ),
    FloatHistogram(
        name="m_srj",
        binrange=(0, 200e3),
        bins=50,
    ),
]

# use kinematicHists as template to construct further down for all regions
kinematicHists = [
    # needs to be the same binning as accEff plot
    FloatHistogram(
        name="m_hh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="m_h1",
        binrange=m_hBinning["binrange"],
        bins=m_hBinning["bins"],
    ),
    FloatHistogram(
        name="m_h2",
        binrange=m_hBinning["binrange"],
        bins=m_hBinning["bins"],
    ),
    FloatHistogram(
        name="m_hh_paper",
        binrange=accEffBinning["binrange"],
        bins=25,
    ),
    FloatHistogram(
        name="m_h1_paper",
        binrange=m_hBinning["binrange"],
        bins=25,
    ),
    FloatHistogram(
        name="m_h2_paper",
        binrange=m_hBinning["binrange"],
        bins=25,
    ),
    FloatHistogram(
        name="m_hh_lessBins",
        binrange=accEffBinning["binrange"],
        bins=15,
    ),
    FloatHistogram(
        name="m_h1_lessBins",
        binrange=m_hBinning["binrange"],
        bins=15,
    ),
    FloatHistogram(
        name="m_h2_lessBins",
        binrange=m_hBinning["binrange"],
        bins=15,
    ),
    FloatHistogram(
        name="pt_h1",
        binrange=pt_hBinning["binrange"],
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h2",
        binrange=pt_hBinning["binrange"],
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="pt_hh",
        binrange=(0, 1e6),
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="pt_hh_scalar",
        binrange=(0.65e6, 1.5e6),
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="dR_VR_h1",
        binrange=dRbins["binrange"],
        bins=dRbins["bins"],
    ),
    FloatHistogram(
        name="dR_VR_h2",
        binrange=dRbins["binrange"],
        bins=dRbins["bins"],
    ),
    FloatHistogram2D(
        name="massplane",
        binrange1=(50_000, 250_000),
        binrange2=(50_000, 250_000),
        bins=100,
    ),
    FloatHistogram(
        name="pt_vbf1",
        binrange=(0, 1e6),
        bins=50,
    ),
    FloatHistogram(
        name="pt_vbf2",
        binrange=(0, 1e6),
        bins=50,
    ),
    FloatHistogram(
        name="m_jjVBF",
        binrange=(0, 3e6),
        bins=50,
    ),
    FloatHistogram(
        name="pt_h1_btag_vr1",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h1_btag_vr2",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h2_btag_vr1",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h2_btag_vr2",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
]

# construct hists for all regions and kinematic vars
regions = [
    "trigger",
    "twoLargeR",
    "SR_2b2b",
    "SR_2b2j",
    "VR_2b2b",
    "VR_2b2j",
    "CR_1b1b",
    "CR_2b1b",
    "CR_2b2b",
    "CR_2b2j",
]

kinVars = []
kinVarsWithRegions = []
for hist in kinematicHists:
    # without selection
    kinVar = getattr(hist, "_name")
    kinVars += [kinVar]
    for reg in regions:
        var = kinVar + "." + reg
        kinVarsWithRegions += [var]
        newHist = copy.deepcopy(hist)
        newHist._name = var
        hists.append(newHist)

collectedKinVars = kinVars

collectedKinVarsWithRegions = kinVarsWithRegions
