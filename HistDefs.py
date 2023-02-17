from tools.Histograms import FloatHistogram, IntHistogram, FloatHistogram2D
import copy
import math

# define hists
accEffBinning = {"binrange": (0, 5_000_000), "bins": 75}
m_hBinning = {"binrange": (0, 300_000), "bins": 100}
pt_hBinning = {"binrange": (0.2e6, 1e6), "bins": 100}
TriggerEffpT = {"binrange": (0, 3_000_000), "bins": 150}
TriggerEffm = {"binrange": (0, 300_000), "bins": 150}
dRbins = {"binrange": (0, 1.2), "bins": 75}
count = {"binrange": (0, 2), "bins": 2}
vrBinning = {"binrange": (0, 0.5e6), "bins": 75}

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
]

# just use kinematicHists as template to construct further down for all regions
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
        name="m_h1_test",
        binrange=m_hBinning["binrange"],
        bins=10,
    ),
    FloatHistogram(
        name="m_h2",
        binrange=m_hBinning["binrange"],
        bins=m_hBinning["bins"],
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
        binrange=(0.4e6, 1.5e6),
        bins=pt_hBinning["bins"],
    ),
    FloatHistogram(
        name="dR_h1",
        binrange=dRbins["binrange"],
        bins=dRbins["bins"],
    ),
    FloatHistogram(
        name="dR_h2",
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
        bins=75,
    ),
    FloatHistogram(
        name="pt_vbf2",
        binrange=(0, 1e6),
        bins=75,
    ),
    FloatHistogram(
        name="pt_h1_btag_vr_1",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h1_btag_vr_2",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h2_btag_vr_1",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
    FloatHistogram(
        name="pt_h2_btag_vr_2",
        binrange=vrBinning["binrange"],
        bins=vrBinning["bins"],
    ),
    FloatHistogram(
        name="m_jjVBF",
        binrange=(0, 3e6),
        bins=75,
    ),
    FloatHistogram(
        name="lrj_pt",
        binrange=(0, 3e6),
        bins=75,
    ),
    FloatHistogram(
        name="lrj_eta",
        binrange=(-5, 5),
        bins=75,
    ),
    FloatHistogram(
        name="lrj_phi",
        binrange=(-2 * math.pi, 2 * math.pi),
        bins=75,
    ),
    FloatHistogram(
        name="lrj_m",
        binrange=(0, 3e6),
        bins=75,
    ),
    FloatHistogram(
        name="srj_pt",
        binrange=(0, 3e6),
        bins=75,
    ),
    FloatHistogram(
        name="srj_eta",
        binrange=(-5, 5),
        bins=75,
    ),
    FloatHistogram(
        name="srj_phi",
        binrange=(-2 * math.pi, 2 * math.pi),
        bins=75,
    ),
    FloatHistogram(
        name="srj_m",
        binrange=(0, 3e6),
        bins=75,
    ),
]

# construct hists for all regions and kinematic vars
regions = [
    "twoLargeR",
    "SR_2b2b",
    "SR_2b2j",
    "VR_2b2b",
    "VR_2b2j",
    "CR_2b2b",
    "CR_2b2j",
]
regions_noVBF = [r + "_noVBF" for r in regions]
regions += regions_noVBF
kinVars = []
kinVarsWithRegions = []
for hist in kinematicHists:
    # without selection
    kinVar = getattr(hist, "_name")
    kinVars += [kinVar]
    for reg in regions:
        var = kinVar + "_" + reg
        kinVarsWithRegions += [var]
        newHist = copy.deepcopy(hist)
        newHist._name = var
        hists.append(newHist)

collectedKinVars = kinVars

collectedKinVarsWithRegions = kinVarsWithRegions
