from tools.Histograms import FloatHistogram, IntHistogram, FloatHistogram2D
import copy

# define hists
accEffBinning = {"binrange": (0, 3_000_000), "bins": 75}
m_hBinning = {"binrange": (0, 300_000), "bins": 100}
pt_hBinning = {"binrange": (0.2e6, 1e6), "bins": 100}
TriggerEffpT = {"binrange": (0, 3_000_000), "bins": 150}
TriggerEffm = {"binrange": (0, 300_000), "bins": 150}
dRbins = {"binrange": (0, 1.2), "bins": 75}
count = {"binrange": (0, 2), "bins": 2}


hists = [
    FloatHistogram(
        name="truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="N_CR_4b",
        binrange=count["binrange"],
        bins=count["bins"],
    ),
    FloatHistogram(
        name="N_CR_2b",
        binrange=count["binrange"],
        bins=count["bins"],
    ),
    FloatHistogram(
        name="N_VR_4b",
        binrange=count["binrange"],
        bins=count["bins"],
    ),
    FloatHistogram(
        name="N_VR_2b",
        binrange=count["binrange"],
        bins=count["bins"],
    ),
    FloatHistogram(
        name="N_SR_2b",
        binrange=count["binrange"],
        bins=count["bins"],
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
        name="btagHigh_2b2b_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
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
]

# just use kinematicHists as template to construct further down for all regions
kinematicHists = [
    # needs to be the same binning as accEff plot
    FloatHistogram(
        name="mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    FloatHistogram(
        name="mh1",
        binrange=m_hBinning["binrange"],
        bins=m_hBinning["bins"],
    ),
    FloatHistogram(
        name="mh2",
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
    FloatHistogram2D(
        name="massplane",
        binrange1=(50_000, 250_000),
        binrange2=(50_000, 250_000),
        bins=100,
    ),
]

# construct hists for all regions and kinematic vars
regions = [
    "twoLargeR",
    "SR_4b",
    "SR_2b",
    "VR_4b",
    "VR_2b",
    "CR_4b",
    "CR_2b",
    "SR_4b_noVBF",
    "SR_2b_noVBF",
    "VR_4b_noVBF",
    "VR_2b_noVBF",
    "CR_4b_noVBF",
    "CR_2b_noVBF",
]

for hist in kinematicHists:
    # without selection
    kinVar = getattr(hist, "_name")
    for reg in regions:
        var = kinVar + "_" + reg
        newHist = copy.deepcopy(hist)
        newHist._name = var
        hists.append(newHist)
