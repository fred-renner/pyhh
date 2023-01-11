from Histograms import FloatHistogram, IntHistogram, FloatHistogram2D

# define hists
accEffBinning = {"binrange": (0, 3_000_000), "bins": 75}
m_hBinning = {"binrange": (0, 300_000), "bins": 100}
pt_hBinning = {"binrange": (0.2e6, 1e6), "bins": 100}
TriggerEffpT = {"binrange": (0, 3_000_000), "bins": 150}
TriggerEffm = {"binrange": (0, 300_000), "bins": 150}
dRbins = {"binrange": (0, 1.2), "bins": 75}

hists = [
    FloatHistogram(
        name="truth_mhh",
        binrange=accEffBinning["binrange"],
        bins=accEffBinning["bins"],
    ),
    # needs to be the same binning as accEff for plot
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
        name="nTotalSelLargeR",
        binrange=(0, 2_500_000),
        bins=100,
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
    FloatHistogram2D(
        name="massplane_77",
        binrange1=(50_000, 250_000),
        binrange2=(50_000, 250_000),
        bins=100,
    ),
    # "vrJetEfficiencyBoosted": IntHistogram(
    #     name="vrJetEfficiencyBoosted",
    #     binrange=(0, 3),
    # ),
]