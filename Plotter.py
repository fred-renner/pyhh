#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import uproot
import matplotlib.pyplot as plt
import numpy as np
import re
import Loader
import matplotlib.font_manager
from matplotlib import ticker
import mplhep as hep

matplotlib.font_manager._rebuild()

plt.style.use(hep.style.ATLAS)


# files to load
filelist = [
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801578.Py8EG_A14NNPDF23LO_XHS_X300_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801584.Py8EG_A14NNPDF23LO_XHS_X400_S200_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.root",
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
]


# vars to load
finalVars = [
    "resolved_DL1dv00_FixedCutBEff_85_hh_m",
    "resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle",
    "resolved_DL1dv00_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle",
    "resolved_DL1dv00_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB",
    "resolved_DL1dv00_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB",
    "resolved_DL1dv00_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB",
    "resolved_DL1dv00_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB",
]

vars_dict = {}

for file in filelist:
    with uproot.open(file) as file_:

        tree = file_["AnalysisMiniTree"]

        plotname = re.search("XHS_(.*)_4b", file).group(1)

        # add overflow bins
        infvar = np.array([np.inf])
        bins = np.linspace(0, 900_000, 100)
        # create empty hist
        hist = np.zeros(bins.size - 1, dtype=float)
        # make generators to load only a certain amount
        generators = Loader.GetGenerators(tree, finalVars)
        for vars_arr in generators:

            # cutting
            vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"] = vars_arr[
                "resolved_DL1dv00_FixedCutBEff_85_hh_m"
            ][vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"] > 0]

            # fill hist
            histEntryYields = np.histogramdd(
                vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"], bins=[bins]
            )[0]
            hist += histEntryYields
        plotname += " (" + str(int(np.sum(hist))) + " Events)"
        hep.histplot(hist, bins, label=plotname, density=True, alpha=0.75)

hep.atlas.text("Simulation", loc=2)
hep.atlas.set_ylabel("Normalized Events")
hep.atlas.set_xlabel("$m_{hh}$ ($10^5$ GeV) ")
plt.tight_layout()
ax = plt.gca()

# move exponent out of canvas so no overlap with labels, not nice I know but couldn't came up with sth better for now
ax.get_xaxis().get_offset_text().set_position((1.5, 0))
ax.get_yaxis().get_offset_text().set_position((2, 0))

plt.legend(loc="upper right")
plt.savefig("m_hh_plot.pdf")
