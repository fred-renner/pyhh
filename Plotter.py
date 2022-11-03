#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.font_manager
from matplotlib import ticker
import mplhep as hep
from h5py import File, Group, Dataset

matplotlib.font_manager._rebuild()
plt.style.use(hep.style.ATLAS)


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

with File("/lustre/fs22/group/atlas/freder/hh/run/histograms/hists.h5", "r") as file:
    names = list(file.keys())
    for name in names:
        plotname = re.search("XHS_(.*)_4b", name).group(1)
        
        hep.histplot(file[name]["histogram"], file[name]["edges"], label=plotname, density=True, alpha=0.75)

hep.atlas.text("Simulation", loc=1)
hep.atlas.set_ylabel("Normalized Events")
hep.atlas.set_xlabel("$m_{hh}$ [GeV]  ")
ax = plt.gca()

ax.get_xaxis().get_offset_text().set_position((1.08, 0))
# ax.get_yaxis().get_offset_text().set_position((2, 0))
plt.tight_layout()

plt.legend(loc="upper right")
plt.savefig("m_hh_plot.pdf")
