#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import uproot
import matplotlib.pyplot as plt
import numpy as np
import re

filelist = [
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801578.Py8EG_A14NNPDF23LO_XHS_X300_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801584.Py8EG_A14NNPDF23LO_XHS_X400_S200_4b.deriv.DAOD_PHYS.e8448_s3681_r13167_p5057.root",
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root",
]
plt.figure(0)

for file in filelist:
    with uproot.open(file) as file_:
        hh_m = {}
        tree = file_["AnalysisMiniTree"]

        # get var names
        resolvedVars = []
        resolvedVars = [s for s in tree.keys() if "resolved" in s]
        boostedVars = []
        boostedVars = [s for s in tree.keys() if "boosted" in s]
        finalVars = resolvedVars + boostedVars

        # construct ranges, gives e.g. for batch_size=1000
        # [[0, 999], [1000, 1999], [2000, 2999],...]
        ranges = []
        batch_ranges = []
        batch_size = 1_000

        for i in range(0, tree.num_entries, batch_size):
            ranges += [i]
        if tree.num_entries not in ranges:
            ranges += [tree.num_entries + 1]
        for i, j in zip(ranges[:-1], ranges[1:]):
            batch_ranges += [[i, j - 1]]

        def batches(batch_ranges):
            for batch in batch_ranges:
                arr = tree.arrays(
                    finalVars, entry_start=batch[0], entry_stop=batch[1], library="np"
                )
                yield arr

        generators = batches(batch_ranges=batch_ranges)
        for vars_arr in generators:
            # do something with vars_arr
            hh_m[file] = vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"]
            # print(vars_arr["boosted_DL1r_FixedCutBEff_77_h2_dR_jets"])
        plotName = re.search('XHS_(.*)_4b', file).group(1)
        plt.hist(
            hh_m[file][hh_m[file] > 0],
            bins=150,
            label=plotName,
            density=True,
            histtype="step",
        )  # density=False would make counts
        plt.ylabel("Normalized Events")
        plt.xlabel("m_hh")
# print(hh_m.keys())
# print(filelist[0])
# plt.hist(
#     [hh_m[filelist[0]], hh_m[filelist[1]], hh_m[filelist[2]]],
#     bins=100,
#     label=["1", "2", "3"],
# )
plt.xlim([0, 900_000])
plt.grid(True)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc="upper right")
plt.savefig("m_hh_plot.pdf")
