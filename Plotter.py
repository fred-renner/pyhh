#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import uproot


file = uproot.open(
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801578.Py8EG_A14NNPDF23LO_XHS_X300_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root"
)

tree = file["AnalysisMiniTree"]
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
    print(vars_arr)
