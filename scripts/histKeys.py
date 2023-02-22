#!/usr/bin/env python3
import sys

sys.path.append("/lustre/fs22/group/atlas/freder/hh/hh-analysis")
from tools.logging import log
from HistDefs import collectedKinVars, collectedKinVarsWithRegions, regions

print("\n")
log.info("*** Regions ***\n")
for r in regions:
    log.info(r)
print("\n")
log.info("*** Variables ***")
for var in collectedKinVars:
    log.info(var)
print("\n")
log.info("format is variable_region")

# import h5py
# file = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-mc20_l1cvv1cv1.h5"

# with h5py.File(file, "r") as f:
#     log.info(f.keys())
