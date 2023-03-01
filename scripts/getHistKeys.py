#!/usr/bin/env python3
import sys

sys.path.append("/lustre/fs22/group/atlas/freder/hh/pyhh")
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
log.info("format is variable.region_btag")
