#!/usr/bin/env python3

from histfiller.histdefs import collectedKinVars, collectedKinVarsWithRegions, regions
from tools.logging import log

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
