#!/usr/bin/env python3
import sys
sys.path.append('/lustre/fs22/group/atlas/freder/hh/pyhh')
import subprocess
from tools.HistFillerTools import ConstructFilelist
import argparse
from tools.logging import log


parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=str, default=None)
args = parser.parse_args()

if args.sample:
    sample = args.sample
else:
    sample = "mc20_SM"

filelist = ConstructFilelist(sampleName=sample)

if "mc" in sample:
    import tools.MetaData

    for file in filelist:
        tools.MetaData.get(file)

# copy template header HistFillConfig.txt to submit file
subprocess.call(
    "cp /lustre/fs22/group/atlas/freder/hh/pyhh/scripts/HistFillConfig.txt"
    f" /lustre/fs22/group/atlas/freder/hh/submit/HistFill_{sample}.sub",
    shell=True,
)

# write jobs per line
log.info(f"Made submit file for {sample} with {len(filelist)} jobs. ")
with open(f"/lustre/fs22/group/atlas/freder/hh/submit/HistFill_{sample}.sub", "a") as f:
    for i, file in enumerate(filelist):
        f.write(f"arguments = {file}")
        f.write("\n")
        f.write("queue")
        f.write("\n")
