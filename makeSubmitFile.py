#!/usr/bin/env python3
import subprocess
from tools.HistFillerTools import ConstructFilelist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=str, default=None)
args = parser.parse_args()

if args.sample:
    sample = args.sample
else:
    sample = "mc20_l1cvv1cv1"

filelist = ConstructFilelist(sampleName=sample)

if "mc" in sample:
    import tools.MetaData

    for file in filelist:
        tools.MetaData.get(file)

# copy template header HistFillConfig.txt to submit file
subprocess.call(
    "cp /lustre/fs22/group/atlas/freder/hh/hh-analysis/scripts/HistFillConfig.txt"
    f" /lustre/fs22/group/atlas/freder/hh/submit/HistFill_{sample}.sub",
    shell=True,
)

# write jobs per line
print(f"Made submit file for {sample} with {len(filelist)} jobs. ")
with open(f"/lustre/fs22/group/atlas/freder/hh/submit/HistFill_{sample}.sub", "a") as f:
    for i, file in enumerate(filelist):
        f.write(f"arguments = $(Proxy_path) $(cpus) {file}")
        f.write("\n")
        f.write("queue")
        f.write("\n")
