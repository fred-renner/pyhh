#!/usr/bin/env python3
import subprocess
from tools.HistFillerTools import ConstructFilelist


sample = "run2"
filelist = ConstructFilelist(sample)

if "mc" in sample:
    import tools.MetaData

    for file in filelist:
        tools.MetaData.get(file)

# copy template header HistFillConfig.txt to submit file
subprocess.call(
    "cp /lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/HistFillConfig.txt"
    " /lustre/fs22/group/atlas/freder/hh/submit/HistFill.sub",
    shell=True,
)

# write jobs per line
print(f"Made submit file for {sample} with {len(filelist)} jobs. ")
with open("/lustre/fs22/group/atlas/freder/hh/submit/HistFill.sub", "a") as f:
    for i, file in enumerate(filelist):
        f.write(f"arguments = $(Proxy_path) $(cpus) {file}")
        f.write("\n")
        f.write("queue")
        f.write("\n")
