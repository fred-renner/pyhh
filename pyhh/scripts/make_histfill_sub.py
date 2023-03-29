#!/usr/bin/env python3
import subprocess

from selector.tools import ConstructFilelist
from tools.logging import log


def run(args):
    if args.sample:
        sample = args.sample
    else:
        sample = "mc20_SM"

    filelist = ConstructFilelist(sampleName=sample)

    if "mc" in sample:
        import selector.metadata

        for file in filelist:
            selector.metadata.get(file)

    # copy template header HistFillConfig.txt to submit file
    subprocess.call(
        (
            "cp /lustre/fs22/group/atlas/freder/hh/pyhh/pyhh/scripts/histfill_config.txt"
            f" /lustre/fs22/group/atlas/freder/hh/submit/histfill_{sample}.sub"
        ),
        shell=True,
    )

    # write jobs per line
    log.info(f"Made submit file for {sample} with {len(filelist)} jobs. ")
    with open(
        f"/lustre/fs22/group/atlas/freder/hh/submit/histfill_{sample}.sub", "a"
    ) as f:
        for i, file in enumerate(filelist):
            f.write(f"arguments = {file}")
            f.write("\n")
            f.write("queue")
            f.write("\n")
