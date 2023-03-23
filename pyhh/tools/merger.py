#!/usr/bin/env python3

import h5py
from histfiller.tools import ConstructFilelist
from tools.logging import log
from tqdm.auto import tqdm


def run(args):
    filelist = ConstructFilelist(args.sample, toMerge=True)
    mergedFile = (
        "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-" + args.sample + ".h5"
    )

    with h5py.File(mergedFile, "w") as mergeFile:
        # copy some file
        with h5py.File(filelist[0], "r") as readFile:
            for hist in readFile.keys():
                readFile.copy(hist, mergeFile)
            # init datastructure values to 0
            for hist in mergeFile.keys():
                histVars = list(mergeFile[hist].keys())
                histVars.remove("edges")
                for ds in histVars:
                    mergeFile[hist][ds][:] = 0

        log.info("Merge files into: " + mergedFile)
        # loop over files to merge and add values into merged file
        pbar = tqdm(total=len(filelist), position=0, leave=True)
        for ith_file in filelist:
            pbar.update(1)
            with h5py.File(ith_file, "r") as f_i:
                for hist in f_i.keys():
                    histVars = list(mergeFile[hist].keys())
                    histVars.remove("edges")
                    for ds in histVars:
                        mergeFile[hist][ds][:] += f_i[hist][ds][:]
        pbar.close()
