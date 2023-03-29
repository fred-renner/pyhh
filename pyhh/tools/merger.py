#!/usr/bin/env python3

import h5py
from selector.tools import ConstructFilelist
from tools.logging import log
from tqdm.auto import tqdm
import selector.configuration
import selector.analysis


def run(args):
    if args.hists:
        filelist = ConstructFilelist(args.sample, mergeProcessedHists=True)
        mergedFile = (
            selector.configuration.outputPath
            + "histograms/hists-"
            + args.sample
            + ".h5"
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
                with h5py.File(ith_file, "r") as f_i:
                    for hist in f_i.keys():
                        histVars = list(mergeFile[hist].keys())
                        histVars.remove("edges")
                        for ds in histVars:
                            mergeFile[hist][ds][:] += f_i[hist][ds][:]
                pbar.update(1)
            pbar.close()

    if args.dumped:
        filelist = ConstructFilelist(args.sample, mergeProcessedDumps=True)
        mergedFile = (
            selector.configuration.outputPath + "dump/dump-" + args.sample + ".h5"
        )
        # create empty dump file with vars structure
        selector.configuration.initDumpFile(mergedFile)

        # append vars to final file
        with h5py.File(mergedFile, "r+") as f:
            pbar = tqdm(total=len(filelist), position=0, leave=True)
            # loop over input files
            for ith_file in filelist:
                for varType in ["bools", "floats"]:
                    if varType == "bools":
                        vars = selector.analysis.boolVars
                    if varType == "floats":
                        vars = selector.analysis.floatVars
                    for var in vars:
                        var_out_ds = f[varType][var]
                        if var_out_ds.shape[0] == 0:
                            idx_start = 0
                        else:
                            idx_start = var_out_ds.shape[0]
                        with h5py.File(ith_file, "r") as f_i:
                            appendValues = f_i[varType][var]
                            idx_end = idx_start + appendValues.shape[0]
                            var_out_ds.resize((idx_end,))
                            var_out_ds[idx_start:idx_end] = appendValues
                pbar.update(1)
        pbar.close()
