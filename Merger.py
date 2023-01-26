#!/usr/bin/env python3
import numpy as np
import h5py
from tqdm.auto import tqdm
from tools.HistFillerTools import ConstructFilelist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=str, default=None)
args = parser.parse_args()

if args.sample:
    sample = args.sample
else:
    sample = "mc20_l1cvv1cv1"

filelist = ConstructFilelist(sample, toMerge=True)
mergedFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-" + sample + ".h5"

# get hist names
hists = []
with h5py.File(filelist[0], "r") as readFile:
    histKeys = list(readFile.keys())

with h5py.File(mergedFile, "w") as mergeFile:
    # copy some file
    with h5py.File(filelist[0], "r") as readFile:
        for group in readFile.keys():
            readFile.copy(group, mergeFile)
    # init datastructure values to 0
    for group in mergeFile.values():
        for ds in group.values():
            ds[:] = np.zeros(ds.shape)

    print("Merge files into: " + mergedFile)
    # loop over files to merge and add values into merged file
    pbar = tqdm(total=len(filelist), position=0, leave=True)
    for ith_file in filelist:
        pbar.update(1)
        with h5py.File(ith_file, "r") as f_i:
            for group in f_i.keys():
                for ds in f_i[group].keys():
                    mergeFile[group][ds][:] += f_i[group][ds][:]
    pbar.close()
