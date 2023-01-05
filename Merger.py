#!/usr/bin/env python3
import numpy as np
import h5py
import glob
import subprocess

# mc20 signal
# 1cvv1cv1
topPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/"
pattern = "user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r*/*"
mergedFile = topPath + "hists-MC20-signal.h5"

# mc20 bkg
# # ttbar
# topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
# pattern = "*ttbar*/*"

# # dijet
# topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
# pattern = "*jetjet*/*"

# data 17
# topPath = "/lustre/fs22/group/atlas/freder/hh/run/testfiles/"
# pattern = "data*"

filelist = []
for file in glob.iglob(topPath + "/" + pattern):
    filelist += [file]

# get hist names
hists = []
with h5py.File(filelist[0], "r") as readFile:
    histKeys = list(readFile.keys())

with h5py.File(mergedFile, "w") as mergeFile:

    # copy some file and init datastructure values to 0
    with h5py.File(filelist[0], "r") as readFile:
        for group in readFile.keys():
            readFile.copy(group, mergeFile)
    # init datastructure values to 0
    for group in mergeFile.values():
        for ds in group.values():
            ds[:] = np.zeros(ds.shape)

    # loop over files to merge and add values into merged file
    for ith_file in filelist:
        with h5py.File(ith_file, "r") as f_i:
            for group in f_i.keys():
                for ds in f_i[group].keys():
                    mergeFile[group][ds][:] += f_i[group][ds][:]
