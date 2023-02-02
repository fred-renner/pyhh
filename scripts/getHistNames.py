#!/usr/bin/env python3
import h5py
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=None)
args = parser.parse_args()

if args.file:
    file = args.file
else:
    file = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-run2.h5"

# for h in kinematicHists:
#     print(h)

# get hist names
with open("histVariables.txt", "w") as f:
    with h5py.File(file, "r") as readFile:
        for h in readFile.keys():
            print(h)
            f.write(h)
            f.write("\n")
