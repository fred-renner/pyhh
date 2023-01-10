#!/usr/bin/env python3
import subprocess
import glob
import MetaData
import time
# mc21 signal
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/"
# pattern = "user.frenner.HH4b.2022_11_25_.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5440_TREE/*"
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_25_.601480.PhPy8EG_HH4b_cHHH10d0.e8472_s3873_r13829_p5440_TREE"
# topPath = "/lustre/fs22/group/atlas/freder/hh/samples/user.frenner.HH4b.2022_11_30.801172.Py8EG_A14NNPDF23LO_jj_JZ7.e8453_s3873_r13829_p5278_TREE"

# mc20 signal
# 1cvv1cv1
topPath = "/lustre/fs22/group/atlas/freder/hh/samples/"
pattern = "user.frenner.HH4b.2022_12_14.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r*/*"

# mc20 bkg
# ttbar
topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
pattern = "*ttbar*/*"

# # dijet
topPath = "/lustre/fs22/group/atlas/dbattulga/ntup_SH_Oct20/bkg/"
pattern = "*jetjet*/*"

# data 17
# topPath = "/lustre/fs22/group/atlas/freder/hh/run/testfiles/"
# pattern = "data*"

# get all files also from subdirectories with wildcard
filelist = []
for file in glob.iglob(topPath + "/" + pattern):
    filelist += [file]
    MetaData.get(file)

# copy template header HistFillConfig.txt to submit file
subprocess.call(
    "cp /lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/HistFillConfig.txt"
    " /lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/HistFill.sub",
    shell=True,
)

# filelist=filelist[:50]

# write jobs per line
with open(
    "/lustre/fs22/group/atlas/freder/hh/hh-analysis/tools/HistFill.sub", "a"
) as f:
    k = 0
    waitTime = 2
    for i, file in enumerate(filelist):
        k += 1
        delay = k * waitTime
        f.write(f"arguments = $(Proxy_path) $(cpus) {file}")
        f.write("\n")
        # f.write(f"deferral_time = (CurrentTime + {delay})")
        # f.write("\n")
        # f.write(f"deferral_prep_time = (CurrentTime + {delay-20})")
        # f.write("\n")
        # f.write(f"deferral_window = 100000000")
        f.write("\n")
        f.write("queue")
        f.write("\n")
