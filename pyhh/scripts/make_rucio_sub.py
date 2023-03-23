#!/usr/bin/env python3

import subprocess


# copy template header HistFillConfig.txt to submit file
subprocess.call(
    (
        (
            "cp /lustre/fs22/group/atlas/freder/hh/pyhh/scripts/rucio_download_config.txt"
            " /lustre/fs22/group/atlas/freder/hh/submit/rucio_download.sub"
        ),
    ),
    shell=True,
)


# fmt: off
files = [
"user.frenner:user.frenner.HH4b.2023_03_13_.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r13144_p5440_TREE ",
"user.frenner:user.frenner.HH4b.2023_03_13_.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r13145_p5440_TREE ",
"user.frenner:user.frenner.HH4b.2023_03_13_.502970.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv1cv1.e8263_s3681_r13167_p5440_TREE ",
"user.frenner:user.frenner.HH4b.2023_03_13_.502971.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv0cv1.e8263_s3681_r13144_p5440_TREE ",
"user.frenner:user.frenner.HH4b.2023_03_13_.502971.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv0cv1.e8263_s3681_r13145_p5440_TREE ",
"user.frenner:user.frenner.HH4b.2023_03_13_.502971.MGPy8EG_hh_bbbb_vbf_novhh_l1cvv0cv1.e8263_s3681_r13167_p5440_TREE ",
]

# write jobs per line

with open("/lustre/fs22/group/atlas/freder/hh/submit/rucio_download.sub", "a") as f:
    for i, file in enumerate(files):
        f.write(f"arguments = $(Proxy_path) {file} /lustre/fs22/group/atlas/freder/hh/samples")
        f.write("\n")
        f.write("queue")
        f.write("\n")
