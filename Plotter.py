#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import ticker
import mplhep as hep
from h5py import File, Group, Dataset

matplotlib.font_manager._rebuild()
plt.style.use(hep.style.ATLAS)


histFile = "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-analysis-variables-mc20_13TeV.801578.Py8EG_A14NNPDF23LO_XHS_X300_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.h5"

with File(histFile, "r") as file:
    for hist in file.keys():
        # access [1:-1] to remove underflow and overflow bins
        if "hh_m" in hist:
            hep.histplot(
                file[hist]["histogram"][1:-1],
                file[hist]["edges"],
                label=hist,
                density=True,
                alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            hep.atlas.set_ylabel("Events")
            hep.atlas.set_xlabel("$m_{hh}$ [GeV]  ")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig("m_hh_plot.pdf")

        if "correctPariring" in hist:
            hep.histplot(
                file[hist]["histogram"][1:-1],
                file[hist]["edges"],
                label=hist,
                density=True,
                alpha=0.75,
            )
            hep.atlas.text(" Simulation", loc=1)
            nEvents = int(sum(file[hist]["histogram"][1:-1]))
            hep.atlas.set_ylabel(f"Events/{nEvents}")
            hep.atlas.set_xlabel("Correct pairing h1(1), h2(2)")
            ax = plt.gca()
            ax.get_xaxis().get_offset_text().set_position((1.09, 0))
            # ax.set_xticks(file[hist]["edges"])
            plt.tight_layout()
            plt.legend(loc="upper right")
            plt.savefig("correctPariring.pdf")
