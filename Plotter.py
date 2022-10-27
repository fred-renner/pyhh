from tqdm import tqdm
import numpy as np
import uproot


file = uproot.open(
    "/lustre/fs22/group/atlas/freder/hh/run/analysis-variables-mc20_13TeV.801578.Py8EG_A14NNPDF23LO_XHS_X300_S70_4b.deriv.DAOD_PHYS.e8448_a899_r13167_p5057.root:AnalysisMiniTree"
)

# get var names
resolvedVars = []
resolvedVars = [s for s in file.keys() if "resolved" in s]

boostedVars = []
boostedVars = [s for s in file.keys() if "boosted" in s]

