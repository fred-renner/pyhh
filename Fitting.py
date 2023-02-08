#!/usr/bin/env python3

import pyhf
import numpy as np
import matplotlib.pyplot as plt
from pyhf.contrib.viz import brazil
import Plotting.loadHists

hists_ = Plotting.loadHists.run()
fitVariable = "mhh_VR_2b2j"
SMsignal = hists_["SMsignal"][fitVariable]
run2 = hists_["run2"][fitVariable]
ttbar = hists_["ttbar"][fitVariable]
dijet = hists_["dijet"][fitVariable]


pyhf.set_backend("numpy")
model = pyhf.simplemodels.uncorrelated_background(
    signal=[30.0, 45.0], bkg=[100.0, 150.0], bkg_uncertainty=[151.0, 20.0]
)
data = [100.0, 145.0] + model.config.auxdata

poi_vals = np.linspace(0, 5, 41)
results = [
    pyhf.infer.hypotest(
        test_poi, data, model, test_stat="qtilde", return_expected_set=True
    )
    for test_poi in poi_vals
]

fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
brazil.plot_results(poi_vals, results, ax=ax)
plt.savefig("fittest.pdf")
