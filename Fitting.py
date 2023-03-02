#!/usr/bin/env python3

import matplotlib.pyplot as plt
from pyhf.contrib.viz import brazil

import numpy as np
import pyhf
import Plotting.loadHists
from Plotting.tools import ErrorPropagation, factorRebin

h = Plotting.loadHists.run("m_hh_lessBins.CR_2b2j")
fitVariable = "m_hh_lessBins.CR_2b2j"

for datatype in h.keys():
    (
        h[datatype][fitVariable]["h"],
        h[datatype][fitVariable]["edges"],
        h[datatype][fitVariable]["err"],
    ) = factorRebin(
        h[datatype][fitVariable]["h"],
        h[datatype][fitVariable]["edges"],
        factor=2,
        err=h[datatype][fitVariable]["err"],
    )
bkg = h["ttbar"][fitVariable]["h"] + h["dijet"][fitVariable]["h"]

bkg_unc = (
    np.array(
        ErrorPropagation(
            h["ttbar"][fitVariable]["err"],
            h["dijet"][fitVariable]["err"],
            operation="+",
        )
    )
    * 2
)

pyhf.set_backend("numpy")
model = pyhf.simplemodels.uncorrelated_background(
    signal=list(h["SMsignal"][fitVariable]["h"]),
    bkg=list(bkg),
    bkg_uncertainty=list(bkg_unc),
)

print(f"  channels: {model.config.channels}")
print(f"     nbins: {model.config.channel_nbins}")
print(f"   samples: {model.config.samples}")
print(f" modifiers: {model.config.modifiers}")
print(f"parameters: {model.config.parameters}")
print(f"  nauxdata: {model.config.nauxdata}")
print(f"   auxdata: {model.config.auxdata}")

model.config.par_map["mu"]["paramset"].suggested_bounds = [(0, 1e5)]
print(model.config.par_map["mu"]["paramset"].suggested_bounds)

init_pars = model.config.suggested_init()
# model.expected_actualdata(init_pars)
model.config.poi_index
bkg_pars = init_pars.copy()
bkg_pars[model.config.poi_index] = 20
# model.expected_actualdata(bkg_pars)
model.config.suggested_fixed()
observations = list(h["run2"][fitVariable]["h"]) + model.config.auxdata  # this is a common pattern!

CLs_obs, CLs_exp = pyhf.infer.hypotest(
    50000,  # null hypothesis
    observations,
    model,
    test_stat="qtilde",
    return_expected_set=True,
)
print(f"      Observed CLs: {CLs_obs:.4f}")
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
    print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.4f}")


# # Simple Upper Limit
poi_values = np.linspace(0, 1e5, 20)
obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
    observations, model, poi_values, level=0.05, return_results=True
)
print(f"Upper limit (obs): μ = {obs_limit:.4f}")
print(f"Upper limit (exp): μ = {exp_limits[2]:.4f}")
fig, ax = plt.subplots()
fig.set_size_inches(10.5, 7)
ax.set_title("m$_{HH}$ Control Region 2b2j")

artists = brazil.plot_results(poi_values, results, ax=ax)

plt.savefig("test.pdf")
