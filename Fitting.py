#!/usr/bin/env python3
import json

import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhf.contrib.viz import brazil
import Plotting.loadHists
from Plotting.tools import ErrorPropagation, factorRebin

hists = Plotting.loadHists.run()
fitVariable = "m_hh_lessBins.CR_2b2j"
edges = hists["SMsignal"][fitVariable]["edges"]

# SMsignal,_,SMsignal_err = factorRebin(hists["SMsignal"][fitVariable]["h"],edges,factor=10,err=hists["SMsignal"][fitVariable]["err"])
# run2,_,run2_err = factorRebin(hists["run2"][fitVariable]["h"],edges,factor=10,err=hists["run2"][fitVariable]["err"])
# ttbar,_,ttbar_err = factorRebin(hists["ttbar"][fitVariable]["h"],edges,factor=10,err=hists["ttbar"][fitVariable]["err"])
# dijet,_,dijet_err = factorRebin(hists["dijet"][fitVariable]["h"],edges,factor=10,err=hists["dijet"][fitVariable]["err"])
SMsignal = hists["SMsignal"][fitVariable]["h"]
SMsignal_err = hists["SMsignal"][fitVariable]["err"]
run2 = hists["run2"][fitVariable]["h"]
run2_err = hists["run2"][fitVariable]["err"]
ttbar = hists["ttbar"][fitVariable]["h"]
ttbar_err = hists["ttbar"][fitVariable]["err"]
dijet = hists["dijet"][fitVariable]["h"]
dijet_err = hists["dijet"][fitVariable]["err"]
bkg = ttbar + dijet

bkg_unc = np.array(ErrorPropagation(ttbar_err, dijet_err, operation="+")) * 2

pyhf.set_backend("numpy")
model = pyhf.simplemodels.uncorrelated_background(
    signal=list(SMsignal), bkg=list(bkg), bkg_uncertainty=list(bkg_unc)
)

print(f"  channels: {model.config.channels}")
print(f"     nbins: {model.config.channel_nbins}")
print(f"   samples: {model.config.samples}")
print(f" modifiers: {model.config.modifiers}")
print(f"parameters: {model.config.parameters}")
print(f"  nauxdata: {model.config.nauxdata}")
print(f"   auxdata: {model.config.auxdata}")
model.config.suggested_init()

print(model.config.suggested_bounds())

# model.config.par_map[name]['paramset']
model.config._par_order
model.config.par_map["mu"]["paramset"].suggested_bounds = [(0, 1e5)]
print(model.config.par_map["mu"]["paramset"].suggested_bounds)


model.config.suggested_fixed()

init_pars = model.config.suggested_init()
model.expected_actualdata(init_pars)

model.config.poi_index

bkg_pars = init_pars.copy()
bkg_pars[model.config.poi_index] = 20
model.expected_actualdata(bkg_pars)

observations = list(run2) + model.config.auxdata  # this is a common pattern!

model.logpdf(pars=bkg_pars, data=observations)
model.logpdf(pars=init_pars, data=observations)
CLs_obs, CLs_exp = pyhf.infer.hypotest(
    50000,  # null hypothesis
    list(run2) + model.config.auxdata,
    model,
    test_stat="q",
    return_expected_set=True,
)
print(f"      Observed CLs: {CLs_obs:.10f}")
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
    print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.10f}")

CLs_obs, CLs_exp = pyhf.infer.hypotest(
    10000,  # null hypothesis
    list(run2) + model.config.auxdata,
    model,
    test_stat="qtilde",
    return_expected_set=True,
)
print(f"      Observed CLs: {CLs_obs:.4f}")
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
    print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.4f}")
## Simple Upper Limit
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
# Perform a hypothesis test scan across POIs
results = [
    pyhf.infer.hypotest(
        poi_value,
        observations,
        model,
        test_stat="qtilde",
        return_expected_set=True,
    )
    for poi_value in poi_values
]
