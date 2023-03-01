#!/usr/bin/env python3

import json

import matplotlib.pyplot as plt
import numpy as np
import Plotting.loadHists
import pyhf
from Plotting.tools import ErrorPropagation
from pyhf.contrib.viz import brazil

hists = Plotting.loadHists.run()
fitVariable = "mhh_VR_2b2j"
SMsignal = hists["SMsignal"][fitVariable]["h"]
SMsignal_err = hists["SMsignal"][fitVariable]["err"]
run2 = hists["run2"][fitVariable]["h"]
run2_err = hists["run2"][fitVariable]["err"]
ttbar = hists["ttbar"][fitVariable]["h"]
ttbar_err = hists["ttbar"][fitVariable]["err"]
dijet = hists["dijet"][fitVariable]["h"]
dijet_err = hists["dijet"][fitVariable]["err"]
bkg = ttbar + dijet
bkg_unc = np.array(ErrorPropagation(ttbar_err, dijet_err, operation="+"))

pyhf.set_backend("numpy")
model = pyhf.simplemodels.uncorrelated_background(
    signal=list(SMsignal), bkg=list(bkg), bkg_uncertainty=list(bkg_unc)
)

print(json.dumps(model.spec, indent=2))
print(f"  channels: {model.config.channels}")
print(f"     nbins: {model.config.channel_nbins}")
print(f"   samples: {model.config.samples}")
print(f" modifiers: {model.config.modifiers}")
print(f"parameters: {model.config.parameters}")
print(f"  nauxdata: {model.config.nauxdata}")
print(f"   auxdata: {model.config.auxdata}")


# print(model.config.param_set("uncorr_bkguncrt").n_parameters)

# print(model.expected_data([0.5, 1.0, 1.0], include_auxdata=False))
print(model.config.par_order)
print(model.config.param_set("uncorr_bkguncrt").n_parameters)
print(model.config.param_set("mu").n_parameters)
print(model.config.suggested_init())
print(model.config.suggested_bounds())
print(model.config.suggested_fixed())
init_pars = model.config.suggested_init()
print(init_pars)
print(model.expected_actualdata(init_pars))

bkg_pars = init_pars.copy()
bkg_pars[model.config.poi_index] = 10
print(model.expected_actualdata(bkg_pars))

observations = list(run2) + model.config.auxdata  # this is a common pattern!
print(model.logpdf(pars=bkg_pars, data=observations))
print(pyhf.infer.mle.fit(data=observations, pdf=model))

CLs_obs, CLs_exp = pyhf.infer.hypotest(
    1.0,  # null hypothesis
    observations,
    model,
    test_stat="q",
    return_expected_set=True,
)
print(f"      Observed CLs: {CLs_obs:.4f}")
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
    print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.4f}")


poi_values = np.linspace(0.1, 5, 50)
obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
    observations, model, poi_values, level=0.05, return_results=True
)
print(f"Upper limit (obs): μ = {obs_limit:.4f}")
print(f"Upper limit (exp): μ = {exp_limits[2]:.4f}")
plt.figure()
fig, ax = plt.subplots()
fig.set_size_inches(10.5, 7)
ax.set_title("Hypothesis Tests")

artists = brazil.plot_results(poi_values, results, ax=ax)
plt.savefig("fittest.pdf")
