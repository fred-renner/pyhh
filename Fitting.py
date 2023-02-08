#!/usr/bin/env python3

import pyhf
import numpy as np
import matplotlib.pyplot as plt
from pyhf.contrib.viz import brazil
import Plotting.loadHists
from Plotting.tools import ErrorPropagation

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









# print(SMsignal)
# print( run2)
# print( ttbar)

# SMsignal=[1,2,3,4,0,0]
# SMsignal_err=[1,2,3,4,0,0]
# run2=[1,2,3,4,0,0]
# run2_err=[1,2,3,4,0,0]
# ttbar=[1,2,3,4,0,0]
# ttbar_err=[1,2,3,4,0,0]
# dijet=[1,2,3,4,0,0]
# dijet_err=[1,2,3,4,0,0]
# bkg=[1,2,3,4,0,0]
# bkg_unc=[1,2,3,4,0,0]
# bkg=ttbar+dijet
# bkg_unc = ErrorPropagation(ttbar_err, dijet_err, operation="+")

pyhf.set_backend("numpy")
model = pyhf.simplemodels.uncorrelated_background(
    signal=SMsignal, bkg=ttbar, bkg_uncertainty=ttbar_err
)
# model = pyhf.simplemodels.uncorrelated_background(
#     signal=[5.0, 10.0], bkg=[50.0, 60.0], bkg_uncertainty=[5.0, 12.0]
# )

print(model.__dict__)
print(f"  channels: {model.config.channels}")
print(f"     nbins: {model.config.channel_nbins}")
print(f"   samples: {model.config.samples}")
print(f" modifiers: {model.config.modifiers}")
print(f"parameters: {model.config.parameters}")
print(f"  nauxdata: {model.config.nauxdata}")
print(f"   auxdata: {model.config.auxdata}")
print(model.config.param_set("uncorr_bkguncrt").n_parameters)
init_pars = model.config.suggested_init()
print(len(init_pars))
model.expected_actualdata(init_pars)
CLs_obs, CLs_exp = pyhf.infer.hypotest(
    1.0,  # null hypothesis
    [53.0, 65.0] + model.config.auxdata,
    model,
    test_stat="q",
    return_expected_set=True,
)
print(f"      Observed CLs: {CLs_obs:.4f}")
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
    print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.4f}")
breakasd
data = run2 + model.config.auxdata

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
