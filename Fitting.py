#!/usr/bin/env python3

import matplotlib.pyplot as plt
from pyhf.contrib.viz import brazil
import json
import numpy as np
import pyhf
import Plotting.loadHists
from Plotting.tools import ErrorPropagation, factorRebin
import os


def getLimits(signalkey):
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
            factor=5,
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
        signal=list(h[signalkey][fitVariable]["h"]),
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

    observations = (
        list(h["run2"][fitVariable]["h"]) + model.config.auxdata
    )  # this is a common pattern!

    # CLs_obs, CLs_exp = pyhf.infer.hypotest(
    #     100,  # null hypothesis
    #     observations,
    #     model,
    #     test_stat="qtilde",
    #     return_expected_set=True,
    # )
    # print(f"      Observed CLs: {CLs_obs:.4f}")
    # for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
    #     print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.4f}")

    # # Simple Upper Limit
    if signalkey == "SM":
        poi_values = np.linspace(0, 1e5, 50)
    else:
        poi_values = np.linspace(0, 1e3, 50)
    obs_limit, exp_limits, (scan, results) = (
        pyhf.infer.intervals.upper_limits.upper_limit(
            observations, model, poi_values, level=0.05, return_results=True
        )
    )
    print(exp_limits)
    print(f"Upper limit (obs): μ = {obs_limit:.4f}")
    print(f"Upper limit (exp): μ = {exp_limits[2]:.4f}")

    # need to return floats for json
    return float(obs_limit), [float(l) for l in exp_limits]


results = {}
resultsFile = "/lustre/fs22/group/atlas/freder/hh/run/fitResults.json"
if not os.path.exists(resultsFile):
    os.mknod(resultsFile)


results["hypotheses"] = ["k2v0", "SM", "k2v0"]
results["k2v"] = [-1, 0, 1]
results["obs"] = []
results["-2s"] = []
results["-1s"] = []
results["exp"] = []
results["2s"] = []
results["1s"] = []

for hyp in results["hypotheses"]:
    obs_limit, exp_limits = getLimits(hyp)
    results["obs"].append(obs_limit)
    results["-2s"].append(exp_limits[0])
    results["-1s"].append(exp_limits[1])
    results["exp"].append(exp_limits[2])
    results["1s"].append(exp_limits[3])
    results["2s"].append(exp_limits[4])
json.dump(results, open(resultsFile, "w"))
