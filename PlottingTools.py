import numpy as np

# https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
# https://lss.fnal.gov/archive/test-tm/2000/fermilab-tm-2286-cd.pdf
# http://phys.kent.edu/~smargeti/STAR/D0/Ullrich-Errors.pdf

# per bin
def EfficiencyErrorBayesian(k, n, bUpper):
    if n == 0:
        if bUpper:
            return 0
        else:
            return 1

    error = np.sqrt(
        ((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - (pow(k + 1, 2) / pow(n + 2, 2))
    )
    ratio = k / n
    if bUpper:
        if (ratio + error) > 1:
            return 1.0
        else:
            return ratio + error
    else:
        if (ratio - error) < 0:
            return 0.0
        else:
            return ratio - error


def getEfficiencyErrors(passed, total):
    """get relative upper and lower error bar positions"""
    upper_err = np.array(
        [
            EfficiencyErrorBayesian(passed, total, bUpper=True)
            for passed, total in zip(passed, total)
        ]
    )
    lower_err = np.array(
        [
            EfficiencyErrorBayesian(passed, total, bUpper=False)
            for passed, total in zip(passed, total)
        ]
    )

    value_position = passed / total
    relative_errors = np.array([value_position - lower_err, upper_err - value_position])
    return relative_errors


import matplotlib.ticker


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def CumulativeEfficiencies(hists, baseline, stopCumulativeFrom):
    # calculate cumulatives and errors for efficiency plots
    ratio = np.array([])
    cumulatives = np.array([])
    baseline_err = np.array([])
    cumulatives_err = np.array([])

    for i in range(len(hists)):
        ratio=np.concatenate((ratio,hists[i] / baseline))
        if i == 0:
            cumulatives=np.concatenate((cumulatives,ratio))
        elif i >= stopCumulativeFrom:
            cumulatives=np.concatenate((cumulatives,cumulatives[stopCumulativeFrom - 1] * ratio[i]))
        else:
            cumulatives=np.concatenate((cumulatives,cumulatives[i - 1] * ratio[i]))
        # error wrt baseline
        baseline_err=np.concatenate((baseline_err,getEfficiencyErrors(passed=hists[i], total=baseline)))

    print(len(baseline_err))
    print(baseline_err[0].shape)
    # error propagation
    for i in range(len(cumulatives)):
        if i == 0:
            cumulatives_err=np.concatenate((cumulatives_err,baseline_err[0]))
        else:
            if i >= stopCumulativeFrom:
                until = stopCumulativeFrom
            else:
                until = i
            err_sum =np.array([])
            for k in range(until):
                err_sum += np.power((baseline_err[k] / ratio[k]), 2)
            propageted_err = ratio[i] * np.sqrt(err_sum)

            # print(len(propageted_err))
            # print(propageted_err[0].shape)
            cumulatives_err=np.concatenate((cumulatives_err,propageted_err))

    #         triggerPass = nTriggerPass_truth_mhh / nTruthEvents
    #         twoLargeR = triggerPass * nTwoLargeR_truth_mhh / nTruthEvents
    #         twoSelLargeR = twoLargeR * nTwoSelLargeR_truth_mhh / nTruthEvents
    #         btagLow_1b1j = twoSelLargeR * btagLow_1b1j / nTruthEvents

    #         # errors
    #         nTriggerPass_err = tools.getEfficiencyErrors(
    #             passed=nTriggerPass_truth_mhh, total=nTruthEvents
    #         )
    #         nTwoLargeR_err = tools.getEfficiencyErrors(
    #             passed=nTwoLargeR_truth_mhh, total=nTruthEvents
    #         )
    #         nTwoSelLargeR_err = tools.getEfficiencyErrors(
    #             passed=nTwoSelLargeR_truth_mhh, total=nTruthEvents
    #         )
    #         btagLow_1b1j_err = tools.getEfficiencyErrors(
    #             passed=btagLow_1b1j, total=nTruthEvents
    #         )
    #         # error propagation
    #         twoLargeR_err = twoLargeR * np.sqrt(
    #             np.power(nTriggerPass_err / triggerPass, 2)
    #             + np.power(nTwoLargeR_err / twoLargeR, 2)
    #         )
    #         twoSelLargeR_err = twoSelLargeR * np.sqrt(
    #             np.power(nTriggerPass_err / triggerPass, 2)
    #             + np.power(nTwoLargeR_err / twoLargeR, 2)
    #             + np.power(nTwoSelLargeR_err / twoSelLargeR, 2)
    #         )
    #         twoSelLargeRhave2b_err = twoSelLargeRhave2b * np.sqrt(
    #             np.power(nTriggerPass_err / triggerPass, 2)
    #             + np.power(nTwoLargeR_err / twoLargeR, 2)
    #             + np.power(nTwoSelLargeR_err / twoSelLargeR, 2)
    #             + np.power(nTwoLargeRHave2BtagVR_err / twoSelLargeRhave2b, 2)
    #         )

    return cumulatives, cumulatives_err
