import numpy as np
import matplotlib.ticker

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

    firstTerm = ((k + 1) * (k + 2)) / ((n + 2) * (n + 3))
    secondTerm = ((k + 1) ** 2) / ((n + 2) ** 2)
    error = np.sqrt(firstTerm - secondTerm)
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
    ratio = []
    cumulatives = []
    baseline_err = []
    cumulatives_err = []

    for i in range(len(hists)):
        ratio.append(hists[i] / baseline)
        print(ratio)
        if i == 0:
            cumulatives.append(ratio[0])
        elif i >= stopCumulativeFrom:
            cumulatives.append(cumulatives[stopCumulativeFrom - 2] * ratio[i])
        else:
            cumulatives.append(cumulatives[i - 1] * ratio[i])
        # error wrt baseline
        baseline_err.append(getEfficiencyErrors(passed=hists[i], total=baseline))

    # error propagation
    for i in range(len(hists)):
        err_sum = 0
        if i == 0:
            cumulatives_err.append(baseline_err[0])
            continue
        elif i >= stopCumulativeFrom:
            for k in range(stopCumulativeFrom - 1):
                err_sum += pow((baseline_err[k] / ratio[k]), 2)
            err_sum += pow((baseline_err[i] / ratio[i]), 2)
        else:
            for k in range(i):
                err_sum += pow((baseline_err[k] / ratio[k]), 2)
        propagated_err = np.array(cumulatives[i]) * np.sqrt(err_sum)
        cumulatives_err.append(propagated_err)
    #         triggerPass = nTriggerPass_truth_mhh / nTruthEvents
    #         twoLargeR = triggerPass * nTwoLargeR_truth_mhh / nTruthEvents

    #         # errors
    #         nTriggerPass_err = tools.getEfficiencyErrors(
    #             passed=nTriggerPass_truth_mhh, total=nTruthEvents
    #         )
    #         nTwoLargeR_err = tools.getEfficiencyErrors(
    #             passed=nTwoLargeR_truth_mhh, total=nTruthEvents
    #         )
    #         # error propagation
    #         twoLargeR_err = twoLargeR * np.sqrt(
    #             np.power(nTriggerPass_err / triggerPass, 2)
    #             + np.power(nTwoLargeR_err / twoLargeR, 2)
    #         )
    return cumulatives, cumulatives_err


m_h1_center = 124.0e3
m_h2_center = 117.0e3
# fm_h1 from signal region optimization:
# https://indico.cern.ch/event/1191598/contributions/5009137/attachments/2494578/4284249/HH4b20220818.pdf
fm_h1 = 1500.0e6
fm_h2 = 1900.0e6
# SR variable (1.6 is the nominal cutoff)
def Xhh(m_h1, m_h2):
    return np.sqrt(
        np.power((m_h1 - m_h1_center) / (fm_h1 / m_h1), 2)
        + np.power((m_h2 - m_h2_center) / (fm_h2 / m_h2), 2)
    )


# https://indico.cern.ch/event/1239101/contributions/5216057/attachments/2575156/4440353/hh4b_230112.pdf
def CR_hh(m_h1, m_h2):
    # need to make in GeV to work with the log function
    return (
        np.sqrt(
            np.power((m_h1 - m_h1_center) * 1e-3 / (0.1 * np.log(m_h1 * 1e-3)), 2)
            + np.power((m_h2 - m_h2_center) * 1e-3 / (0.1 * np.log(m_h2 * 1e-3)), 2)
        )
        * 1e3
    )


def ErrorPropagation(sigmaA, sigmaB, operation, A=None, B=None):
    """_summary_

    Parameters
    ----------
    sigmaA : ndarray
        standard error of A
    sigmaB : ndarray
        _description_
    operation : str
        operator
    A : ndarray, optional
        A values, by default None
    B : ndarray, optional
        B values, by default None

    Returns
    -------
    np.ndarray
        propagated error
    """

    if "+" or "-" in operation:
        error = np.sqrt(np.power(sigmaA, 2) + np.power(sigmaB, 2))
    if "*" in operation:
        error = np.abs(A * B) * np.sqrt(
            np.power(np.divide(sigmaA, A), 2) + np.power(np.divide(sigmaB, B), 2)
        )
    if "/" in operation:
        error = np.abs(A / B) * np.sqrt(
            np.power(np.divide(sigmaA, A), 2) + np.power(np.divide(sigmaB, B), 2)
        )

    return error


def rebin(
    h,
    edges,
    err=None,
    bins=10,
):
    """rebin a histogram

    Parameters
    ----------
    h : ndarray
        histogram counts
    edges : ndarray
         edges to h
    err : ndarray, optional
        error to h, by default None
    bins : int, optional
        nr of new bins, by default 10

    Returns
    -------
    ndarray, ndarray, ndarray
        newH, newEdges, newErr

    Raises
    ------
    ValueError
        if more bins requested than originally given
    """

    # a rebinning with more bins is not really useful
    if bins > (edges.shape[0] - 1):
        raise ValueError("More bins than before")
    # make new edges for bin nr and given bin range
    newEdges = np.linspace(edges[0], edges[-1], bins + 1)
    # get the binindices in which the given hist values end up with the new binning
    # -1 to start counting from zero
    histIndicesForNewEdges = np.digitize(edges, newEdges)[:-1] - 1
    newH = np.zeros(bins)
    newErr = np.zeros(bins)
    for i in range(0, bins):
        # get all the values from h that belong in the i-th bin of the new hist
        # and calculate the mean value and error
        newH[i] = h[histIndicesForNewEdges == i].mean()
        if err is not None:
            newErr[i] = err[histIndicesForNewEdges == i].mean()

    return newH, newEdges, newErr
