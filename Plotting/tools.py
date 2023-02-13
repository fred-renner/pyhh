import numpy as np
import matplotlib.ticker


def EfficiencyErrorBayesian(k, n, bUpper):
    # per bin
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
    """
    get relative upper and lower error bar positions by calculating a bayesian
    Error based on:
    # https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
    # https://lss.fnal.gov/archive/test-tm/2000/fermilab-tm-2286-cd.pdf
    # http://phys.kent.edu/~smargeti/STAR/D0/Ullrich-Errors.pdf

    Parameters
    ----------
    passed : np.ndarray
        values that passed a cut
    total : np.ndarray
        baseline

    Returns
    -------
    np.ndarray
        2xN array holding relative errors to values
    """
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


def ErrorPropagation(
    sigmaA,
    sigmaB,
    operation,
    A=None,
    B=None,
):
    """
    calculate propagated error

    Parameters
    ----------
    sigmaA : ndarray
        standard error of A
    sigmaB : ndarray
        standard error of B
    operation : str
        +, -, *, /
    A : ndarray, optional
        A values, by default None
    B : ndarray, optional
        B values, by default None

    Returns
    -------
    np.ndarray
        propagated error
    """

    if "+" in operation or "-" in operation:
        error = np.sqrt(np.power(sigmaA, 2) + np.power(sigmaB, 2))
    if "*" in operation:
        error = np.abs(A * B) * np.sqrt(
            np.power(np.divide(sigmaA, A, out=np.zeros_like(sigmaA), where=A != 0), 2)
            + np.power(np.divide(sigmaB, B, out=np.zeros_like(sigmaB), where=B != 0), 2)
        )
    if "/" in operation:
        error = np.abs(A / B) * np.sqrt(
            np.power(np.divide(sigmaA, A, out=np.zeros_like(sigmaA), where=A != 0), 2)
            + np.power(np.divide(sigmaB, B, out=np.zeros_like(sigmaB), where=B != 0), 2)
        )

    return error


def factorRebin(
    h,
    edges,
    factor=int(2),
    err=None,
):
    """
    rebin a histogram with equally spaced bins, the last resulting bin entry not
    necessarily has the same bin width as the other ones.

    Parameters
    ----------
    h : ndarray
        histogram counts
    edges : ndarray
         edges to h
    factor : int, optional
        factor by which to reduce the hist bins, by default int(2)
    err : ndarray, optional
        error to h, by default None

    Returns
    -------
    ndarray, ndarray, ndarray
        newH, newEdgesIndices, newErr

    """

    newBinNr = int(h.shape[0] / factor)
    # edges indices of old bins that suit the factor of new bins e.g. for 99
    # bins and a factor of 10 gives
    # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    newEdgesIndices = subintervals(0, h.shape[0], newBinNr)
    # get indices ranges in old bins for new bins, append rest if last index
    # not matching original last index
    # [[0, 9], [10, 19], [20, 29], [30, 39], [40, 49], [50, 59], [60, 69], [70, 79], [80, 89], [90, 99]]
    intervals = []
    for i, j in zip(newEdgesIndices[:-1], newEdgesIndices[1:]):
        intervals += [[i, j - 1]]
    intervals[-1][-1] = h.shape[0]
    if newEdgesIndices[-1] != h.shape[0]:
        newEdgesIndices[-1] = h.shape[0]

    hNew = []
    errNew = []
    for slice in intervals:
        hNew.append(np.sum(h[slice[0] : slice[1]]))
        # error propagate
        if err is not None:
            squaredSigma = 0
            for e in err[slice[0] : slice[1]]:
                squaredSigma += np.power(e, 2)
            errNew.append(np.sqrt(squaredSigma))

    edgesNew = edges[newEdgesIndices]
    return np.array(hNew), np.array(edgesNew), np.array(errNew)


def subintervals(a, b, n):
    # n subintervals in the range [a,b]
    lst = [int(a + x * (b - a) / n) for x in range(n + 1)]
    return lst
