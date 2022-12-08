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
