import numpy as np

# from root
# https://root.cern.ch/doc/master/TEfficiency_8cxx_source.html#l02754

# Double_t TEfficiency::Normal(Double_t total,Double_t passed,Double_t level,Bool_t bUpper)
# {
#    Double_t alpha = (1.0 - level)/2;
#    if (total == 0) return (bUpper) ? 1 : 0;
#    Double_t average = passed / total;
#    Double_t sigma = std::sqrt(average * (1 - average) / total);
#    Double_t delta = ROOT::Math::normal_quantile(1 - alpha,sigma);

#    if(bUpper)
#       return ((average + delta) > 1) ? 1.0 : (average + delta);
#    else
#       return ((average - delta) < 0) ? 0.0 : (average - delta);
# }

# per bin
def EfficiencyErrorNormal(passed, total, bUpper, confidenceLevel=0.5):
    alpha = (1.0 - confidenceLevel) / 2
    if total == 0:
        if bUpper:
            return 0
        else:
            return 1
    ratio = passed / total
    sigma = np.sqrt(ratio * (1 - ratio) / total)
    delta = np.quantile(a=sigma, q=(1 - alpha))

    if bUpper:
        if (ratio + delta) > 1:
            return 1.0
        else:
            return ratio + delta
    else:
        if (ratio - delta) < 0:
            return 0.0
        else:
            return ratio - delta


def getEfficiencyErrors(passed, total):

    upper_err = np.array(
        [
            EfficiencyErrorNormal(passed, total, bUpper=True)
            for passed, total in zip(passed, total)
        ]
    )
    lower_err = np.array(
        [
            EfficiencyErrorNormal(passed, total, bUpper=False)
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
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format