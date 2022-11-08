import operator
import numpy as np


def do(histkey, vars_arr):
    """
    do analysis on a given histogram type
    Parameters
    ----------
    histkey : str
        histogram type
    vars_arr : np.ndarray
        loaded vars in memory

    Returns
    -------
    np.ndarray
        values to fill hist
    """

    if histkey == "hh_m_85":
        hh_m = vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"]
        hh_m_selected = hh_m[hh_m > 0]
        return hh_m_selected

    if histkey == "correctPariringResolved":
        matchCritierion = 0.2
        # fmt: off
        # remove defaults
        h1_nonDefaults = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"] != -1
        h2_nonDefaults = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"] != -1
        h1_sameInitial = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"][h1_nonDefaults] > 0
        h2_sameInitial = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle"][h2_nonDefaults] > 0
        h1_dR_lead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB"][h1_nonDefaults] < matchCritierion
        h1_dR_sublead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB"][h1_nonDefaults] < matchCritierion
        h2_dR_lead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB"][h2_nonDefaults] < matchCritierion
        h2_dR_sublead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB"][h2_nonDefaults] < matchCritierion
        # fmt: on
        # add  s/h

        # this works because of numpy
        matched_h1 = h1_sameInitial & h1_dR_lead & h1_dR_sublead
        matched_h2 = h2_sameInitial & h2_dR_lead & h2_dR_sublead

        # encode h1 match with 1 and h2 match with 2, remove zeros for h2 otherwise double count total dihiggs
        matched = np.concatenate([matched_h1 * 1, (matched_h2 + 2)])

        return matched
