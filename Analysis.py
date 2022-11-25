import numpy as np
import vector

np.set_printoptions(threshold=np.inf)


# # task that operates on an item
# def task(item):
# 	# do one thing using item
# 	# do another thing using item
#
# # slow for loop executed sequentially
# for item in items:
# 	task(item)
# # create a process pool that uses all cpus
# with multiprocessing.Pool() as pool:
# 	# call the function for each item in parallel
# 	pool.map(task, items)


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

    # select large R jets
    lrj_pt = vars_arr["recojet_antikt10_NOSYS_pt"]
    lrj_eta = vars_arr["recojet_antikt10_NOSYS_eta"]
    trigger = vars_arr["trigPassed_HLT_j460_a10sd_cssk_pf_jes_ftf_preselj225_L1SC111_CJ15"]
    # event nr per iteration
    nEvents = len(lrj_pt)

    # init some variables
    # make list holding the large R jet selection indices per event
    sel_lrj = [x for x in range(nEvents)]
    nLargeR = np.zeros(nEvents, dtype=int)
    nTwoLargeRevents = np.zeros(nEvents, dtype=bool)
    truth_m_hh = np.zeros(nEvents)

    for event in range(nEvents):
        # pt, eta cuts
        ptMin = lrj_pt[event] > 250.0
        etaMin = lrj_eta[event] < 2.0
        selected = ptMin & etaMin
        nJets = np.count_nonzero(selected)
        # delete bool arrays if there are less then 2
        if nJets < 2:
            selected = np.zeros(nJets, dtype=bool)
        else:
            nTwoLargeRevents[event] = True
        # count largR and save bool select array
        nLargeR[event] = nJets
        # truth m_hh mass
        sel_lrj[event] = selected
        truth_h1_p4 = vector.obj(
            pt=vars_arr["truth_H1_pt"][event],
            eta=vars_arr["truth_H1_eta"][event],
            phi=vars_arr["truth_H1_phi"][event],
            m=vars_arr["truth_H1_m"][event],
        )
        truth_h2_p4 = vector.obj(
            pt=vars_arr["truth_H2_pt"][event],
            eta=vars_arr["truth_H2_eta"][event],
            phi=vars_arr["truth_H2_phi"][event],
            m=vars_arr["truth_H2_m"][event],
        )
        truth_m_hh[event] = (truth_h1_p4 + truth_h2_p4).mass

    if histkey == "events_truth_mhh":
        # return only the truth m_hh values for events with at least two large R
        valuesToBin = truth_m_hh[:]
        return valuesToBin

    if histkey == "nTriggerPass_truth_mhh":
        # return only the truth m_hh values for events with at least two large R
        valuesToBin = truth_m_hh[trigger]
        print(valuesToBin)
        return valuesToBin

    if histkey == "nTwoSelLargeR_truth_mhh":
        # return only the truth m_hh values for events with at least two large R
        valuesToBin = truth_m_hh[nTwoLargeRevents]
        return valuesToBin

    if histkey == "nTotalSelLargeR":
        # duplicate the truth mhh values with the nr of large R jets for the hist
        nSelLargeR = np.full((nEvents, np.max(nLargeR)), np.inf)
        for event in range(nEvents):
            n = nLargeR[event]
            nSelLargeR[event, :n] = np.full(n, truth_m_hh[event])
        return nSelLargeR.flatten()

    if histkey == "hh_m_85":
        hh_m = vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"]
        hh_m_selected = hh_m[hh_m > 0]
        return hh_m_selected

    if histkey == "massplane_85":
        h1_m = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_m"]
        h1_m_selected = h1_m[h1_m > 0]
        h2_m = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_m"]
        h2_m_selected = h2_m[h2_m > 0]
        return np.array([h1_m_selected, h2_m_selected]).T

    if histkey == "pairingEfficiencyResolved":
        matchCriterion = 0.2
        # fmt: off
        # remove defaults
        nonDefaults = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"] != -1
        h1_sameInitial = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
        h2_sameInitial = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
        h1_dR_lead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
        h1_dR_sublead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
        h2_dR_lead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
        h2_dR_sublead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
        # fmt: on
        # add  s/h
        # this works because of numpy
        matched_h1 = h1_sameInitial & h1_dR_lead & h1_dR_sublead
        matched_h2 = h2_sameInitial & h2_dR_lead & h2_dR_sublead

        # encode h1 match with 1 and h2 match with 2, remove zeros for h2 otherwise double count total dihiggs
        matched = np.concatenate([matched_h1 * 1, (matched_h2 + 2)])

        return matched

    if histkey == "vrJetEfficiencyBoosted":
        matchCriterion = 0.2
        # fmt: off
        # remove defaults        
        nonDefaults = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"] != -1
        h1_sameInitial = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
        h2_sameInitial = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
        h1_dR_lead = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
        h1_dR_sublead = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
        h2_dR_lead = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
        h2_dR_sublead = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
        # fmt: on

        matched_h1 = h1_sameInitial & h1_dR_lead & h1_dR_sublead
        matched_h2 = h2_sameInitial & h2_dR_lead & h2_dR_sublead

        # encode h1 match with 1 and h2 match with 2, remove zeros for h2 otherwise double count total dihiggs
        matched = np.concatenate([matched_h1 * 1, (matched_h2 + 2)])

        return matched
