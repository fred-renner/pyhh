import numpy as np
import vector
import multiprocessing.dummy

np.set_printoptions(threshold=np.inf)


class ObjectSelection:
    # select large R jets
    def __init__(self, vars_arr):
        # fmt: off
        self.lrj_pt = vars_arr["recojet_antikt10_NOSYS_pt"]
        self.lrj_eta = vars_arr["recojet_antikt10_NOSYS_eta"]
        self.trigger = vars_arr["trigPassed_HLT_j460_a10sd_cssk_pf_jes_ftf_preselj225_L1SC111_CJ15"]
        self.vars_arr = vars_arr
        # fmt: on
        # event nr per iteration
        self.nEvents = len(self.lrj_pt)

        # init some variables
        # make list holding the large R jet selection indices per event
        self.sel_lrj = [x for x in range(self.nEvents)]
        self.nLargeR = np.zeros(self.nEvents, dtype=int)
        self.nTwoLargeRevents = np.zeros(self.nEvents, dtype=bool)
        self.truth_m_hh = np.zeros(self.nEvents)

    def select(self, event):
        # pt, eta cuts
        ptMin = self.lrj_pt[event] > 250.0
        etaMin = self.lrj_eta[event] < 2.0
        selected = ptMin & etaMin
        nJets = np.count_nonzero(selected)
        # delete bool arrays if there are less then 2
        if nJets < 2:
            selected = np.zeros(nJets, dtype=bool)
        else:
            self.nTwoLargeRevents[event] = True
        # count largeR and save bool select array
        self.nLargeR[event] = nJets
        # truth m_hh mass
        self.sel_lrj[event] = selected
        truth_h1_p4 = vector.obj(
            pt=self.vars_arr["truth_H1_pt"][event],
            eta=self.vars_arr["truth_H1_eta"][event],
            phi=self.vars_arr["truth_H1_phi"][event],
            m=self.vars_arr["truth_H1_m"][event],
        )
        truth_h2_p4 = vector.obj(
            pt=self.vars_arr["truth_H2_pt"][event],
            eta=self.vars_arr["truth_H2_eta"][event],
            phi=self.vars_arr["truth_H2_phi"][event],
            m=self.vars_arr["truth_H2_m"][event],
        )
        self.truth_m_hh[event] = (truth_h1_p4 + truth_h2_p4).mass


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

    objects = ObjectSelection(vars_arr)

    # parallelize selection

    cpus = multiprocessing.cpu_count()
    # cpus = 1  # for debugging
    # the dummy version is needed to write to our actual object here and not
    # instead of having the child processes make copies
    with multiprocessing.Pool(cpus) as pool:
        pool.map(objects.select, range(objects.nEvents))
    # for event in range(objects.nEvents):
    #     objects.select(event)
    if histkey == "events_truth_mhh":
        # return only the truth m_hh values for events with the selection
        valuesToBin = objects.truth_m_hh[:]
        return valuesToBin

    if histkey == "nTriggerPass_truth_mhh":
        # return only the truth m_hh values for events with the selection
        valuesToBin = objects.truth_m_hh[objects.trigger]
        return valuesToBin

    if histkey == "nTwoSelLargeR_truth_mhh":
        # return only the truth m_hh values for events with the selection
        valuesToBin = objects.truth_m_hh[objects.nTwoLargeRevents]
        return valuesToBin

    if histkey == "nTotalSelLargeR":
        # duplicate the truth mhh values with the nr of large R jets for the hist
        nSelLargeR = np.full((objects.nEvents, np.max(objects.nLargeR)), np.inf)
        for event in range(objects.nEvents):
            n = objects.nLargeR[event]
            nSelLargeR[event, :n] = np.full(n, objects.truth_m_hh[event])
        return nSelLargeR.flatten()

    if histkey == "hh_m_85":
        hh_m = vars_arr["resolved_DL1dv00_FixedCutBEff_85_hh_m"]
        hh_m_selected = hh_m[hh_m > 0]
        return hh_m_selected

    if histkey == "massplane_85":
        h1_m = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_m"]
        h2_m = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_m"]
        selection = (h1_m > 0) & (h2_m > 0)
        h1_m_selected = h1_m[selection]
        h2_m_selected = h2_m[selection]
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
