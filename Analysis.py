import numpy as np
import vector
import time
import awkward as ak

np.set_printoptions(threshold=np.inf)


def Run(batch, tree, vars):
    vars_arr = tree.arrays(
        vars, entry_start=batch[0], entry_stop=batch[1], library="np"
    )
    objects = ObjectSelection(vars_arr)
    objects.select()
    return objects.returnResults()


class ObjectSelection:
    # select large R jets
    def __init__(self, vars_arr):
        # fmt: off
        self.vars_arr = vars_arr
        self.lrj_pt = vars_arr["recojet_antikt10_NOSYS_pt"]
        self.lrj_eta = vars_arr["recojet_antikt10_NOSYS_eta"]
        self.lrj_m = vars_arr["recojet_antikt10_NOSYS_m"]
        self.trigger = vars_arr["trigPassed_HLT_j460_a10sd_cssk_pf_jes_ftf_preselj225_L1SC111_CJ15"]
        self.triggerRef = vars_arr["trigPassed_HLT_j420_35smcINF_a10t_lcw_jes_L1SC111_CJ15"]
        # fmt: on
        # event nr per iteration
        self.nEvents = len(self.lrj_pt)
        self.eventRange = range(self.nEvents)

        # init some variables
        # make list holding the large R jet selection indices per event
        self.sel_lrj = [x for x in range(self.nEvents)]
        # int init
        self.nLargeR = np.zeros(self.nEvents, dtype=int)
        intInitArray = np.full(self.nEvents, -1, dtype=int)
        self.nLargeRSelected = intInitArray
        self.leadingLargeRindex = intInitArray
        # bool init
        boolInitArray = np.zeros(self.nEvents, dtype=bool)
        self.nTwoLargeRevents = boolInitArray
        self.AtLeastOneLargeR = boolInitArray
        self.AtLeastOneLargeR = boolInitArray
        self.leadingLargeRmassGreater100 = boolInitArray
        # float init
        floatInitArray = np.full(self.nEvents, -1, dtype=float)
        self.m_hh = floatInitArray
        self.truth_m_hh = floatInitArray
        self.leadingLargeRpt = floatInitArray

    # mcEventWeights
    # pileupWeight_NOSYS
    # generatorWeight_NOSYS

    def select(self):
        for event in self.eventRange:
            self.largeRSelect(event)
            self.truth_mhh(event)
            self.getLeadingLargeR(event)
            self.getLeadingLargeRpts(event)
        self.hh_m_85()
        self.massplane_85()
        self.nTotalSelLargeR()

    def largeRSelect(self, event):
        # pt, eta cuts
        ptMin = self.lrj_pt[event] > 250.0
        etaMin = self.lrj_eta[event] < 2.0
        selected = ptMin & etaMin
        nJetsSelected = np.count_nonzero(selected)
        # delete bool arrays if there are less then 2
        if nJetsSelected < 2:
            selected = np.zeros(nJetsSelected, dtype=bool)
        else:
            self.nTwoLargeRevents[event] = True
        # count largeR and save bool select array
        self.nLargeRSelected[event] = nJetsSelected
        self.sel_lrj[event] = selected

    def getLeadingLargeR(self, event):
        # check which event has at least one Large R
        self.nLargeR[event] = self.lrj_pt[event].shape[0]
        if self.nLargeR[event] > 0:
            self.AtLeastOneLargeR[event] = True
            self.leadingLargeRindex[event] = np.argmax(self.lrj_pt[event])

    def getLeadingLargeRpts(self, event):
        if self.AtLeastOneLargeR[event]:
            if self.lrj_m[event][self.leadingLargeRindex[event]] > 100:
                self.leadingLargeRmassGreater100 = True

    def nTotalSelLargeR(self):
        self.nSelLargeRFlat = self.ReplicateBins(
            binnedObject=self.truth_m_hh, Counts=self.nLargeRSelected
        )

    def ReplicateBins(self, binnedObject, Counts):
        # duplicate the binnedObject bin value with the given Counts per event
        replicatedBins = np.full((self.nEvents, np.max(Counts)), -np.inf)
        for event in self.eventRange:
            n = Counts[event]
            replicatedBins[event, :n] = np.full(n, binnedObject[event])
        return replicatedBins.flatten()

    def truth_mhh(self, event):
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

    def hh_m_85(self):
        hh_m = self.vars_arr["boosted_DL1r_FixedCutBEff_85_hh_m"]
        self.hh_m_selected = hh_m[hh_m > 0]

    def massplane_85(self):
        h1_m = self.vars_arr["boosted_DL1r_FixedCutBEff_85_h1_m"]
        h2_m = self.vars_arr["boosted_DL1r_FixedCutBEff_85_h2_m"]
        selection = (h1_m > 0) & (h2_m > 0)
        h1_m_selected = h1_m[selection]
        h2_m_selected = h2_m[selection]
        self.m_h1h2 = np.array([h1_m_selected, h2_m_selected]).T

    def returnResults(self):
        results = {
            "truth_mhh": self.truth_m_hh[:],
            "nTriggerPass_truth_mhh": self.truth_m_hh[self.trigger[:]],
            "nTwoSelLargeR_truth_mhh": self.truth_m_hh[self.nTwoLargeRevents],
            "nTotalSelLargeR": self.nSelLargeRFlat,
            "hh_m_85": self.hh_m_selected,
            "massplane_85": self.m_h1h2,
            "leadingLargeRpT": self.leadingLargeRpt,
            "trigger_leadingLargeRpT": self.leadingLargeRpt[
                (self.trigger & self.leadingLargeRmassGreater100)
            ],
            "triggerRef_leadingLargeRpT": self.leadingLargeRpt[self.triggerRef],
        }
        return results

    # if histkey == "pairingEfficiencyResolved":
    #     matchCriterion = 0.2
    #     # fmt: off
    #     # remove defaults
    #     nonDefaults = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"] != -1
    #     h1_sameInitial = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
    #     h2_sameInitial = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
    #     h1_dR_lead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     h1_dR_sublead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     h2_dR_lead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     h2_dR_sublead = vars_arr["resolved_DL1dv00_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     # fmt: on
    #     # add  s/h
    #     # this works because of numpy
    #     matched_h1 = h1_sameInitial & h1_dR_lead & h1_dR_sublead
    #     matched_h2 = h2_sameInitial & h2_dR_lead & h2_dR_sublead

    #     # encode h1 match with 1 and h2 match with 2, remove zeros for h2 otherwise double count total dihiggs
    #     matched = np.concatenate([matched_h1 * 1, (matched_h2 + 2)])

    #     return matched

    # if histkey == "vrJetEfficiencyBoosted":
    #     matchCriterion = 0.2
    #     # fmt: off
    #     # remove defaults
    #     nonDefaults = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"] != -1
    #     h1_sameInitial = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
    #     h2_sameInitial = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_closestTruthBsHaveSameInitialParticle"][nonDefaults] > 0
    #     h1_dR_lead = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     h1_dR_sublead = vars_arr["boosted_DL1r_FixedCutBEff_85_h1_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     h2_dR_lead = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_dR_leadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     h2_dR_sublead = vars_arr["boosted_DL1r_FixedCutBEff_85_h2_dR_subleadingJet_closestTruthB"][nonDefaults] < matchCriterion
    #     # fmt: on

    #     matched_h1 = h1_sameInitial & h1_dR_lead & h1_dR_sublead
    #     matched_h2 = h2_sameInitial & h2_dR_lead & h2_dR_sublead

    #     # encode h1 match with 1 and h2 match with 2, remove zeros for h2 otherwise double count total dihiggs
    #     matched = np.concatenate([matched_h1 * 1, (matched_h2 + 2)])

    #     return matched
