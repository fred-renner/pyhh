import numpy as np
import vector
import time
import awkward as ak
import uproot
from operator import xor

np.set_printoptions(threshold=np.inf)


def Run(batch, tree, vars):
    vars_arr = tree.arrays(
        vars, entry_start=batch[0], entry_stop=batch[1], library="np"
    )
    objects = ObjectSelection(vars_arr)
    objects.select()
    return objects.returnResults()


class ObjectSelection:
    def __init__(self, vars_arr):

        # mcEventWeights
        # pileupWeight_NOSYS
        # generatorWeight_NOSYS
        if any("truth" in x for x in vars_arr):
            self.hasTruth = True
        else:
            self.hasTruth = False
        # fmt: off
        self.vars_arr = vars_arr
        # mc21
        if "trigPassed_HLT_j460_a10t_lcw_jes_L1J100" in vars_arr:
            self.trigger = vars_arr["trigPassed_HLT_j460_a10t_lcw_jes_L1J100"]
            self.triggerRef = vars_arr["trigPassed_HLT_j420_35smcINF_a10t_lcw_jes_L1J100"]
        # mc20 
        # r13144
        elif "trigPassed_HLT_j420_a10t_lcw_jes_40smcINF_L1J100" in vars_arr: 
            self.trigger = vars_arr["trigPassed_HLT_j420_a10t_lcw_jes_40smcINF_L1J100"]
            self.triggerRef = vars_arr["trigPassed_HLT_j390_a10t_lcw_jes_30smcINF_L1J100"]
        # r13145
        elif "trigPassed_HLT_j420_a10t_lcw_jes_35smcINF_L1J100" in vars_arr:
            self.trigger = vars_arr["trigPassed_HLT_j420_a10t_lcw_jes_35smcINF_L1J100"]
            self.triggerRef = vars_arr["trigPassed_HLT_j390_a10t_lcw_jes_30smcINF_L1J100"]
        # r13167
        elif "trigPassed_HLT_j420_a10_lcw_L1J100" in vars_arr:
            self.trigger = vars_arr["trigPassed_HLT_j420_a10_lcw_L1J100"]
            self.triggerRef = vars_arr["trigPassed_HLT_j360_a10_lcw_sub_L1J100"]
        
        
        self.lrj_pt = vars_arr["recojet_antikt10_NOSYS_pt"]
        self.lrj_eta = vars_arr["recojet_antikt10_NOSYS_eta"]
        self.lrj_phi = vars_arr["recojet_antikt10_NOSYS_phi"]
        self.lrj_m = vars_arr["recojet_antikt10_NOSYS_m"]
        self.vr_btag_77 = vars_arr["recojet_antikt10_NOSYS_leadingVRTrackJetsBtag_DL1r_FixedCutBEff_77"]
        # self.vr_dontOverlap = vars_arr["passRelativeDeltaRToVRJetCut"]

        # fmt: on
        # event amount per iteration
        self.nEvents = len(self.lrj_pt)
        self.eventRange = range(self.nEvents)

        # init some variables
        # make list holding the large R jet selection indices per event
        self.selPtSort_lrj = [x for x in self.eventRange]
        # int init
        intInitArray = np.full(self.nEvents, -1, dtype=int)
        self.nLargeR = np.copy(intInitArray)
        self.nLargeRSelected = np.copy(intInitArray)
        self.leadingLargeRindex = np.copy(intInitArray)

        # bool init
        boolInitArray = np.zeros(self.nEvents, dtype=bool)
        self.atLeastOneLargeR = np.copy(boolInitArray)
        self.selectedTwoLargeRevents = np.copy(boolInitArray)
        self.btagLow_1b1j = np.copy(boolInitArray)
        self.btagLow_2b1j = np.copy(boolInitArray)
        self.btagLow_2b2j = np.copy(boolInitArray)
        self.btagHigh_1b1b = np.copy(boolInitArray)
        self.btagHigh_2b1b = np.copy(boolInitArray)
        self.btagHigh_2b2b = np.copy(boolInitArray)

        self.leadingLargeRmassGreater100 = np.copy(boolInitArray)
        self.leadingLargeRpTGreater500 = np.copy(boolInitArray)

        # float init
        floatInitArray = np.full(self.nEvents, -1.0, dtype=float)
        self.m_hh = np.copy(floatInitArray)
        self.h1_m = np.copy(floatInitArray)
        self.h2_m = np.copy(floatInitArray)
        self.truth_m_hh = np.copy(floatInitArray)
        self.leadingLargeRpt = np.copy(floatInitArray)
        self.leadingLargeRm = np.copy(floatInitArray)

    def select(self):
        for event in self.eventRange:
            # order matters!
            # if not self.vr_dontOverlap[event]:
            #     # should we actually throw away the whole event?
            #     continue
            self.largeRSelectSort(event)
            self.getLeadingLargeR(event)
            self.getLeadingLargeRcuts(event)
            self.getVRtags(event)
            self.hh_m_77(event)
            if self.hasTruth:
                self.truth_mhh(event)

        self.nTotalSelLargeR()

    def largeRSelectSort(self, event):
        # pt, eta cuts and sort
        ptMin = self.lrj_pt[event] > 250_000.0
        etaMin = np.abs(self.lrj_eta[event]) < 2.0
        selected = np.array((ptMin & etaMin), dtype=bool)
        # counting
        nJetsSelected = np.count_nonzero(selected)
        self.nLargeR[event] = self.lrj_pt[event].shape[0]
        self.nLargeRSelected[event] = nJetsSelected
        # empty array if there are less then 2
        if nJetsSelected < 2:
            self.selPtSort_lrj[event] = np.array([])
        else:
            self.selectedTwoLargeRevents[event] = True
            # selected is a bool array for lrj_pt[event]
            # now sort by getting jet indices in decreasing order of pt
            self.selPtSort_lrj[event] = np.flip(
                np.argsort(self.lrj_pt[event][selected])
            )[0:2]

    def getLeadingLargeR(self, event):
        # check which event has at least one Large R
        if self.nLargeR[event] > 0:
            self.atLeastOneLargeR[event] = True
            maxPtIndex = np.argmax(self.lrj_pt[event])
            self.leadingLargeRindex[event] = maxPtIndex
            self.leadingLargeRpt[event] = self.lrj_pt[event][maxPtIndex]
            maxMIndex = np.argmax(self.lrj_m[event])
            self.leadingLargeRm[event] = self.lrj_m[event][maxMIndex]

    def getLeadingLargeRcuts(self, event):
        if self.atLeastOneLargeR[event]:
            if self.leadingLargeRm[event] > 100_000.0:
                self.leadingLargeRmassGreater100[event] = True
            if self.leadingLargeRpt[event] > 500_000.0:
                self.leadingLargeRpTGreater500[event] = True

    def nTotalSelLargeR(self):
        self.nSelLargeRFlat = self.ReplicateBins(
            binnedObject=self.truth_m_hh, Counts=self.nLargeRSelected
        )

    def getVRtags(self, event):
        if self.selectedTwoLargeRevents[event]:
            # print(self.vr_btag_77[event]._values)
            # print(self.vr_btag_77[event]._values[0])
            # get leading jets
            j1_index = self.selPtSort_lrj[event][0]
            j2_index = self.selPtSort_lrj[event][1]
            # get their corresponding vr jets
            j1_VRs = self.vr_btag_77[event]._values[j1_index]
            j2_VRs = self.vr_btag_77[event]._values[j2_index]
            # count their tags
            j1_VRs_Btag = np.count_nonzero(j1_VRs)
            j2_VRs_Btag = np.count_nonzero(j2_VRs)

            j1_VRs_noBtag = len(j1_VRs) - j1_VRs_Btag
            j2_VRs_noBtag = len(j2_VRs) - j2_VRs_Btag

            # this is not mutually exclusive....
            if xor(
                j1_VRs_Btag == 1 and j2_VRs_noBtag >= 1,
                j2_VRs_Btag == 1 and j1_VRs_noBtag >= 1,
            ):
                self.btagLow_1b1j[event] = True
            if xor(
                j1_VRs_Btag >= 2 and j2_VRs_noBtag >= 1,
                j2_VRs_Btag >= 2 and j1_VRs_noBtag >= 1,
            ):
                self.btagLow_2b1j[event] = True
            if xor(
                j1_VRs_Btag >= 2 and j2_VRs_noBtag >= 2,
                j2_VRs_Btag >= 2 and j1_VRs_noBtag >= 2,
            ):
                self.btagLow_2b2j[event] = True
            if j1_VRs_Btag == 1 and j2_VRs_Btag == 1:
                self.btagHigh_1b1b[event] = True
            if xor(
                j1_VRs_Btag >= 2 and j2_VRs_Btag == 1,
                j2_VRs_Btag >= 2 and j1_VRs_Btag == 1,
            ):
                self.btagHigh_2b1b[event] = True
            if j1_VRs_Btag >= 2 and j2_VRs_Btag >= 2:
                self.btagHigh_2b2b[event] = True
            # print("self.btagLow_1b1j ", self.btagLow_1b1j[event])
            # print("self.btagLow_2b1j ", self.btagLow_2b1j[event])
            # print("self.btagLow_2b2j ", self.btagLow_2b2j[event])
            # print("self.btagHigh_1b1b ", self.btagHigh_1b1b[event])
            # print("self.btagHigh_2b1b ", self.btagHigh_2b1b[event])
            # print("self.btagHigh_2b2b ", self.btagHigh_2b2b[event])

    # //.Define("passBJetSkimBoosted","passTwoFatJets ? ntagsBoosted[0] + ntagsBoosted[1] > 0 : false")
    # //.Define("passBJetSkim_min2bsBoosted","passTwoFatJets ? ntagsBoosted[0] >= 1 && ntagsBoosted[1] >= 1 : false")
    # //.Define("pass4TagBoosted","passTwoFatJets ? ntagsBoosted[0] + ntagsBoosted[1] == 4 : false")
    # //.Define("pass3TagBoosted","passTwoFatJets ? ntagsBoosted[0] + ntagsBoosted[1] == 3 : false")
    # //.Define("pass2TagSplitBoosted","passTwoFatJets ? ntagsBoosted[0] == 1 && ntagsBoosted[1] == 1 : false")
    # //.Define("pass2TagBoosted","passTwoFatJets ? (ntagsBoosted[0] == 2 && ntagsBoosted[1] == 0)||(ntagsBoosted[0] == 0 && ntagsBoosted[1] == 2) : false")
    # //.Define("pass1TagBoosted","passTwoFatJets ? ntagsBoosted[0] + ntagsBoosted[1] == 1 : false")
    # //.Define("pass4Trk","passTwoFatJets ? trks[0] + trks[1] == 4 : false")
    # //.Define("pass3Trk","passTwoFatJets ? trks[0] + trks[1] == 3 : false")
    # //.Define("pass2Trk","passTwoFatJets ? trks[0] >= 1 && trks[1] >= 1 : false")
    # // Use eventNumber to randomize events for background sharing.
    # // To reserve the last bit for similar randomization in reweighting procedure,
    # // do this randomization on (eventNumber>>1).
    # //.Define("eventRand","(eventNumber)%5")
    # // tag region definitions
    # //.Define("is_4bBoosted","pass4TagBoosted")
    # //.Define("is_3bBoosted","pass3TagBoosted")
    # //.Define("is_2bsBoosted","pass2TagSplitBoosted")
    # //.Define("is_4b_bkgmodelBoosted","pass2TagBoosted && pass4Trk && eventRand < "+std::to_string(bkgShareBoosted))
    # //.Define("is_3b_bkgmodelBoosted","pass2TagBoosted && (pass3Trk || (pass4Trk && eventRand >= "+std::to_string(bkgShareBoosted)+"))")
    # //.Define("is_2bs_bkgmodelBoosted","pass1TagBoosted && pass2Trk")

    def ReplicateBins(self, binnedObject, Counts):
        # duplicate the binnedObject bin value with the given Counts per event
        replicatedBins = np.full((self.nEvents, np.max(Counts)), -np.inf)
        for event in self.eventRange:
            n = Counts[event]
            # account for skipped events/defaults
            if n == -1:
                continue
            else:
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

    def hh_m_77(self, event):
        # hh_m = self.vars_arr["boosted_DL1r_FixedCutBEff_77_hh_m"]
        # self.hh_m_selected = hh_m[hh_m > 0]
        if self.selectedTwoLargeRevents[event]:
            leadingLargeRindex = self.selPtSort_lrj[event][0]
            subleadingLargeRindex = self.selPtSort_lrj[event][1]
            h1_p4 = vector.obj(
                pt=self.lrj_pt[event][leadingLargeRindex],
                eta=self.lrj_eta[event][leadingLargeRindex],
                phi=self.lrj_phi[event][leadingLargeRindex],
                m=self.lrj_m[event][leadingLargeRindex],
            )
            h2_p4 = vector.obj(
                pt=self.lrj_pt[event][subleadingLargeRindex],
                eta=self.lrj_eta[event][subleadingLargeRindex],
                phi=self.lrj_phi[event][subleadingLargeRindex],
                m=self.lrj_m[event][subleadingLargeRindex],
            )
            self.h1_m[event] = self.lrj_m[event][leadingLargeRindex]
            self.h2_m[event] = self.lrj_m[event][subleadingLargeRindex]
            self.m_hh[event] = (h1_p4 + h2_p4).mass

    # def massplane_77(self):
    #     h1_m = self.vars_arr["boosted_DL1r_FixedCutBEff_77_h1_m"]
    #     h2_m = self.vars_arr["boosted_DL1r_FixedCutBEff_77_h2_m"]
    #     selection = (h1_m > 0) & (h2_m > 0)
    #     h1_m_selected = h1_m[selection]
    #     h2_m_selected = h2_m[selection]
    #     self.m_h1h2 = np.array([h1_m_selected, h2_m_selected]).T

    def returnResults(self):

        results = {
            "truth_mhh": self.truth_m_hh,
            "mhh": self.m_hh,
            "mh2": self.h1_m[self.btagHigh_2b2b],
            "mh1": self.h2_m[self.btagHigh_2b2b],
            "nTriggerPass_mhh": self.m_hh[self.trigger],
            "nTwoLargeR_mhh": self.m_hh[(self.nLargeR >= 2)],
            "nTwoSelLargeR_mhh": self.m_hh[self.selectedTwoLargeRevents],
            "btagLow_1b1j_mhh": self.m_hh[self.btagLow_1b1j],
            "btagLow_2b1j_mhh": self.m_hh[self.btagLow_2b1j],
            "btagLow_2b2j_mhh": self.m_hh[self.btagLow_2b2j],
            "btagHigh_1b1b_mhh": self.m_hh[self.btagHigh_1b1b],
            "btagHigh_2b1b_mhh": self.m_hh[self.btagHigh_2b1b],
            "btagHigh_2b2b_mhh": self.m_hh[self.btagHigh_2b2b],
            "nTotalSelLargeR": self.nSelLargeRFlat,
            "massplane_77": np.array(
                [
                    self.h1_m[self.btagHigh_2b2b],
                    self.h2_m[self.btagHigh_2b2b],
                ]
            ).T,
            "leadingLargeRpT": self.leadingLargeRpt,
            "trigger_leadingLargeRpT": self.leadingLargeRpt[
                (self.trigger & self.leadingLargeRmassGreater100)
            ],
            "triggerRef_leadingLargeRpT": self.leadingLargeRpt[
                (self.triggerRef & self.leadingLargeRmassGreater100)
            ],
            "trigger_leadingLargeRm": self.leadingLargeRm[
                (self.trigger & self.leadingLargeRpTGreater500)
            ],
            "triggerRef_leadingLargeRm": self.leadingLargeRm[
                (self.triggerRef & self.leadingLargeRpTGreater500)
            ],
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
