import numpy as np
import vector
import time
import awkward as ak
import uproot
from operator import xor

np.set_printoptions(threshold=np.inf)


def Run(batch, metaData, tree, vars):
    vars_arr = tree.arrays(
        vars, entry_start=batch[0], entry_stop=batch[1], library="np"
    )
    objects = ObjectSelection(metaData, vars_arr)
    objects.select()
    return objects.returnResults()


def get_lumi(years: list):
    """
    Get luminosity value per given year in fb-1

    Parameters
    ----------
    years: list
        Years corresponding to desired lumi

    """
    lumi = {
        "2015": 3.4454,
        "2016": 33.4022,
        "2017": 44.6306,
        "2018": 58.7916,
        # "2022": 140.06894,  ############## just for testing ##################
        "all": 140.06894,
    }
    l = 0
    for yr in years:
        l += lumi[yr]

    return l


class ObjectSelection:
    def __init__(self, metaData, vars_arr):
        """
        initializing all vars and reserving memory

        Parameters
        ----------
        metaData : dict
            mainly info for weights
        vars_arr : dict
            holding vars loaded with uproot
        """
        self.mc = True if len(metaData) != 0 else False
        if self.mc:
            lumi = get_lumi(metaData["dataYears"])
            # crosssection comes in nb-1 (* 1e6 = fb-1)
            sigma = metaData["crossSection"] * 1e6
            sum_of_weights = metaData["initial_sum_of_weights"]
            self.weightFactor = sigma * lumi * metaData["genFiltEff"] / sum_of_weights
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
        self.srj_pt = vars_arr["recojet_antikt4_NOSYS_pt"]
        self.srj_eta = vars_arr["recojet_antikt4_NOSYS_eta"]
        self.srj_phi = vars_arr["recojet_antikt4_NOSYS_phi"]
        self.srj_m = vars_arr["recojet_antikt4_NOSYS_m"]
        self.vr_btag_77 = vars_arr["recojet_antikt10_NOSYS_leadingVRTrackJetsBtag_DL1r_FixedCutBEff_77"]
        self.vr_deltaR12 = vars_arr["recojet_antikt10_NOSYS_leadingVRTrackJetsDeltaR12"]
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
        self.subleadingLargeRindex = np.copy(intInitArray)

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
        self.SR = np.copy(boolInitArray)
        self.CR = np.copy(boolInitArray)
        self.VR = np.copy(boolInitArray)
        self.leadingLargeRmassGreater100 = np.copy(boolInitArray)
        self.leadingLargeRpTGreater500 = np.copy(boolInitArray)

        # float init
        self.weights = np.full(self.nEvents, 1.0, dtype=float)
        floatInitArray = np.full(self.nEvents, -1.0, dtype=float)
        self.m_hh = np.copy(floatInitArray)
        self.m_h1 = np.copy(floatInitArray)
        self.m_h2 = np.copy(floatInitArray)
        self.pt_h1 = np.copy(floatInitArray)
        self.pt_h2 = np.copy(floatInitArray)
        self.pt_hh = np.copy(floatInitArray)
        self.pt_hh_scalar = np.copy(floatInitArray)
        self.dR_h1 = np.copy(floatInitArray)
        self.dR_h2 = np.copy(floatInitArray)
        self.X_HH = np.copy(floatInitArray)
        self.R_VR = np.copy(floatInitArray)
        self.R_CR = np.copy(floatInitArray)
        self.truth_m_hh = np.copy(floatInitArray)
        self.leadingLargeRpt = np.copy(floatInitArray)
        self.leadingLargeRm = np.copy(floatInitArray)

    def select(self):
        for event in self.eventRange:
            # order matters!
            # if not self.vr_dontOverlap[event]:
            #     continue
            if self.mc:
                self.weights[event] = (
                    self.weightFactor
                    * self.vars_arr["pileupWeight_NOSYS"][event]
                    * self.vars_arr["mcEventWeights"][event][0]
                    # mcEventWeights[:][0] == generatorWeight_NOSYS
                )
            self.largeRSelectSort(event)
            self.getLeadingLargeR(event)
            self.getLeadingLargeRcuts(event)
            self.getVRs(event)
            self.hh_p4(event)
            # self.vbfSelect(event)
            self.hh_regions(event)
            if self.hasTruth:
                self.truth_mhh(event)
        self.nTotalSelLargeR()

    def largeRSelectSort(self, event):
        # pt, eta cuts and sort
        # Jet/ETmiss recommendation 200 < pT < 3000 GeV, 50 < m < 600 GeV
        ptCuts = (self.lrj_pt[event] > 0.2e6) & (self.lrj_pt[event] < 3e6)
        mCuts = (self.lrj_m[event] > 0.05e6) & (self.lrj_m[event] < 0.6e6)
        etaCut = np.abs(self.lrj_eta[event]) < 2.0
        selected = np.array((ptCuts & mCuts & etaCut), dtype=bool)
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
            ptOrder = np.flip(np.argsort(self.lrj_pt[event]))
            # now choose the according bools from selected in pt order
            selectedInPtOrder = selected[ptOrder]
            # applying these bools back on ptOrder gives the corresponding
            # Indices of the jets that come from self.lrj_pt[event]
            self.selPtSort_lrj[event] = ptOrder[selectedInPtOrder]
            self.leadingLargeRindex[event] = self.selPtSort_lrj[event][0]
            self.subleadingLargeRindex[event] = self.selPtSort_lrj[event][1]

    def getLeadingLargeR(self, event):
        # check which event has at least one Large R
        if self.nLargeR[event] > 0:
            self.atLeastOneLargeR[event] = True
            # cannot use selPtSort as they are selected!
            maxPtIndex = self.leadingLargeRindex[event]
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
            binnedObject=self.m_hh, Counts=self.nLargeRSelected
        )

    def getVRs(self, event):
        if self.selectedTwoLargeRevents[event]:
            # print(self.vr_btag_77[event]._values)
            # print(self.vr_btag_77[event]._values[0])
            # get leading jets
            j1_index = self.leadingLargeRindex[event]
            j2_index = self.subleadingLargeRindex[event]
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

            self.dR_h1[event] = self.vr_deltaR12[event][j1_index]
            self.dR_h2[event] = self.vr_deltaR12[event][j2_index]

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

    def hh_p4(self, event):
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
            self.m_h1[event] = self.lrj_m[event][leadingLargeRindex]
            self.m_h2[event] = self.lrj_m[event][subleadingLargeRindex]
            self.m_hh[event] = (h1_p4 + h2_p4).mass
            self.pt_h1[event] = self.lrj_pt[event][leadingLargeRindex]
            self.pt_h2[event] = self.lrj_pt[event][subleadingLargeRindex]
            self.pt_hh[event] = (h1_p4 + h2_p4).pt
            self.pt_hh_scalar[event] = h1_p4.pt + h2_p4.pt

    def vbfSelect(self, event):
        if self.selectedTwoLargeRevents[event]:
            etaCut = np.abs(self.srj_eta[event]) < 4.5
            # I dont understand this one, also misses jvt pass tight
            someCut = (self.srj_pt[event] > 60) & (np.abs(self.srj_eta[event]) > 2.4)
            print(range(self.srj_pt[event].shape[0]))
            print(self.srj_pt[event])
            # task is to find two then make bool true for this event and write them
            # could throw away event indices that doesnt pass cuts
            for jet in range(self.srj_pt[event].shape[0]):
                srj_p4 = vector.obj(
                    pt=self.srj_pt[event][jet],
                    eta=self.srj_eta[event][jet],
                    phi=self.srj_phi[event][jet],
                    m=self.srj_m[event][jet],
                )
            # breakpoint
            # they also write dRmin per jet to both H_p4

    def hh_regions(self, event):
        # from roosted branch
        validation_shift = 1.03
        control_shift = 1.05
        Xhh_cut = 1.6
        validation_cut = 30.0
        control_cut = 45.0
        m_h1_center = 124.0
        m_h2_center = 117.0
        # fm_h1 from signal region optimization:
        # https://indico.cern.ch/event/1191598/contributions/5009137/attachments/2494578/4284249/HH4b20220818.pdf
        fm_h1 = 1500.0
        fm_h2 = 1900.0
        # calculate region variables
        if self.selectedTwoLargeRevents[event]:
            self.X_HH[event] = np.sqrt(
                np.power(
                    (self.m_h1[event] - m_h1_center) / (fm_h1 / self.m_h1[event]), 2
                )
                + np.power(
                    (self.m_h2[event] - m_h2_center) / (fm_h2 / self.m_h2[event]), 2
                )
            )

            self.R_VR[event] = np.sqrt(
                np.power((self.m_h1[event] - m_h1_center) * validation_shift, 2)
                + np.power((self.m_h2[event] - m_h2_center) * validation_shift, 2)
            )
            self.R_CR[event] = np.sqrt(
                np.power((self.m_h1[event] - m_h1_center) * control_shift, 2)
                + np.power((self.m_h2[event] - m_h2_center) * control_shift, 2)
            )

            self.SR[event] = self.X_HH[event] < Xhh_cut
            self.VR[event] = (self.X_HH[event] > 1.6) & (
                self.R_VR[event] < validation_cut
            )
            self.CR[event] = (
                (self.X_HH[event] > 1.6)
                & (self.R_VR[event] > validation_cut)
                & (self.R_CR[event] < control_cut)
            )

    def returnResults(self):
        """
        lookup table for histname, variable to write, selection to apply

        Returns
        -------
        result : dict
            key: hist, holding tuple: (values, weights)
        """
        finalSel = {
            "truth_mhh": {
                "var": self.truth_m_hh,
                "sel": None,
            },
            "mhh": {
                "var": self.m_hh,
                "sel": self.btagHigh_2b2b,
            },
            "mh1": {
                "var": self.m_h1,
                "sel": self.btagHigh_2b2b,
            },
            "mh2": {
                "var": self.m_h2,
                "sel": self.btagHigh_2b2b,
            },
            "pt_h1": {
                "var": self.pt_h1,
                "sel": self.btagHigh_2b2b,
            },
            "pt_h2": {
                "var": self.pt_h2,
                "sel": self.btagHigh_2b2b,
            },
            "pt_hh": {
                "var": self.pt_hh,
                "sel": self.btagHigh_2b2b,
            },
            "pt_hh_scalar": {
                "var": self.pt_hh_scalar,
                "sel": self.btagHigh_2b2b,
            },
            "dR_h1": {
                "var": self.dR_h1,
                "sel": self.btagHigh_2b2b,
            },
            "dR_h2": {
                "var": self.dR_h2,
                "sel": self.btagHigh_2b2b,
            },
            "nTriggerPass_mhh": {
                "var": self.m_hh,
                "sel": self.trigger,
            },
            "nTwoLargeR_mhh": {
                "var": self.m_hh,
                "sel": self.nLargeR >= 2,
            },
            "nTwoSelLargeR_mhh": {
                "var": self.m_hh,
                "sel": self.selectedTwoLargeRevents,
            },
            "btagLow_1b1j_mhh": {
                "var": self.m_hh,
                "sel": self.btagLow_1b1j,
            },
            "btagLow_2b1j_mhh": {
                "var": self.m_hh,
                "sel": self.btagLow_2b1j,
            },
            "btagLow_2b2j_mhh": {
                "var": self.m_hh,
                "sel": self.btagLow_2b2j,
            },
            "btagHigh_1b1b_mhh": {
                "var": self.m_hh,
                "sel": self.btagHigh_1b1b,
            },
            "btagHigh_2b1b_mhh": {
                "var": self.m_hh,
                "sel": self.btagHigh_2b1b,
            },
            "btagHigh_2b2b_mhh": {
                "var": self.m_hh,
                "sel": self.btagHigh_2b2b,
            },
            "nTotalSelLargeR": {
                "var": self.nSelLargeRFlat,
                "sel": "ones",
            },
            "leadingLargeRpT": {
                "var": self.leadingLargeRpt,
                "sel": None,
            },
            "leadingLargeRpT_trigger": {
                "var": self.leadingLargeRpt,
                "sel": self.trigger,
            },
            "trigger_leadingLargeRpT": {
                "var": self.leadingLargeRpt,
                "sel": (self.trigger & self.leadingLargeRmassGreater100),
            },
            "triggerRef_leadingLargeRpT": {
                "var": self.leadingLargeRpt,
                "sel": (self.triggerRef & self.leadingLargeRmassGreater100),
            },
            "trigger_leadingLargeRm": {
                "var": self.leadingLargeRm,
                "sel": (self.trigger & self.leadingLargeRpTGreater500),
            },
            "triggerRef_leadingLargeRm": {
                "var": self.leadingLargeRm,
                "sel": (self.triggerRef & self.leadingLargeRpTGreater500),
            },
        }

        results = {}
        for hist in finalSel.keys():
            results[hist] = self.resultWithWeights(
                finalSel[hist]["var"], finalSel[hist]["sel"]
            )

        # 2D by hand
        results["massplane_77"] = (
            np.array(
                [
                    self.m_h1[self.btagHigh_2b2b],
                    self.m_h2[self.btagHigh_2b2b],
                ]
            ).T,
            np.array(self.weights[self.btagHigh_2b2b]),
        )

        return results

    def resultWithWeights(self, var, sel=None):
        """
        select values of vars and attach weights

        Parameters
        ----------
        var : np.ndarray
            array with values

        sel : np.ndarray, optional
            array holding a booleans to select on var , by default None

        Returns
        -------
        out : list
            selected vars with weights
        """
        if sel is None:
            return [var, self.weights]
        if sel is "ones":
            return [var, np.ones(var.shape)]
        else:
            return [var[sel], self.weights[sel]]

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
