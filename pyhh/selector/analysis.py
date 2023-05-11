import copy
import itertools

import numpy as np
import selector.tools
import selector.helpers

np.set_printoptions(threshold=np.inf)


def run(batch, config, metaData, tree):
    vars_arr = tree.arrays(
        config.load_vars, entry_start=batch[0], entry_stop=batch[1], library="np"
    )
    objects = ObjectSelection(config, metaData, vars_arr)
    objects.select()

    return objects.returnResults()


# event level vars
boolVars = [
    "atLeastOneLargeR",
    "selectedTwoLargeRevents",
    "btagLow_1b1j",
    "btagLow_2b1j",
    "btagLow_2b2j",
    "btagHigh_1b1b",
    "btagHigh_2b1b",
    "btagHigh_2b2b",
    "SR",
    "CR",
    "VR",
    "VBFjetsPass",
    "leadingLargeRmassGreater100",
    "leadingLargeRpTGreater500",
]
floatVars = [
    "weights",
    "m_hh",
    "m_h1",
    "m_h2",
    "pt_h1",
    "pt_h2",
    "pt_hh",
    "pt_hh_scalar",
    "dR_VR_h1",
    "dR_VR_h2",
    "X_HH",
    "CR_hh",
    "truth_m_hh",
    "leadingLargeRpt",
    "leadingLargeRm",
    "pt_vbf1",
    "pt_vbf2",
    "m_jjVBF",
    "pt_h1_btag_vr_1",
    "pt_h1_btag_vr_2",
    "pt_h2_btag_vr_1",
    "pt_h2_btag_vr_2",
]


class ObjectSelection:
    def __init__(self, config, metaData, vars_arr):
        """
        initializing all vars and reserving memory

        Parameters
        ----------
        config : object
            basic setup configured in configuration.py
        metaData : dict
            mainly info for weights
        vars_arr : dict
            holding vars loaded with uproot
        """
        self.config = config
        if config.isData:
            self.mc = False
            self.data = True
        else:
            self.mc = True
            self.data = False
        if self.mc:
            lumi = selector.tools.get_lumi(metaData["dataYears"])
            # crosssection comes in nb-1 (* 1e6 = fb-1)
            xsec = metaData["crossSection"] * 1e6
            sum_of_weights = metaData["sum_of_weights"]
            self.weightFactor = (
                xsec
                * lumi
                * metaData["kFactor"]
                * metaData["genFiltEff"]
                / sum_of_weights
            )
        if any("truth" in x for x in vars_arr):
            self.hasTruth = True
        else:
            self.hasTruth = False

        # fmt: off
        # recoUFOjet_antikt10
        # recojet_antikt10

        self.vars_arr = vars_arr
        self.lrj_pt = vars_arr["recojet_antikt10_NOSYS_pt"]
        self.lrj_eta = vars_arr["recojet_antikt10_NOSYS_eta"]
        self.lrj_phi = vars_arr["recojet_antikt10_NOSYS_phi"]
        self.lrj_m = vars_arr["recojet_antikt10_NOSYS_m"]
        self.srj_pt = vars_arr["recojet_antikt4_NOSYS_pt"]
        self.srj_eta = vars_arr["recojet_antikt4_NOSYS_eta"]
        self.srj_phi = vars_arr["recojet_antikt4_NOSYS_phi"]
        self.srj_m = vars_arr["recojet_antikt4_NOSYS_m"]
        self.vr_btag_77 = vars_arr["recojet_antikt10_NOSYS_leadingVRTrackJetsBtag_DL1r_FixedCutBEff_77"]
        self.vr_pt = vars_arr["recojet_antikt10_NOSYS_leadingVRTrackJetsPt"]
        self.vr_deltaR12 = vars_arr["recojet_antikt10_NOSYS_leadingVRTrackJetsDeltaR12"]
        self.vr_dontOverlap = vars_arr["passRelativeDeltaRToVRJetCut"]
            
        if self.mc:
            # mc20 
            # r13144
            if "trigPassed_HLT_j420_a10t_lcw_jes_40smcINF_L1J100" in vars_arr: 
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
        if self.data:
            yr = metaData["dataYear"]
            # same trigger as reference for 2015 and 2016
            if yr == "2015": 
                self.trigger = vars_arr["trigPassed_HLT_j360_a10_lcw_sub_L1J100"]
                self.triggerRef = vars_arr["trigPassed_HLT_j360_a10_lcw_sub_L1J100"]
            if yr == "2016": 
                self.trigger = vars_arr["trigPassed_HLT_j420_a10_lcw_L1J100"]
                self.triggerRef = vars_arr["trigPassed_HLT_j420_a10_lcw_L1J100"]
            if yr == "2017": 
                self.trigger = vars_arr["trigPassed_HLT_j420_a10t_lcw_jes_40smcINF_L1J100"]
                self.triggerRef = vars_arr["trigPassed_HLT_j390_a10t_lcw_jes_30smcINF_L1J100"]
            if yr == "2018": 
                self.trigger = vars_arr["trigPassed_HLT_j420_a10t_lcw_jes_35smcINF_L1J100"]
                self.triggerRef = vars_arr["trigPassed_HLT_j390_a10t_lcw_jes_30smcINF_L1J100"]
        # fmt: on

        # event amount per iteration
        self.nEvents = len(self.lrj_pt)
        self.eventRange = range(self.nEvents)

        # vectors of vectors init
        # if we don't know the size, lists are faster
        self.selPtSort_lrjIndices = [[] for x in self.eventRange]

        # reserve np arrays and write to object
        # int init
        intInitArray = np.full(self.nEvents, -1, dtype=int)
        self.nLargeR = np.copy(intInitArray)
        self.nLargeRBasicSelected = np.copy(intInitArray)
        self.selLargeR1Index = np.copy(intInitArray)
        self.selLargeR2Index = np.copy(intInitArray)

        # bool init
        boolInitArray = np.zeros(self.nEvents, dtype=bool)
        for var in boolVars:
            setattr(self, var, np.copy(boolInitArray))

        # float init
        floatInitArray = np.full(self.nEvents, -1.0, dtype=float)
        for var in floatVars:
            setattr(self, var, np.copy(floatInitArray))
        # init weights to one
        self.weights = np.full(self.nEvents, 1.0, dtype=float)

    def select(self):
        """
        This does the actual analysis/selection steps
        """

        for event in self.eventRange:
            if self.trigger[event]:
                # order matters!
                if not self.vr_dontOverlap[event]:
                    continue
                if self.mc:
                    self.weights[event] = (
                        self.weightFactor
                        * self.vars_arr["pileupWeight_NOSYS"][event]
                        * self.vars_arr["mcEventWeights"][event][0]
                    )
                selector.helpers.largeRSelect(self, event)
                selector.helpers.TriggerReference(self, event)
                selector.helpers.getVRs(self, event)
                selector.helpers.hh_p4(self, event)
                selector.helpers.vbfSelect(self, event)
                selector.helpers.hh_selections(self, event)
                if self.hasTruth:
                    selector.helpers.truth_mhh(self, event)

    def returnResults(self):
        """
        This writes everything to a returned results dict.
        if fill hists: apply selections to variables and weights
        if dump variables: write bools and floats to results.

        Returns
        -------
        result : dict
            in the form: result[histKey] = [values, weights]
        """

        selections = {
            "trigger": self.trigger,
            "twoLargeR": self.selectedTwoLargeRevents,
            "SR_2b2j": self.SR & self.btagLow_2b2j & self.VBFjetsPass,
            "VR_2b2b": self.VR & self.btagHigh_2b2b & self.VBFjetsPass,
            "VR_2b2j": self.VR & self.btagLow_2b2j & self.VBFjetsPass,
            "CR_1b1b": self.CR & self.btagHigh_1b1b & self.VBFjetsPass,
            "CR_1b1j": self.CR & self.btagLow_1b1j & self.VBFjetsPass,
            "CR_2b1b": self.CR & self.btagHigh_2b1b & self.VBFjetsPass,
            "CR_2b1j": self.CR & self.btagLow_2b1j & self.VBFjetsPass,
            "CR_2b2b": self.CR & self.btagHigh_2b2b & self.VBFjetsPass,
            "CR_2b2j": self.CR & self.btagLow_2b2j & self.VBFjetsPass,
        }
        if self.config.BLIND:
            selections["SR_2b2b"] = np.zeros(self.nEvents, dtype=bool)
        else:
            selections["SR_2b2b"] = self.SR & self.btagHigh_2b2b & self.VBFjetsPass

        # make kinematics vars for all selections, e.g. mhh_CR_4b, mhh_CR_2b, etc.
        # and write to finalSel
        finalSel, kinematics = histvariableDefs(self)
        # e.g. region==CR_2b, selectionBool==boolArray_CR_2b
        for region, selectionBool in selections.items():
            for kinVar, kinVarDict in kinematics.items():
                # write selectionBool e.g. SR_4b
                kinVarDict["sel"] = selectionBool
                # need to make a deep copy as dict assignments just creates references
                finalSel[kinVar + "." + region] = copy.deepcopy(kinVarDict)

        results = {}
        # if fill hists go over all defined hists
        if self.config.fill:
            selector.helpers.fill(self, selections, finalSel, results)

        if self.config.dump:
            results["selections"] = selections
            results["bools"] = {}
            results["floats"] = {}
            for var in boolVars:
                results["bools"][var] = getattr(self, var)
            for var in floatVars:
                results["floats"][var] = getattr(self, var)

        return results


def histvariableDefs(self):
    # singular vars
    finalSel = {
        "truth_mhh": {
            "var": self.truth_m_hh,
            "sel": None,
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
            "sel": self.trigger & self.leadingLargeRmassGreater100,
        },
        "triggerRef_leadingLargeRpT": {
            "var": self.leadingLargeRpt,
            "sel": self.triggerRef & self.leadingLargeRmassGreater100,
        },
        "trigger_leadingLargeRm": {
            "var": self.leadingLargeRm,
            "sel": self.trigger & self.leadingLargeRpTGreater500,
        },
        "triggerRef_leadingLargeRm": {
            "var": self.leadingLargeRm,
            "sel": self.triggerRef & self.leadingLargeRpTGreater500,
        },
        "pt_lrj": {
            "var": self.lrj_pt,
            "sel": None,
        },
        "eta_lrj": {
            "var": self.lrj_eta,
            "sel": None,
        },
        "phi_lrj": {
            "var": self.lrj_phi,
            "sel": None,
        },
        "m_lrj": {
            "var": self.lrj_m,
            "sel": None,
        },
        "pt_srj": {
            "var": self.srj_pt,
            "sel": None,
        },
        "eta_srj": {
            "var": self.srj_eta,
            "sel": None,
        },
        "phi_srj": {
            "var": self.srj_phi,
            "sel": None,
        },
        "m_srj": {
            "var": self.srj_m,
            "sel": None,
        },
    }

    kinematics = {
        "m_hh": {
            "var": self.m_hh,
            "sel": None,
        },
        "m_h1": {
            "var": self.m_h1,
            "sel": None,
        },
        "m_h2": {
            "var": self.m_h2,
            "sel": None,
        },
        "m_hh_paper": {
            "var": self.m_hh,
            "sel": None,
        },
        "m_h1_paper": {
            "var": self.m_h1,
            "sel": None,
        },
        "m_h2_paper": {
            "var": self.m_h2,
            "sel": None,
        },
        "m_hh_lessBins": {
            "var": self.m_hh,
            "sel": None,
        },
        "m_h1_lessBins": {
            "var": self.m_h1,
            "sel": None,
        },
        "m_h2_lessBins": {
            "var": self.m_h2,
            "sel": None,
        },
        "pt_h1": {
            "var": self.pt_h1,
            "sel": None,
        },
        "pt_h2": {
            "var": self.pt_h2,
            "sel": None,
        },
        "pt_hh": {
            "var": self.pt_hh,
            "sel": None,
        },
        "pt_hh_scalar": {
            "var": self.pt_hh_scalar,
            "sel": None,
        },
        "dR_VR_h1": {
            "var": self.dR_VR_h1,
            "sel": None,
        },
        "dR_VR_h2": {
            "var": self.dR_VR_h2,
            "sel": None,
        },
        "pt_vbf1": {
            "var": self.pt_vbf1,
            "sel": None,
        },
        "pt_vbf2": {
            "var": self.pt_vbf2,
            "sel": None,
        },
        "m_jjVBF": {
            "var": self.m_jjVBF,
            "sel": None,
        },
        "pt_h1_btag_vr1": {
            "var": self.pt_h1_btag_vr_1,
            "sel": None,
        },
        "pt_h1_btag_vr2": {
            "var": self.pt_h1_btag_vr_2,
            "sel": None,
        },
        "pt_h2_btag_vr1": {
            "var": self.pt_h2_btag_vr_1,
            "sel": None,
        },
        "pt_h2_btag_vr2": {
            "var": self.pt_h2_btag_vr_2,
            "sel": None,
        },
    }
    return finalSel, kinematics
