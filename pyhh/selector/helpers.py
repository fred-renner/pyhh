import itertools
from operator import xor

import numpy as np
import vector
from plotter.tools import CR_hh, Xhh


def largeRSelect(self, event):
    self.nLargeR[event] = self.lrj_pt[event].shape[0]
    # pt, eta cuts and sort
    # Jet/ETmiss recommendation 200 < pT < 3000 GeV, 50 < m < 600 GeV
    ptCuts = (self.lrj_pt[event] > 200e3) & (self.lrj_pt[event] < 3000e3)
    mCuts = (self.lrj_m[event] > 50e3) & (self.lrj_m[event] < 600e3)
    # this eta cut is in old boosted analysis
    etaCut = np.abs(self.lrj_eta[event]) < 2.0
    selected = np.array((ptCuts & mCuts & etaCut), dtype=bool)
    # counting
    nJetsSelected = np.count_nonzero(selected)
    self.nLargeRBasicSelected[event] = nJetsSelected
    if nJetsSelected >= 2:
        # selected is a bool array for lrj_pt[event]
        # now sort by getting jet indices in decreasing order of pt
        ptOrder = np.flip(np.argsort(self.lrj_pt[event]))
        # now choose the according bools from selected in pt order
        selectedInPtOrder = selected[ptOrder]
        # applying these bools back on ptOrder gives the corresponding
        # Indices of the jets that come from self.lrj_pt[event]
        indices = ptOrder[selectedInPtOrder]
        self.selPtSort_lrjIndices[event] = indices
        jetPt1 = self.lrj_pt[event][self.selPtSort_lrjIndices[event][0]]
        jetPt2 = self.lrj_pt[event][self.selPtSort_lrjIndices[event][1]]
        if (jetPt1 > 450e3) & (jetPt2 > 250e3):
            self.selectedTwoLargeRevents[event] = True
            self.selLargeR1Index[event] = self.selPtSort_lrjIndices[event][0]
            self.selLargeR2Index[event] = self.selPtSort_lrjIndices[event][1]


def TriggerReference(self, event):
    # check if event has at least one Large R
    if self.nLargeR[event] > 0:
        self.atLeastOneLargeR[event] = True
        # cannot use selPtSort as they are selected!
        maxPtIndex = np.argmax(self.lrj_pt[event])
        self.leadingLargeRpt[event] = self.lrj_pt[event][maxPtIndex]
        maxMIndex = np.argmax(self.lrj_m[event])
        self.leadingLargeRm[event] = self.lrj_m[event][maxMIndex]
        if self.leadingLargeRm[event] > 100_000.0:
            self.leadingLargeRmassGreater100[event] = True
        if self.leadingLargeRpt[event] > 500_000.0:
            self.leadingLargeRpTGreater500[event] = True


def getVRs(self, event):
    if self.selectedTwoLargeRevents[event]:
        # get their corresponding vr jets, (vector of vectors)
        # need ._values as it comes as STL vector uproot object, instead of
        # .tolist() it comes already as np.ndarray
        j1_VRs = self.vr_btag_77[event]._values[self.selLargeR1Index[event]]._values
        j2_VRs = self.vr_btag_77[event]._values[self.selLargeR2Index[event]]._values
        # count their tags
        j1_VRs_Btag = np.count_nonzero(j1_VRs)
        j2_VRs_Btag = np.count_nonzero(j2_VRs)
        j1_VRs_noBtag = len(j1_VRs) - j1_VRs_Btag
        j2_VRs_noBtag = len(j2_VRs) - j2_VRs_Btag

        # this is not mutually exclusive with >=
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

        self.dR_VR_h1[event] = self.vr_deltaR12[event][self.selLargeR1Index[event]]
        self.dR_VR_h2[event] = self.vr_deltaR12[event][self.selLargeR2Index[event]]

        # get pt of the btagged ones
        h1_btag_VR_pts = self.vr_pt[event][self.selLargeR1Index[event]][
            j1_VRs.astype(bool)
        ]
        h2_btag_VR_pts = self.vr_pt[event][self.selLargeR2Index[event]][
            j2_VRs.astype(bool)
        ]
        # the following is fine as they come pt sorted already
        # h1
        if h1_btag_VR_pts.shape[0] == 2:
            self.pt_h1_btag_vr_1[event] = h1_btag_VR_pts[0]
            self.pt_h1_btag_vr_2[event] = h1_btag_VR_pts[1]
        # h2
        if h2_btag_VR_pts.shape[0] == 2:
            self.pt_h2_btag_vr_1[event] = h2_btag_VR_pts[0]
            self.pt_h2_btag_vr_2[event] = h2_btag_VR_pts[1]


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
        self.h1_p4 = vector.obj(
            pt=self.lrj_pt[event][self.selLargeR1Index[event]],
            eta=self.lrj_eta[event][self.selLargeR1Index[event]],
            phi=self.lrj_phi[event][self.selLargeR1Index[event]],
            m=self.lrj_m[event][self.selLargeR1Index[event]],
        )
        self.h2_p4 = vector.obj(
            pt=self.lrj_pt[event][self.selLargeR2Index[event]],
            eta=self.lrj_eta[event][self.selLargeR2Index[event]],
            phi=self.lrj_phi[event][self.selLargeR2Index[event]],
            m=self.lrj_m[event][self.selLargeR2Index[event]],
        )
        self.m_h1[event] = self.lrj_m[event][self.selLargeR1Index[event]]
        self.m_h2[event] = self.lrj_m[event][self.selLargeR2Index[event]]
        self.m_hh[event] = (self.h1_p4 + self.h2_p4).mass

        self.pt_h1[event] = self.lrj_pt[event][self.selLargeR1Index[event]]
        self.pt_h2[event] = self.lrj_pt[event][self.selLargeR2Index[event]]
        self.pt_hh[event] = (self.h1_p4 + self.h2_p4).pt
        self.pt_hh_scalar[event] = self.h1_p4.pt + self.h2_p4.pt


def vbfSelect(self, event):
    # https://indico.cern.ch/event/1184186/contributions/4974848/attachments/2483923/4264523/2022-07-21%20Introduction.pdf
    if self.selectedTwoLargeRevents[event]:
        etaCut = np.abs(self.srj_eta[event]) < 4.5
        selected = np.array((etaCut), dtype=bool)
        nJetsSelected = np.count_nonzero(selected)
        passedJets_p4 = []
        if nJetsSelected >= 2:
            # get the indices of the selected ones
            selectedIndices = np.nonzero(selected)[0]
            for i in selectedIndices:
                # check if they are outside of the large R's
                jet_p4 = vector.obj(
                    pt=self.srj_pt[event][i],
                    eta=self.srj_eta[event][i],
                    phi=self.srj_phi[event][i],
                    m=self.srj_m[event][i],
                )
                if (
                    (jet_p4.deltaR(self.h1_p4) > 1.4)
                    & (jet_p4.deltaR(self.h2_p4) > 1.4)
                    & (jet_p4.pt > 20e3)
                ):
                    passedJets_p4.append(jet_p4)
        # save two leading vbf jets if they pass further cuts
        if len(passedJets_p4) >= 2:
            # calc mass and eta of all jet jet combinations
            # e.g. [(0, 1), (0, 2), (1, 2)] to cut
            jet_combinations = np.array(
                list(itertools.combinations(range(len(passedJets_p4)), 2))
            )
            m_jjs = np.ndarray(jet_combinations.shape[0], dtype=float)
            m_jj_pass = np.ndarray(jet_combinations.shape[0], dtype=bool)
            eta_jj_pass = np.ndarray(jet_combinations.shape[0], dtype=bool)
            for i, twoIndices in enumerate(jet_combinations):
                jetX = passedJets_p4[twoIndices[0]]
                jetY = passedJets_p4[twoIndices[1]]
                m_jjs[i] = (jetX + jetY).mass
                m_jj_pass[i] = m_jjs[i] > 1e6
                eta_jj_pass[i] = np.abs(jetX.eta - jetY.eta) > 3
            passMassEta = m_jj_pass & eta_jj_pass
            if np.count_nonzero(passMassEta) >= 1:
                largestPtSum = 0
                for twoIndices in jet_combinations[passMassEta]:
                    jet1Pt = passedJets_p4[twoIndices[0]].pt
                    jet2Pt = passedJets_p4[twoIndices[1]].pt
                    PtSum = jet1Pt + jet2Pt
                    if largestPtSum < PtSum:
                        largestPtSum = PtSum
                        if jet1Pt < jet2Pt:
                            twoIndices = twoIndices[::-1]
                        self.VBFjetsPass[event] = True
                        jetX = passedJets_p4[twoIndices[0]]
                        jetY = passedJets_p4[twoIndices[1]]
                        self.pt_vbf1[event] = jetX.pt
                        self.pt_vbf2[event] = jetY.pt
                        self.m_jjVBF[event] = (jetX + jetY).mass


def hh_selections(self, event):
    # calculate region variables
    if self.selectedTwoLargeRevents[event]:
        self.X_HH[event] = Xhh(self.m_h1[event], self.m_h2[event])
        self.CR_hh[event] = CR_hh(self.m_h1[event], self.m_h2[event])

        self.SR[event] = self.X_HH[event] < 1.6
        self.VR[event] = (self.X_HH[event] > 1.6) & (self.CR_hh[event] < 100.0e3)
        self.CR[event] = (self.CR_hh[event] > 100e3) & (
            (self.m_h1[event] + self.m_h2[event]) > 130e3
        )

        # CR VR overlap for stats
        # https://indico.cern.ch/event/1239101/contributions/5216057/attachments/2575156/4440353/hh4b_230112.pdf
        # self.CR[event] = (self.CR_hh[event] > 100e3) & (
        #     (self.m_h1[event] + self.m_h2[event]) > 130e3
        # )


def fill(self, selections, finalSel, results):
    for hist in finalSel.keys():
        # if list of lists build var/weights manually
        if isinstance(finalSel[hist]["var"], list) or (
            finalSel[hist]["var"].dtype == object
        ):
            finalSel[hist]["var"], w = flatten2d(
                finalSel[hist]["var"],
                self.weights,
                finalSel[hist]["sel"],
            )
            finalSel[hist]["sel"] = None
        else:
            w = None
            # get final values with according weights
        results[hist] = resultWithWeights(
            self,
            var=finalSel[hist]["var"],
            sel=finalSel[hist]["sel"],
            userWeight=w,
        )

        # add massplane
    for region, selectionBool in selections.items():
        results["massplane." + region] = [
            np.array(
                [
                    self.m_h1[selectionBool],
                    self.m_h2[selectionBool],
                ]
            ).T,
            np.array(self.weights[selectionBool]),
        ]
    return


def resultWithWeights(self, var, sel=None, userWeight=None):
    """
    select varSelDicts of vars and attach weights

    Parameters
    ----------
    var : np.ndarray
        array with varSelDicts
    sel : np.ndarray, optional
        array holding booleans to select on var, by default None
    weight : float, optional
        weights for the hists, by default None

    Returns
    -------
    varWithWeights : list
        selected vars with weights
    """

    if sel is None:
        if userWeight is None:
            varWithWeights = [var, self.weights]
        else:
            varWithWeights = [var, userWeight]
    else:
        if userWeight is None:
            varWithWeights = [var[sel], self.weights[sel]]

    return varWithWeights


def flatten2d(arr, weights, sel=None):
    """
    flatten 2d inhomogeneous array and replicate weights values per event

    Parameters
    ----------
    arr : list of lists
        list holding lists of values
    weights : nd.array
        weights per event same shape as len(arr)
    sel : np.array, optional
        event selection, by default None
    Returns
    -------
    flatArr, flatArrWeights

    """
    if sel is None:
        selection = np.full(len(arr), True, dtype=bool)
    else:
        selection = sel

    # get selection, list is dimensional imhomogeneous
    selectedArr = list(itertools.compress(arr, selection))
    # itertools.chain.from_iterable flattens
    flatArr = np.array(list(itertools.chain.from_iterable(selectedArr)))
    # replicate weight values per number of objects in event
    flatArrWeights = np.array(
        list(
            itertools.chain.from_iterable(
                [
                    np.repeat(w, len(values))
                    for values, w in zip(selectedArr, weights[selection])
                ]
            )
        )
    )

    return flatArr, flatArrWeights
