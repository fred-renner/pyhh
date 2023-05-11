import os

import selector.analysis
import selector.tools

outputPath = "/lustre/fs22/group/atlas/freder/hh/run/"


class setup:
    def __init__(self, args):
        if args.batchMode:
            # to run on same cpu core as main program, even with cpus=1 a child
            # process is spawned on another cpu core if not dummy
            import multiprocessing.dummy as multiprocessing

        else:
            import multiprocessing

        # auto setup outputfile from filename
        self.file = args.file
        # default to an mc 20 signal file
        if not args.file:
            self.file = selector.tools.ConstructFilelist("mc20_SM", verbose=False)[0]
        fileParts = self.file.split("/")
        dataset = fileParts[-2]
        file = fileParts[-1]

        histpath = outputPath + "histograms/" + dataset + "/"
        dumppath = outputPath + "dump/" + dataset + "/"

        if args.fill:
            if not os.path.isdir(histpath):
                os.makedirs(histpath)
        if args.dump:
            if not os.path.isdir(dumppath):
                os.makedirs(dumppath)
        self.hist_out_file = histpath + file + ".h5"
        self.dump_file = dumppath + file + ".h5"

        # auto setup blind if data
        if "data1" in self.file or "data2" in self.file:
            self.isData = True
        else:
            self.isData = False
        self.BLIND = self.isData

        # if fill hists
        self.fill = args.fill
        self.do_systematics = True
        self.systematics = [
            "JET_JER",
            # "JET_JMR",
            # "JET_JD2R",
            "JET_Comb_Modelling",
            # "JET_Comb_Baseline",
            # "JET_Comb_Tracking",
            # "JET_Comb_TotalStat",
            # "JET_MassRes_Hbb",
            # "JET_MassRes_Top",
            # "JET_MassRes_WZ",
            # "JET_Rtrk_Modelling",
            # "JET_Rtrk_Baseline",
            # "JET_Rtrk_Tracking",
            # "JET_Rtrk_TotalStat",
        ]

        # init dump file for dumping of selected vars
        self.dump = args.dump

        # basic settings
        if args.debug:
            self.hist_out_file = outputPath + "histograms/hists-debug.h5"
            self.nEvents = 100
            self.batchSize = 50
            self.cpus = 1
            self.fill = True
            self.dump = True
            if not os.path.isdir(dumppath):
                os.makedirs(dumppath)
        else:
            self.nEvents = "All"
            self.batchSize = 20_000
            if args.batchMode:
                self.cpus = 1
            else:
                self.cpus = multiprocessing.cpu_count() - 2
