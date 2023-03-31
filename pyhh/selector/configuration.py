import os
import pathlib
import selector.tools
import selector.analysis

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
        self.histOutFile = histpath + file + ".h5"
        self.dumpFile = dumppath + file + ".h5"

        # figure out which vars to load from analysis script
        start = 'vars_arr["'
        end = '"]'
        self.vars = []
        analysisPath = pathlib.Path(__file__).parent / "analysis.py"

        for line in open(analysisPath, "r"):
            if "vars_arr[" in line:
                if "#" not in line:
                    self.vars.append((line.split(start))[1].split(end)[0])

        # auto setup blind if data
        if "data1" in self.file or "data2" in self.file:
            self.isData = True
        else:
            self.isData = False
        self.BLIND = self.isData

        # if fill hists
        self.fill = args.fill

        # init dump file for dumping of selected vars
        self.dump = args.dump
        if args.dump:
            self.dump = args.dump
            selector.tools.initDumpFile(self.dumpFile)

        # basic settings
        if args.debug:
            self.histOutFile = outputPath + "histograms/hists-debug.h5"
            self.nEvents = 1
            self.batchSize = 1
            self.cpus = 1
            self.fill = True
            self.dump = True
        else:
            self.nEvents = "All"
            self.batchSize = 20_000
            if args.batchMode:
                self.cpus = 1
            else:
                self.cpus = multiprocessing.cpu_count() - 2

