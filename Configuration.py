import os

import tools.HistFillerTools


class Setup:
    def __init__(self, args):
        if args.batchMode:
            # to run on same cpu core as main program, even with cpus=1 a child
            # process is spawned on another cpu core if not dummy
            import multiprocessing.dummy as multiprocessing

        else:
            import multiprocessing

        outputPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/"

        if args.file:
            # make full output path
            self.filelist = [args.file]
            fileParts = self.filelist[0].split("/")
            dataset = fileParts[-2]
            datasetPath = outputPath + dataset
            file = fileParts[-1]
            if not os.path.isdir(datasetPath):
                os.makedirs(datasetPath)
            self.histOutFile = outputPath + dataset + "/" + file + ".h5"
        else:
            # default to mc 20 signal
            self.filelist = tools.HistFillerTools.ConstructFilelist("mc20_SM")
            # make hist out file name from filename
            dataset = self.filelist[0].split("/")
            histOutFileName = "hists-" + dataset[-2] + ".h5"
            self.histOutFile = outputPath + histOutFileName

        # figure out which vars to load from analysis script
        start = 'vars_arr["'
        end = '"]'
        self.vars = []
        for line in open("/lustre/fs22/group/atlas/freder/hh/pyhh/Analysis.py", "r"):
            if "vars_arr[" in line:
                if "#" not in line:
                    self.vars.append((line.split(start))[1].split(end)[0])

        # general settings
        if args.debug:
            self.filelist = self.filelist[:3]
            self.histOutFile = outputPath + "hists-debug.h5"
            self.nEvents = 1000
            self.cpus = 1
            self.batchSize = 1000
        else:
            self.nEvents = "All"
            self.batchSize = 20_000
            if args.cpus:
                self.cpus = args.cpus
            elif args.batchMode:
                self.cpus = 1
            else:
                self.cpus = multiprocessing.cpu_count() - 2

        # auto setup blind if data
        if any("data" in file for file in self.filelist):
            self.isData = True
        else:
            self.isData = False
        self.BLIND = self.isData
