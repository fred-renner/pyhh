import tools.HistFillerTools
import os


class Config:
    def __init__(self, args):

        if args.batchMode:
            # to run on same cpu core
            import multiprocessing.dummy as multiprocessing

            self.cpus = 1
        if args.file:
            self.filelist = [args.file]
            fileParts = self.filelist[0].split("/")
            dataset = fileParts[-2]
            datasetPath = "/lustre/fs22/group/atlas/freder/hh/run/histograms/" + dataset
            file = fileParts[-1]
            if not os.path.isdir(datasetPath):
                os.makedirs(datasetPath)

            self.histOutFile = (
                "/lustre/fs22/group/atlas/freder/hh/run/histograms/"
                + dataset
                + "/"
                + file
                + ".h5"
            )
        else:
            # default to mc 20 signal
            self.filelist = tools.HistFillerTools.ConstructFilelist("mc20_l1cvv1cv1")
            # self.filelist = tools.HistFillerTools.ConstructFilelist("mc20_ttbar")
            # self.filelist = tools.HistFillerTools.ConstructFilelist("run2")
            # make hist out file name from filename
            if "histOutFileName" not in locals():
                dataset = self.filelist[0].split("/")
                histOutFileName = "hists-" + dataset[-2] + ".h5"

            self.histOutFile = (
                "/lustre/fs22/group/atlas/freder/hh/run/histograms/" + histOutFileName
            )

        # figure out which vars to load from analysis script
        start = 'vars_arr["'
        end = '"]'
        self.vars = []
        for line in open(
            "/lustre/fs22/group/atlas/freder/hh/hh-analysis/Analysis.py", "r"
        ):
            if "vars_arr[" in line:
                if "#" not in line:
                    self.vars.append((line.split(start))[1].split(end)[0])

        # general settings
        if args.debug:
            self.filelist = self.filelist[:3]
            self.histOutFile = (
                "/lustre/fs22/group/atlas/freder/hh/run/histograms/hists-debug.h5"
            )
            self.nEvents = 1000
            self.cpus = 1
            self.batchSize = 1000
        else:
            self.nEvents = "All"
            self.batchSize = 20_000
            if args.cpus:
                self.cpus = args.cpus
            else:
                self.cpus = multiprocessing.cpu_count() - 4

        # auto setup blind if data
        if any("data" in file for file in self.filelist):
            self.isData = True
        else:
            self.isData = False
        self.BLIND = self.isData
