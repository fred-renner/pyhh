#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser("pyhh")
    subparsers = parser.add_subparsers(dest="command")
    # select args
    parser_select = subparsers.add_parser("select", help="run object selection")
    parser_select.add_argument("--file", type=str, required=True)
    parser_select.add_argument("--fill", action="store_true", default=False)
    parser_select.add_argument("--dump", action="store_true", default=False)
    parser_select.add_argument("--debug", action="store_true", default=False)
    parser_select.add_argument("--batchMode", action="store_true", default=False)
    # submit all
    parser_submit = subparsers.add_parser(
        "make-submit", help="make HTCondor submit file"
    )
    parser_submit.add_argument("--sample", type=str, default=None, required=True)

    # merge args
    parser_merge = subparsers.add_parser(
        "merge", help="merge files of same logical dataset"
    )
    parser_merge.add_argument("--sample", type=str, default=None, required=True)
    parser_merge.add_argument("--hists", action="store_true")
    parser_merge.add_argument("--dumped", action="store_true")
    # # plot args
    parser_plot = subparsers.add_parser("plot", help="run plotting")
    parser_plot.add_argument("--sample", type=str, default=None)
    # # fit args
    parser_fit = subparsers.add_parser("fit", help="run fitting")

    args = parser.parse_args()

    if args.command == "select":
        import selector.main

        selector.main.run(args)

    if args.command == "make-submit":
        import scripts.make_histfill_sub

        scripts.make_histfill_sub.run(args)

    if args.command == "merge":
        import tools.merger

        tools.merger.run(args)

    if args.command == "plot":
        import plotter.main

        plotter.main.run()

    if args.command == "fit":
        import fitter.main

        fitter.main.run()


if __name__ == "__main__":
    main()
