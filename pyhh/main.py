#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser("pyhh")
    subparsers = parser.add_subparsers(dest="command")
    # select args
    parser_select = subparsers.add_parser("select", help="run object selection")
    parser_select.add_argument("--file", type=str, default=None)
    parser_select.add_argument("--fill", action="store_true")
    parser_select.add_argument("--dump", action="store_true")
    parser_select.add_argument("--debug", action="store_true")
    parser_select.add_argument("--batchMode", action="store_true")
    # merge args
    parser_merge = subparsers.add_parser("merge", help="merge files of same logical dataset")
    # parser_merge.add_argument("--fill", action="store_true")
    # parser_merge.add_argument("--fill", type=str, default=None)
    parser_merge = subparsers.add_parser("merge", help="merge files of same logical dataset")
    parser_merge.add_argument("--sample", type=str, default=None)
    # # plot args
    parser_plot = subparsers.add_parser("plot", help="run plotting")
    parser_plot.add_argument("--sample", type=str, default=None)
    # # fit args
    parser_fit = subparsers.add_parser("fit", help="run fitting")

    args = parser.parse_args()

    if args.command == "select":
        import histfiller.main

        histfiller.main.run(args)

    if args.command == "merge":
        import tools.merger

        tools.merger.run(args)

    if args.command == "plot":
        import plotter.main

        plotter.main.run()


if __name__ == "__main__":
    main()
