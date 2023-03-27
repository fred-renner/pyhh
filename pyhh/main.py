#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser("pyhh")
    subparsers = parser.add_subparsers(dest="command")
    # select args
    parser_fill = subparsers.add_parser("select", help="run object selection")
    parser_fill.add_argument("--file", type=str, default=None, required=False)
    parser_fill.add_argument("--debug", action="store_true")
    parser_fill.add_argument("--batchMode", action="store_true")
    # merge args
    parser_merge = subparsers.add_parser("merge", help="merge hist files")
    parser_merge.add_argument("--sample", type=str, default=None)
    # # plot args
    parser_plot = subparsers.add_parser("plot", help="run plotting")
    parser_plot.add_argument("--sample", type=str, default=None)
    # # fit args
    parser_fill = subparsers.add_parser("fit", help="run fitting")
    
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
