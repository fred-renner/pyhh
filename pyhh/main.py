#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser()
    # TODO make subargs
    # fill args
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batchMode", action="store_true")
    parser.add_argument("--file", type=str, default=None)
    # merge args
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--sample", type=str, default=None)

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fit", action="store_true")

    args = parser.parse_args()

    if args.fill:
        import histfiller.main

        histfiller.main.run(args)

    if args.merge:
        import tools.merger

        tools.merger.run(args)

    if args.plot:
        import plotter.main

        plotter.main.run()


if __name__ == "__main__":
    main()
