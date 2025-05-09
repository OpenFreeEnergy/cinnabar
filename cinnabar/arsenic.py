"""
cinnabar.arsenic
================

Containing the command line interface usage of cinnabar.
"""
import argparse

from . import FEMap, plotting


def main():

    parser = argparse.ArgumentParser(description="Get input")
    parser.add_argument("csv", type=str, help="Path to the results csv file")
    parser.add_argument(
        "--plot",
        default=["all"],
        choices=["all", "ddg", "dg", "all ddg", "network"],
        nargs="+",
        help="Which plots to generate",
    )
    parser.add_argument("--stats", default="all", help="Which statistics to generate")
    parser.add_argument("--plotly", action="store_true")
    parser.add_argument(
        "--method", type=str, default="", help="Name of the method, used for labelling"
    )
    parser.add_argument(
        "--target", type=str, default="", help="Name of the target, used for labelling"
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix for figure filenames generated"
    )
    parser.add_argument(
        "--title", type=str, default="", help="Title for plots generated"
    )

    args = parser.parse_args()

    if args.title == "":
        args.title = f"{args.method} {args.target}"

    if args.plot == ["all"]:
        args.plot = ["ddg", "dg", "all ddg", "network"]

    network = FEMap.from_csv(args.csv)
    # this generates the three plots that we need
    if "network" in args.plot:
        network.draw_graph(title=args.title, filename=f"{args.prefix}network.png")
    if "ddg" in args.plot:
        plotting.plot_DDGs(
            network.to_legacy_graph(), title=args.title, filename=f"{args.prefix}DDGs.png"
        )
    if "dg" in args.plot:
        plotting.plot_DGs(
            network.to_legacy_graph(), title=args.title, filename=f"{args.prefix}DGs.png"
        )
    if "all ddg" in args.plot:
        plotting.plot_all_DDGs(
            network.to_legacy_graph(), title=args.title, filename=f"{args.prefix}all_DDGs.png"
        )


if __name__ == "__main__":
    main()
