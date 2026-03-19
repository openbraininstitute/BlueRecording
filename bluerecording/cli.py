import argparse
from pathlib import Path
from . import getPositions
from . import __version__

def main():
    parser = argparse.ArgumentParser(
        prog="bluerecording",
        description="Bluerecording CLI"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # get_positions command
    gp_parser = subparsers.add_parser(
        "get_positions",
        help="Retrieve positions from the system"
    )
    gp_parser.add_argument(
        "path_to_simconfig",
        type=str,
        help="Path to the simulation configuration file"
    )
    gp_parser.add_argument(
        "path_to_positions_folder",
        type=str,
        help="Path to the folder where positions will be stored"
    )
    gp_parser.add_argument(
        "--no-replace-axons",
        action="store_false",
        dest="replace_axons",
        help="Do not replace existing axons (default: replace)"
    )

    args = parser.parse_args()

    if args.command == "get_positions":
        getPositions.getPositions(
            path_to_simconfig=args.path_to_simconfig,
            path_to_positions_folder=args.path_to_positions_folder,
            replace_axons=args.replace_axons
        )