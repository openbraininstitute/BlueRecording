import argparse
from pathlib import Path
from . import positions
from .circuit import init_circuit
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
        node_manager, ids, cols, population = init_circuit(args.path_to_simconfig)
        positions_df, _ = positions.get_positions(
            node_manager, ids, cols, population,
            path_to_simconfig=args.path_to_simconfig,
            replace_axons=args.replace_axons,
        )
        positions.save_positions(positions_df, args.path_to_positions_folder)