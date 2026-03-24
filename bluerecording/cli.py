import argparse
from pathlib import Path
from . import positions
from .circuit import init_circuit
from .writeH5 import DEFAULT_SIGMA
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

    # write_positions command
    gp_parser = subparsers.add_parser(
        "write_positions",
        help="Compute and save segment positions to disk"
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

    # write_weights command
    ww_parser = subparsers.add_parser(
        "write_weights",
        help="Compute electrode weights for all cells in the circuit"
    )
    ww_parser.add_argument(
        "path_to_simconfig",
        type=str,
        help="Path to the simulation configuration file"
    )
    ww_parser.add_argument(
        "electrode_csv",
        type=str,
        help="Path to the electrode CSV file"
    )
    ww_parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output H5 weights file"
    )
    ww_parser.add_argument(
        "--no-replace-axons",
        action="store_false",
        dest="replace_axons",
        help="Do not replace existing axons (default: replace)"
    )
    ww_parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=None,
        help=f"Extracellular conductivity in S/m (default: {DEFAULT_SIGMA})"
    )
    ww_parser.add_argument(
        "--path-to-fields",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to H5 potential field files for reciprocity electrodes"
    )

    args = parser.parse_args()

    if args.command == "write_positions":
        node_manager, ids, cols, population = init_circuit(args.path_to_simconfig)
        positions_df, _ = positions.get_positions(
            node_manager, ids, cols, population,
            path_to_simconfig=args.path_to_simconfig,
            replace_axons=args.replace_axons,
        )
        positions.save_positions(positions_df, args.path_to_positions_folder)

    elif args.command == "write_weights":
        node_manager, ids, cols, population = init_circuit(args.path_to_simconfig)
        positions_df, cols = positions.get_positions(
            node_manager, ids, cols, population,
            path_to_simconfig=args.path_to_simconfig,
            replace_axons=args.replace_axons,
        )
        # TODO: connect to initializeH5File + writeH5File