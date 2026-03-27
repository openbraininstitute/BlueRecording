import argparse
from pathlib import Path
from . import positions
from .circuit import init_circuit
from .writeH5 import DEFAULT_SIGMA, writeH5File
from .writeH5_prelim import initializeH5File
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
        "output_path",
        type=str,
        help="Path to the output H5 weights file, or a directory (weights.h5 will be created inside)"
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
        node_manager, ids, cols, population, _ = init_circuit(args.path_to_simconfig)
        positions_df, _ = positions.get_positions(
            node_manager, ids, cols, population,
            path_to_simconfig=args.path_to_simconfig,
            replace_axons=args.replace_axons,
        )
        positions.save_positions(positions_df, args.path_to_positions_folder)

    elif args.command == "write_weights":
        node_manager, ids, cols, population, population_name = init_circuit(args.path_to_simconfig)
        positions_df, cols = positions.get_positions(
            node_manager, ids, cols, population,
            path_to_simconfig=args.path_to_simconfig,
            replace_axons=args.replace_axons,
        )
        output_file = Path(args.output_path)
        if output_file.is_dir() or not output_file.suffix:
            output_file.mkdir(parents=True, exist_ok=True)
            output_file = output_file / "weights.h5"
        initializeH5File(cols, population_name, str(output_file), args.electrode_csv)
        writeH5File(positions_df, cols, population_name, str(output_file),
                    sigma=args.sigma, path_to_fields=args.path_to_fields)