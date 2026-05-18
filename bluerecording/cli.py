import argparse
from pathlib import Path

from . import __version__, positions
from .circuit import init_circuit
from .positions import compute_positions, save_positions
from .weights import DEFAULT_SIGMA, Electrode, get_weights, save_weights


def main():
    parser = argparse.ArgumentParser(prog="bluerecording", description="Bluerecording CLI")

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # write_positions command
    gp_parser = subparsers.add_parser("write_positions", help="Compute and save segment positions to disk")
    gp_parser.add_argument("path_to_simconfig", type=str, help="Path to the simulation or circuit configuration file")
    gp_parser.add_argument(
        "path_to_positions_folder", type=str, help="Path to the folder where positions will be stored"
    )
    gp_parser.add_argument(
        "--no-replace-axons",
        action="store_false",
        dest="replace_axons",
        help="Do not replace existing axons (default: replace)",
    )

    # write_weights command
    ww_parser = subparsers.add_parser("write_weights", help="Compute electrode weights for all cells in the circuit")
    ww_parser.add_argument("path_to_simconfig", type=str, help="Path to the simulation or circuit configuration file")
    ww_parser.add_argument("electrode_csv", type=str, help="Path to the electrode CSV file")
    ww_parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output H5 weights file (e.g. /path/to/weights.h5)",
    )
    ww_parser.add_argument(
        "--no-replace-axons",
        action="store_false",
        dest="replace_axons",
        help="Do not replace existing axons (default: replace)",
    )
    ww_parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=None,
        help=f"Extracellular conductivity in S/m (uses {DEFAULT_SIGMA} if not specified)",
    )
    ww_parser.add_argument(
        "--path-to-fields",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to H5 potential field files for reciprocity electrodes",
    )
    ww_parser.add_argument(
        "--with-neurite-type",
        action="store_true",
        default=False,
        dest="with_neurite_type",
        help="Append a neurite_types dataset to the weights file",
    )
    ww_parser.add_argument(
        "--write-positions",
        action="store_true",
        default=False,
        dest="write_positions",
        help="Also save segment positions alongside the weights file",
    )

    args = parser.parse_args()

    if args.command == "write_positions":
        positions_df, _, _ = compute_positions(args.path_to_simconfig, replace_axons=args.replace_axons)
        save_positions(positions_df, args.path_to_positions_folder)

    elif args.command == "write_weights":
        node_manager, ids, cols, population, population_name, morphologies_dir = init_circuit(args.path_to_simconfig)

        output_file = Path(args.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        electrodes = Electrode.from_csv(args.electrode_csv)

        positions_df, cols, neurite_types = positions.get_positions(
            node_manager,
            ids,
            cols,
            population,
            morphologies_dir=morphologies_dir,
            replace_axons=args.replace_axons,
        )
        weights = get_weights(
            positions_df,
            cols,
            electrodes=electrodes,
            sigma=args.sigma,
            path_to_fields=args.path_to_fields,
        )
        save_weights(
            weights,
            cols,
            population_name,
            str(output_file),
            electrodes=electrodes,
            neurite_types=neurite_types if args.with_neurite_type else None,
        )
        if args.write_positions:
            save_positions(positions_df, output_file.parent)
