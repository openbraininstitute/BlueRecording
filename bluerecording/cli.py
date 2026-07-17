import argparse
import sys
from pathlib import Path

from . import __version__, positions
from .circuit import init_circuit
from .compare import compare_weights
from .positions import compute_positions, save_positions
from .weights import DEFAULT_ELECTRODE_CHUNK_SIZE, DEFAULT_SIGMA, Electrode, get_weights, save_weights


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
    ww_parser.add_argument("electrode_file", type=str, help="Path to the electrode JSON file")
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
    ww_parser.add_argument(
        "--electrode-chunk-size",
        type=int,
        default=DEFAULT_ELECTRODE_CHUNK_SIZE,
        dest="electrode_chunk_size",
        help=f"Max electrodes per computation chunk (default: {DEFAULT_ELECTRODE_CHUNK_SIZE})",
    )

    # compare_weights command
    cw_parser = subparsers.add_parser("compare_weights", help="Compare two weights H5 files (order-agnostic)")
    cw_parser.add_argument("reference", type=str, help="Path to the reference weights H5 file")
    cw_parser.add_argument("target", type=str, help="Path to the target weights H5 file")
    cw_parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance (default: 1e-6)")
    cw_parser.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance (default: 1e-9)")
    cw_parser.add_argument("--population", type=str, default=None, help="Population name (auto-detected if omitted)")

    args = parser.parse_args()

    if args.command == "write_positions":
        positions_df, _, _ = compute_positions(args.path_to_simconfig, replace_axons=args.replace_axons)
        save_positions(positions_df, args.path_to_positions_folder)

    elif args.command == "write_weights":
        cells, cols, population, population_name, morphologies_dir = init_circuit(args.path_to_simconfig)

        output_file = Path(args.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        electrodes = Electrode.from_json(args.electrode_file)

        positions_df, cols, neurite_types = positions.get_positions(
            cells,
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
            electrode_chunk_size=args.electrode_chunk_size,
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

    elif args.command == "compare_weights":
        match, report = compare_weights(
            args.reference,
            args.target,
            rtol=args.rtol,
            atol=args.atol,
            population_name=args.population,
        )
        print(report)
        sys.exit(0 if match else 1)
