# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from bluerecording.circuit import init_circuit
from bluerecording.positions import get_positions, save_positions

if __name__=='__main__':
    path_to_simconfig = sys.argv[1]  # simulation config
    path_to_positions_folder = sys.argv[2]  # positions folder

    replace_axons = True  # default
    if len(sys.argv) > 3:  # optional third argument
        replace_axons = sys.argv[3].lower() in ('true', '1', 'yes')

    node_manager, ids, cols, population, _ = init_circuit(path_to_simconfig)
    positions_df, _ = get_positions(node_manager, ids, cols, population,
                                    path_to_simconfig=path_to_simconfig,
                                    replace_axons=replace_axons)
    save_positions(positions_df, path_to_positions_folder)
