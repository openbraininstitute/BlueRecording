# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from bluerecording.get_positions import get_positions

if __name__=='__main__':
    path_to_simconfig = sys.argv[1]  # simulation config
    path_to_positions_folder = sys.argv[2]  # positions folder

    replace_axons = True  # default
    if len(sys.argv) > 3:  # optional third argument
        replace_axons = sys.argv[3].lower() in ('true', '1', 'yes')

    get_positions(path_to_simconfig=path_to_simconfig, path_to_positions_folder=path_to_positions_folder, replace_axons=replace_axons)
