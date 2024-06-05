import numpy as np

from airfoil_utils import generate_airfoils_singlefile

# Generate a single file of all original airfoils and corresponding L by D ratios
airfoil_set = 'train'
generate_airfoils_singlefile(airfoil_set)