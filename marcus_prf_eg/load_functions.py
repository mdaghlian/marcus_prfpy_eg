import os
opj = os.path.join
import numpy as np
from .utils import *

path_to_code = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

path_saved = opj(path_to_code, 'saved_data')
