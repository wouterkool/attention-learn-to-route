# module genius.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
genius module - Implements GENIUS, an algorithm for generation of a solution.

"""
__version__="1.0"

from pctsp.model.pctsp import *
from pctsp.model import solution

import numpy as np

def genius(pctsp):
    s = solution.random(pctsp, size=3)
    s = geni(pstsp, s)
    s = us(pctsp, s)

    return s

def geni(pctsp, s):
    return

def us(pctsp, s):
    return
