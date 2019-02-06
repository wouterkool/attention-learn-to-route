# module pctsp.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
pctsp module - Implements Pctsp, a class that describes an instance of the problem..

"""
__version__="1.0"

import numpy as np
import re

class Pctsp(object):
    """
    Attributes:
       c (:obj:`list` of :obj:`list`): Costs from i to j
       p (:obj:`list` of :obj:`int`): Prize for visiting each city i
       gama (:obj:`list` of :obj:`int`): Penalty for not visiting each city i
    """
    def __init__(self):
        self.prize = []
        self.penal = []
        self.cost = []
        self.prize_min = 0

    def load(self, file_name, prize_min):

        f = open(file_name,'r')
        for i,line in enumerate(f):
            if i is 5: break
            if i is 1: self.prize = np.fromstring(line, dtype=int, sep=' ')
            if i is 4: self.penal = np.fromstring(line, dtype=int, sep=' ')

        f.close()

        self.cost = np.loadtxt(file_name, dtype=int, skiprows=7)
        self.prize_min = prize_min

        assert sum(self.prize) >= prize_min, "Infeasible"


