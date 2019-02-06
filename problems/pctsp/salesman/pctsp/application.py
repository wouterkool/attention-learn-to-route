# module application.py
#
# Copyright (c) 2015 Rafael Reis
#
"""
application module - Main module that solves the Prize Collecting Travelling Salesman Problem

"""

from pctsp.model.pctsp import *
from pctsp.model import solution
from pctsp.algo.genius import genius
from pctsp.algo import ilocal_search as ils
from pkg_resources import resource_filename
import random

INPUT_INSTANCE_FILE = resource_filename('pctsp', 'data/problem_20_100_100_1000.pctsp')

def solve_instance(filename, min_prize, runs=10, seed=1234):
    random.seed(seed)
    pctsp = Pctsp()
    pctsp.load(filename, min_prize)
    s = solution.random(pctsp, size=int(len(pctsp.prize) * 0.7))
    s = ils.ilocal_search(s, n_runs=runs)

    return (s.route[1:], s.quality)

def main():
    """Main function, that solves the PCTSP.

    """
    pctsp = Pctsp()
    pctsp.load(INPUT_INSTANCE_FILE, 386)
    #pctsp.prize = np.array([0, 4, 8, 3])
    #pctsp.penal = np.array([1000, 7, 11, 17])
    #pctsp.cost = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    # print(pctsp.type)

    size = int(len(pctsp.prize)*0.7)

    s = solution.random(pctsp, size=size)
    print(s.route)
    print(s.size)
    print(s.quality)
    print(s.is_valid())

    print("\n")

    # s = genius(pctsp)
    # print(s.route)
    # print(s.quality)

    s = ils.ilocal_search(s)
    print(s.route)
    print(s.size)
    print(s.quality)
    print(s.is_valid())

if __name__ == '__main__':
    main()
