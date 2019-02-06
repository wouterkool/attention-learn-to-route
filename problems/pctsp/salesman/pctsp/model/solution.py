# module solution.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
solution module - Implements Solution, a class that describes a solution for the problem.

"""
__version__="1.0"

import numpy as np
import copy
import sys
from random import shuffle

def random(pctsp, start_size):
    s = Solution(pctsp)
    length = len(pctsp.prize)

    # Modification: start from start_size but increase after maximum number of iterations in case no feasible solution
    # is found. When the full length is used, there should always be a feasible solution
    for size in range(start_size, length + 1):
        if size: s.size = size

        i = 0
        min_solutions = 30
        max_solutions = 1000

        while i < min_solutions or (i < max_solutions and not s.is_valid()):
            r = Solution(pctsp)
            if size: r.size = size
            cities = list(range(1, length, 1))
            shuffle(cities) # Shuffle in place
            r.route = [0] + cities # The city 0 is always the first

            if r.quality < s.quality and r.is_valid():
                s = r

            i += 1
        if s.is_valid():
            break
    assert s.is_valid()
    return s


class Solution(object):
    """
    Attributes:
       route (:obj:`list` of :obj:`int`): The list of cities in the visiting order
       size (:obj:`int`): The quantity of the first cities to be considered in the route list
       quality (:obj:`int`): The quality of the solution
    """

    def __init__(self, pctsp, size=None):
        self._route = []
        
        if size:
            self.size = size
        else:
            self.size = len(pctsp.prize) # Default size value is the total of cities
        
        self.quality = sys.maxsize
        self.pctsp = pctsp
        self.prize = 0

    """
    Computes the quality of the solution.
    """
    def compute(self):
        self.prize = 0
        self.quality = 0

        for i,city in enumerate(self._route):
            if i < self.size:
                self.prize += self.pctsp.prize[city]
                if i > 0:
                    previousCity = self._route[i - 1]
                    self.quality += self.pctsp.cost[previousCity][city]
                if i + 1 == self.size:
                    self.quality += self.pctsp.cost[city][0]
            else:
                self.quality += self.pctsp.penal[city]

    def copy(self):
        cp = copy.copy(self)
        cp._route = list(self._route)

        return cp
    
    def swap(self, i, j):
        city_i = self._route[i]
        city_i_prev = self._route[i-1]
        city_i_next = self._route[(i+1) % self.size]
        
        city_j = self._route[j]

        self.quality = (self.quality
                - self.pctsp.cost[city_i_prev][city_i] - self.pctsp.cost[city_i][city_i_next]
                + self.pctsp.cost[city_i_prev][city_j] + self.pctsp.cost[city_j][city_i_next]
                - self.pctsp.penal[city_j] + self.pctsp.penal[city_i])
        self.prize = self.prize - self.pctsp.prize[city_i] + self.pctsp.prize[city_j]

        self._route[j], self._route[i] = self._route[i], self._route[j]

    def is_valid(self):
        return self.prize >= self.pctsp.prize_min

    def add_city(self):
        city_l = self._route[self.size - 1]
        city_add = self._route[self.size]
        
        self.quality = (self.quality
            - self.pctsp.cost[city_l][0]
            - self.pctsp.penal[city_add]
            + self.pctsp.cost[city_l][city_add]
            + self.pctsp.cost[city_add][0])
        
        self.size += 1
        self.prize += self.pctsp.prize[city_add]

    def remove_city(self, index):
        city_rem = self._route[index]
        city_rem_prev = self._route[index-1]
        city_rem_next = self._route[(index+1)%self.size]

        self.quality = (self.quality
            - self.pctsp.cost[city_rem_prev][city_rem] - self.pctsp.cost[city_rem][city_rem_next]
            + self.pctsp.penal[city_rem]
            + self.pctsp.cost[city_rem_prev][city_rem_next])
        self.prize -= self.pctsp.prize[city_rem]

        del self._route[index]        
        self._route.append(city_rem)

        self.size -= 1

    def remove_cities(self, quant):
        for i in range(self.size-quant,self.size):
            city_rem = self._route[i]
            city_rem_prev = self._route[i-1]

            self.quality = (self.quality 
                - self.pctsp.cost[city_rem_prev][city_rem]
                + self.pctsp.penal[city_rem])
            self.prize -= self.pctsp.prize[city_rem]

        city_rem = self._route[self.size-1]
        city_l = self._route[self.size-quant-1]
        self.quality = (self.quality - self.pctsp.cost[city_rem][0]
            + self.pctsp.cost[city_l][0])

        self.size -= quant

    def print_route(self):
        print(self._route)

    @property
    def route(self):
        return self._route

    @route.setter
    def route(self, r):
        self._route = r
        self.compute()
