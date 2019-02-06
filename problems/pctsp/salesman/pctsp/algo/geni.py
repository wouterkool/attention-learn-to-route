# module geni.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
geni module - Auxiliary functions to the GENI method.

"""
__version__="1.0"

import numpy as np
import sys

def geni(v, s, max_i):
    quality_1 = 0
    quality_2 = 0

    s_star = Solution()
    s_start.quality = sys.maxint

    for i in range(1, max_i):
        quality_1 = quality_after_insertion_1(v, i, )
        quality_2 = quality_after_insertion_2()

        if quality_1 < quality_2 and quality_1 < s_star.quality:
            s_star = insertion_1(s)
        elif quality_2 < quality_1 and quality_2 < s_star.quality:
            s_star = insertion_2(s)

    return s_star
