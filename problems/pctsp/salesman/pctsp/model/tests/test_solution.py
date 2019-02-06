# python -m pctsp.model.tests.test_solution
import unittest

from pctsp.model import solution
from pctsp.model import pctsp
import numpy as np

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.p = pctsp.Pctsp()
        self.p.prize = np.array([0, 4, 8, 3])
        self.p.penal = np.array([1000, 7, 11, 17])
        self.p.cost = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    def test_quality(self):
        s = solution.Solution(self.p)
        s.route = [0, 1, 2, 3]
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 4)

    def test_quality_2(self):
        s = solution.Solution(self.p, size=2)
        s.route = [0, 1, 2, 3]
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 30)

    def test_swap(self):
        s = solution.Solution(self.p, size=3)
        s.route = [0, 1, 2, 3]
        
        s.swap(1,3)
        print("Quality: ", s.quality)
        print("route:", s.route)
        self.assertEqual(s.quality, 10)

    def test_add_city(self):
        s = solution.Solution(self.p, size=3)
        s.route = [0, 1, 2, 3]
        
        s.add_city()
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 4)

    def test_remove_city(self):
        s = solution.Solution(self.p)
        s.route = [0, 1, 2, 3]

        s.remove_city(3)
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 20)

    def test_remove_cities(self):
        s = solution.Solution(self.p)
        s.route = [0, 1, 2, 3]

        s.remove_cities(quant=3)
        self.assertEqual(s.quality, 35)

if __name__ == '__main__':
    unittest.main()
