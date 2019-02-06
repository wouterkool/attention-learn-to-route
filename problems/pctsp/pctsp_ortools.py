#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
# Copyright 2015 Tin Arm Engineering AB
# Copyright 2018 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capacitated Vehicle Routing Problem (CVRP).

   This is a sample using the routing library python wrapper to solve a CVRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

from __future__ import print_function
from collections import namedtuple
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import math

###########################
# Problem Data Definition #
###########################
# Vehicle declaration
Vehicle = namedtuple('Vehicle', ['capacity'])


def float_to_scaled_int(v):
    return int(v * 10000000 + 0.5)


class DataProblem():
  """Stores the data for the problem"""

  def __init__(self, depot, loc, prize, penalty, min_prize):
    """Initializes the data for the problem"""
    # Locations in block unit
    self._locations = [(float_to_scaled_int(l[0]), float_to_scaled_int(l[1])) for l in [depot] + loc]

    self._prizes = [float_to_scaled_int(v) for v in prize]

    self._penalties = [float_to_scaled_int(v) for v in penalty]

    # Check that min_prize is feasible
    assert sum(prize) >= min_prize
    # After scaling and rounding, however, it can possible not be feasible so relax constraint
    self._min_prize = min(float_to_scaled_int(min_prize), sum(self.prizes))

  @property
  def vehicle(self):
    """Gets a vehicle"""
    return Vehicle()

  @property
  def num_vehicles(self):
    """Gets number of vehicles"""
    return 1

  @property
  def locations(self):
    """Gets locations"""
    return self._locations

  @property
  def num_locations(self):
    """Gets number of locations"""
    return len(self.locations)

  @property
  def depot(self):
    """Gets depot location index"""
    return 0

  @property
  def prizes(self):
    """Gets prizes at each location"""
    return self._prizes

  @property
  def penalties(self):
      """Gets penalties at each location"""
      return self._penalties

  @property
  def min_prize(self):
      """Gets penalties at each location"""
      return self._min_prize


#######################
# Problem Constraints #
#######################
def euclidian_distance(position_1, position_2):
  """Computes the Euclidian distance between two points"""
  return int(math.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2) + 0.5)


class CreateDistanceEvaluator(object):  # pylint: disable=too-few-public-methods
  """Creates callback to return distance between points."""

  def __init__(self, data):
    """Initializes the distance matrix."""
    self._distances = {}

    # precompute distance between location to have distance callback in O(1)
    for from_node in xrange(data.num_locations):
      self._distances[from_node] = {}
      for to_node in xrange(data.num_locations):
        if from_node == to_node:
          self._distances[from_node][to_node] = 0
        else:
          self._distances[from_node][to_node] = (
              euclidian_distance(data.locations[from_node],
                                 data.locations[to_node]))

  def distance_evaluator(self, from_node, to_node):
    """Returns the manhattan distance between the two nodes"""
    return self._distances[from_node][to_node]


class CreatePrizeEvaluator(object):  # pylint: disable=too-few-public-methods
  """Creates callback to get prizes at each location."""

  def __init__(self, data):
    """Initializes the prize array."""
    self._prizes = data.prizes

  def prize_evaluator(self, from_node, to_node):
    """Returns the prize of the current node"""
    del to_node
    return 0 if from_node == 0 else self._prizes[from_node - 1]


def add_min_prize_constraints(routing, data, prize_evaluator, min_prize):
  """Adds capacity constraint"""
  prize = 'Prize'
  routing.AddDimension(
      prize_evaluator,
      0,  # null capacity slack
      sum(data.prizes),  # No upper bound
      True,  # start cumul to zero
      prize)
  capacity_dimension = routing.GetDimensionOrDie(prize)
  for vehicle in xrange(data.num_vehicles):  # only single vehicle
      capacity_dimension.CumulVar(routing.End(vehicle)).RemoveInterval(0, min_prize)


def add_distance_constraint(routing, distance_evaluator, maximum_distance):
    """Add Global Span constraint"""
    distance = "Distance"
    routing.AddDimension(
        distance_evaluator,
        0, # null slack
        maximum_distance, # maximum distance per vehicle
        True, # start cumul to zero
        distance)


###########
# Printer #
###########
def print_solution(data, routing, assignment):
  """Prints assignment on console"""
  print('Objective: {}'.format(assignment.ObjectiveValue()))
  total_distance = 0
  total_load = 0
  capacity_dimension = routing.GetDimensionOrDie('Capacity')
  for vehicle_id in xrange(data.num_vehicles):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
    distance = 0
    while not routing.IsEnd(index):
      load_var = capacity_dimension.CumulVar(index)
      plan_output += ' {} Load({}) -> '.format(
          routing.IndexToNode(index), assignment.Value(load_var))
      previous_index = index
      index = assignment.Value(routing.NextVar(index))
      distance += routing.GetArcCostForVehicle(previous_index, index,
                                               vehicle_id)
    load_var = capacity_dimension.CumulVar(index)
    plan_output += ' {0} Load({1})\n'.format(
        routing.IndexToNode(index), assignment.Value(load_var))
    plan_output += 'Distance of the route: {}m\n'.format(distance)
    plan_output += 'Load of the route: {}\n'.format(assignment.Value(load_var))
    print(plan_output)
    total_distance += distance
    total_load += assignment.Value(load_var)
  print('Total Distance of all routes: {}m'.format(total_distance))
  print('Total Load of all routes: {}'.format(total_load))


def solve_pctsp_ortools(depot, loc, prize, penalty, min_prize, sec_local_search=0):
    data = DataProblem(depot, loc, prize, penalty, min_prize)

    # Create Routing Model
    routing = pywrapcp.RoutingModel(data.num_locations, data.num_vehicles,
                                    data.depot)

    # Define weight of each edge
    distance_evaluator = CreateDistanceEvaluator(data).distance_evaluator
    routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator)

    # Add minimum total prize constraint
    prize_evaluator = CreatePrizeEvaluator(data).prize_evaluator
    add_min_prize_constraints(routing, data, prize_evaluator, data.min_prize)

    # Add penalties for missed nodes
    nodes = [routing.AddDisjunction([int(c + 1)], p) for c, p in enumerate(data.penalties)]

    # Setting first solution heuristic (cheapest addition).
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    if sec_local_search > 0:
        # Additionally do local search
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit_ms = 1000 * sec_local_search
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    assert assignment is not None, "ORTools was unable to find a feasible solution"

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node_index = routing.IndexToNode(index)
        route.append(node_index)
        index = assignment.Value(routing.NextVar(index))
    return assignment.ObjectiveValue() / 10000000., route
