#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

from gurobipy import *


def solve_euclidian_pctsp(depot, loc, penalty, prize, min_prize, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan pctsp problem to pctsptimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    points = [depot] + loc
    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if tour is not None:
                # add subtour elimination constraint for every pair of cities in tour
                # model.cbLazy(quicksum(model._vars[i, j]
                #                       for i, j in itertools.combinations(tour, 2))
                #              <= len(tour) - 1)

                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= quicksum(model._dvars[i] for i in tour) * (len(tour) - 1) / float(len(tour)))

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges, exclude_depot=True):
        unvisited = list(range(n))
        #cycle = range(n + 1)  # initial length has 1 more city
        cycle = None
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            # If we do not yet have a cycle or this is the shorter cycle, keep this cycle
            # Unless it contains the depot while we do not want the depot
            if (
                (cycle is None or len(cycle) > len(thiscycle))
                    and len(thiscycle) > 1 and not (0 in thiscycle and exclude_depot)
            ):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in pctspposite direction

    # Depot vars can be 2
    for i,j in vars.keys():
        if i == 0 or j == 0:
            vars[i,j].vtype = GRB.INTEGER
            vars[i,j].ub = 2

    penalty_dict = {
        i + 1: -p  # We get penalties for the nodes not visited so we 'save the penalties' for the nodes visited
        for i, p in enumerate(penalty)
    }
    delta = m.addVars(range(1, n), obj=penalty_dict, vtype=GRB.BINARY, name='delta')

    # Add degree-2 constraint (2 * delta for nodes which are not the depot)
    m.addConstrs(vars.sum(i,'*') == (2 if i == 0 else 2 * delta[i]) for i in range(n))


    # Minimum prize constraint
    assert min_prize <= sum(prize)
    # Subtract 1 from i since prizes are 0 indexed while delta vars start with 1 (0 is depot)
    m.addConstr(quicksum(var * prize[i - 1] for i, var in delta.items()) >= min_prize)

    # optimize model

    m._vars = vars
    m._dvars = delta
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    # For the correct objective, we need to add the sum of the penalties, which are subtracted when nodes are visited
    # this is important for the relative gap termination criterion
    m.objcon = sum(penalty)
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected, exclude_depot=False)
    assert tour[0] == 0, "Tour should start with depot"

    return m.objVal, tour