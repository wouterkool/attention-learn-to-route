# System Design Document

## Mixed-Fleet Hub-and-Spoke Transit Optimization

### 1. Purpose

This document defines the system architecture for the project novelty: a **day-ahead mixed-fleet hub-and-spoke transit optimizer** for shared mobility operations.

The goal is to plan an entire service day in advance, rather than re-optimize in real time. The system therefore computes:

- hub assignments
- vehicle type assignments
- stop ordering within each route
- transfer feasibility
- service schedules
- utilization and cost metrics

The design is motivated by the project PDF:
[Flexible Network Strategy Mixed Fleet Hub-and-Spoke Transit Optimization.pdf](/home/ygn/Documents/RV Univ/Simulation/Flexible%20Network%20Strategy%20Mixed%20Fleet%20Hub-and-Spoke%20Transit%20Optimization.pdf)

### 2. Scope and Non-Scope

#### 2.1 In Scope

- Day-ahead planning for a full operating window
- Mixed fleet operations
  - feeder vehicles
  - trunk vehicles
- Hub-and-spoke routing
- Passenger aggregation into hubs
- Route ordering inside each service cluster
- Transfer validation
- Capacity-aware assignment
- Utilization-aware planning
- Cost and service-quality evaluation

#### 2.2 Out of Scope

- Real-time dynamic re-optimization
- Midday rerouting in response to live disruptions
- Online dispatch after the plan is frozen
- Multi-day learning of operational history

The system assumes all demand for the planning horizon is known before optimization begins.

### 3. Problem Statement

Given:

- a set of passenger requests
- a set of vehicle types with different capacities
- a set of candidate hubs
- planning constraints such as walking radius, transfer limits, and vehicle capacity

compute a service plan that assigns each request to a feasible route chain and vehicle type such that:

- total fleet miles are minimized
- vehicle utilization is maximized
- transfer feasibility is preserved
- passenger travel burden remains acceptable
- the full day-ahead schedule is operationally executable

This is a hierarchical transit optimization problem, not a pure shortest-tour problem.

### 4. Core Assumptions

- Demand is known in advance for the planning horizon.
- Requests are grouped by time window before optimization.
- Hubs are fixed or drawn from a limited candidate set.
- Vehicle fleets are heterogeneous.
- Feasibility must hold before the schedule is published.
- The objective is a balanced operational plan, not only minimum route distance.

### 5. System Overview

The system is organized into five layers:

1. **Demand ingestion layer**
   - reads passenger requests and vehicle inventory
2. **Hub selection layer**
   - identifies candidate hubs and demand clusters
3. **Fleet assignment layer**
   - assigns feeder and trunk vehicles
4. **Route construction layer**
   - orders stops within each cluster or route
5. **Schedule validation layer**
   - checks capacity, transfer, and time-window feasibility

The baseline attention model is used only in the route construction layer.

### 6. Data Model

#### 6.1 Passenger Request

Each request should contain:

- `request_id`
- `passenger_count`
- `origin_lat`
- `origin_lon`
- `destination_lat`
- `destination_lon`
- `request_time`
- `time_window_start`
- `time_window_end`
- `priority`
- `service_type`

#### 6.2 Vehicle Record

Each vehicle should contain:

- `vehicle_id`
- `vehicle_type`
- `capacity`
- `depot_id`
- `start_time`
- `end_time`
- `availability`
- `fixed_cost`
- `per_km_cost`

#### 6.3 Hub Record

Each hub should contain:

- `hub_id`
- `latitude`
- `longitude`
- `service_radius`
- `max_transfer_wait`
- `hub_capacity`
- `hub_type`

#### 6.4 Route Record

Each route should contain:

- `route_id`
- `assigned_vehicle_id`
- `route_type`
- `hub_sequence`
- `stop_sequence`
- `departure_time`
- `arrival_time`
- `load_profile`
- `feasibility_status`

### 7. Objective Function

The optimization target should combine several costs.

#### 7.1 Primary Terms

- total vehicle distance
- total operating time
- unserved demand penalty
- transfer penalty
- capacity violation penalty

#### 7.2 Secondary Terms

- low utilization penalty
- excessive walking penalty
- late arrival penalty
- route fragmentation penalty

#### 7.3 Example Objective

```text
Loss =
  α * fleet_distance
  + β * operating_time
  + γ * transfer_penalty
  + δ * walking_penalty
  + ε * infeasibility_penalty
  + ζ * low_utilization_penalty
```

The weights should be tuned empirically and documented in experiments.

### 8. Architectural Modules

#### 8.1 Demand Aggregation Module

Purpose:
- group requests into spatial and temporal clusters
- estimate local demand density
- detect high-traffic corridors

Possible methods:
- k-means clustering
- DBSCAN
- grid-based density aggregation
- learned clustering score

Outputs:
- request clusters
- cluster centroids
- demand volumes
- candidate hub affinities

#### 8.2 Hub Selection Module

Purpose:
- choose transfer hubs that minimize access cost and maximize consolidation benefit

Hub selection criteria:
- proximity to clustered demand
- road accessibility
- corridor alignment
- transfer feasibility
- capacity for passenger exchange

Outputs:
- selected hubs
- cluster-to-hub assignments
- transfer eligibility map

#### 8.3 Fleet Assignment Module

Purpose:
- assign feeder and trunk vehicles to each route segment
- ensure vehicle capacity matches expected load

Rules:
- use feeder vehicles for local pickup and drop-off
- use trunk vehicles for long-haul corridor service
- avoid over-allocating large vehicles to low-density areas

Outputs:
- vehicle-to-route assignments
- route type labels
- capacity feasibility status

#### 8.4 Route Construction Module

Purpose:
- order pickup and drop-off stops within each route

This is where the current TSP attention model can be reused as a local decoder, but the current
default route-construction strategy should be the transit-specific learned orderer trained on
synthetic day-ahead instances. The TSP attention decoder remains available as a fallback reference.

Inputs:
- cluster members
- assigned vehicle type
- hub locations
- capacity constraints

Outputs:
- stop sequence
- local route distance
- route-level ordering score

#### 8.5 Schedule Validation Module

Purpose:
- verify that the full day-ahead plan is executable

Checks:
- all requests assigned
- vehicle capacity respected
- transfer limits satisfied
- route start/end times valid
- hub capacity not exceeded

Outputs:
- feasible / infeasible flags
- violation diagnostics
- final schedule score

### 9. How the Baseline Fits

The published TSP attention model is not the final solution.

It is used as:

- a local routing solver for route ordering
- a quality benchmark for stop-sequence construction
- a learnable submodule inside the larger transit pipeline

It is not used for:

- hub discovery
- fleet assignment
- schedule validation
- global transit planning

### 10. Proposed Full Pipeline

#### Step 1: Read demand and fleet data
- ingest all requests for the day
- ingest available feeder and trunk vehicles

#### Step 2: Aggregate demand
- cluster requests by location and time
- estimate where hubs should be used

#### Step 3: Assign hubs
- connect clusters to one or more hubs
- check walking and transfer feasibility

#### Step 4: Assign vehicles
- decide whether each segment uses a feeder or trunk vehicle
- enforce capacity and operating constraints

#### Step 5: Construct routes
- use the routing decoder to order stops inside each segment

#### Step 6: Validate the plan
- ensure the resulting day-ahead schedule is feasible
- compute objective metrics

### 11. Baseline System

The baseline should be a simpler non-learned or lightly learned pipeline:

- direct point-to-point routing
- heuristic hub assignment
- fixed vehicle allocation rules
- greedy or OR-Tools route ordering

This baseline will be used to show the benefit of the full mixed-fleet design.

### 12. Novelty Statement

The novelty of the project is the transition from a standard point-to-point routing formulation to a **day-ahead hierarchical transit optimizer** that jointly handles:

- hub selection
- mixed-fleet assignment
- route ordering
- transfer feasibility
- operational utilization

The current attention-based TSP model is retained as the local route-ordering component, but the scientific contribution is the larger transit planning framework.

### 13. Evaluation Plan

The final system should be evaluated on:

- total fleet miles
- average vehicle utilization
- on-time rate
- average walking distance
- number of transfers
- infeasibility rate
- runtime for day-ahead planning

Benchmark comparisons:

- direct point-to-point routing
- hub-and-spoke heuristic
- mixed-fleet baseline
- proposed hierarchical optimizer

### 14. Implementation Roadmap

#### Phase 1
- define data schema
- implement demand clustering
- implement fixed hub assignment heuristic

#### Phase 2
- implement mixed-fleet vehicle assignment
- integrate local route ordering

#### Phase 3
- connect all modules into a single planning pipeline
- validate day-ahead schedules

#### Phase 4
- benchmark against baseline methods
- tune objective weights

### 15. Notes for the Next Build Step

The next development task should be to implement the system in this order:

1. demand clustering and hub proposals
2. mixed-fleet assignment logic
3. route ordering using the existing attention model
4. feasibility validation
5. reporting and metrics
