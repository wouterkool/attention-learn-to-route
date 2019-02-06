import argparse
import os
import numpy as np
from utils import run_all_in_pool
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output
import re
import time
from datetime import timedelta
import random
from scipy.spatial import distance_matrix
from .salesman.pctsp.model.pctsp import Pctsp
from .salesman.pctsp.algo.ilocal_search import ilocal_search
from .salesman.pctsp.model import solution

MAX_LENGTH_TOL = 1e-5


def get_pctsp_executable():
    path = os.path.join("pctsp", "PCTSP", "PCPTSP")
    sourcefile = os.path.join(path, "main.cpp")
    execfile = os.path.join(path, "main.out")
    if not os.path.isfile(execfile):
        print ("Compiling...")
        check_call(["g++", "-g", "-Wall", sourcefile, "-std=c++11", "-o", execfile])
        print ("Done!")
    assert os.path.isfile(execfile), "{} does not exist! Compilation failed?".format(execfile)
    return os.path.abspath(execfile)


def solve_pctsp_log(executable, directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize, runs=10):

    problem_filename = os.path.join(directory, "{}.pctsp{}.pctsp".format(name, runs))
    output_filename = os.path.join(directory, "{}.pctsp{}.pkl".format(name, runs))
    log_filename = os.path.join(directory, "{}.pctsp{}.log".format(name, runs))

    try:
        # May have already been run
        if not os.path.isfile(output_filename):
            write_pctsp(problem_filename, depot, loc, penalty, deterministic_prize, name=name)
            with open(log_filename, 'w') as f:
                start = time.time()
                output = check_output(
                    # exe, filename, min_total_prize (=1), num_runs
                    [executable, problem_filename, float_to_scaled_int_str(1.), str(runs)],
                    stderr=f
                ).decode('utf-8')
                duration = time.time() - start
                f.write(output)

            save_dataset((output, duration), output_filename)
        else:
            output, duration = load_dataset(output_filename)

        # Now parse output
        tour = None
        for line in output.splitlines():
            heading = "Best Result Route: "
            if line[:len(heading)] == heading:
                tour = np.array(line[len(heading):].split(" ")).astype(int)
                break
        assert tour is not None, "Could not find tour in output!"

        assert tour[0] == 0, "Tour should start with depot"
        assert tour[-1] == 0, "Tour should end with depot"
        tour = tour[1:-1]  # Strip off depot

        return calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour), tour.tolist(), duration
    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_stochastic_pctsp_log(
        executable, directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize, runs=10, append='all'):

    try:

        problem_filename = os.path.join(directory, "{}.stochpctsp{}{}.pctsp".format(name, append, runs))
        output_filename = os.path.join(directory, "{}.stochpctsp{}{}.pkl".format(name, append, runs))
        log_filename = os.path.join(directory, "{}.stochpctsp{}{}.log".format(name, append, runs))

        # May have already been run
        if not os.path.isfile(output_filename):

            total_start = time.time()

            outputs = []
            durations = []
            final_tour = []

            coord = [depot] + loc

            mask = np.zeros(len(coord), dtype=bool)
            dist = distance_matrix(coord, coord)
            penalty = np.array(penalty)
            deterministic_prize = np.array(deterministic_prize)

            it = 0
            total_collected_prize = 0.
            # As long as we have not visited all nodes we repeat
            # even though we have already satisfied the total prize collected constraint
            # since the algorithm may decide to include more nodes to avoid further penalties
            while len(final_tour) < len(stochastic_prize):

                # Mask all nodes already visited (not the depot)
                mask[final_tour] = True

                # The distance from the 'start' or 'depot' is the distance from the 'current node'
                # this way we mimic as if we have a separate start and end by the assymetric distance matrix
                # Note: this violates the triangle inequality and the distance from 'depot to depot' becomes nonzero
                # but the program seems to deal with this well
                if len(final_tour) > 0:  # in the first iteration we are at depot and distance matrix is ok
                    dist[0, :] = dist[final_tour[-1], :]

                remaining_deterministic_prize = deterministic_prize[~mask[1:]]
                write_pctsp_dist(problem_filename,
                                 dist[np.ix_(~mask, ~mask)], penalty[~mask[1:]], remaining_deterministic_prize)
                # If the remaining deterministic prize is less than the prize we should still collect
                # set this lower value as constraint since otherwise problem is infeasible
                # compute total remaining deterministic prize after converting to ints
                # otherwise we may still have problems with rounding
                # Note we need to clip 1 - total_collected_prize between 0 (constraint can already be satisfied)
                # and the maximum achievable with the remaining_deterministic_prize
                min_prize_int = max(0, min(
                    float_to_scaled_int(1. - total_collected_prize),
                    sum([float_to_scaled_int(v) for v in remaining_deterministic_prize])
                ))
                with open(log_filename, 'a') as f:
                    start = time.time()
                    output = check_output(
                        # exe, filename, min_total_prize (=1), num_runs
                        [executable, problem_filename, str(min_prize_int), str(runs)],
                        stderr=f
                    ).decode('utf-8')
                    durations.append(time.time() - start)
                    outputs.append(output)

                # Now parse output
                tour = None
                for line in output.splitlines():
                    heading = "Best Result Route: "
                    if line[:len(heading)] == heading:
                        tour = np.array(line[len(heading):].split(" ")).astype(int)
                        break
                assert tour is not None, "Could not find tour in output!"

                assert tour[0] == 0, "Tour should start with depot"
                assert tour[-1] == 0, "Tour should end with depot"
                tour = tour[1:-1]  # Strip off depot

                # Now find to which nodes these correspond
                tour_node_ids = np.arange(len(coord), dtype=int)[~mask][tour]

                if len(tour_node_ids) == 0:
                    # The inner algorithm can decide to stop, but does not have to
                    assert total_collected_prize > 1 - 1e-5, "Collected prize should be one"
                    break

                if append == 'first':
                    final_tour.append(tour_node_ids[0])
                elif append == 'half':
                    final_tour.extend(tour_node_ids[:max(len(tour_node_ids) // 2, 1)])
                else:
                    assert append == 'all'
                    final_tour.extend(tour_node_ids)

                total_collected_prize = calc_pctsp_total(stochastic_prize, final_tour)
                it = it + 1

            os.remove(problem_filename)
            final_cost = calc_pctsp_cost(depot, loc, penalty, stochastic_prize, final_tour)
            total_duration = time.time() - total_start
            save_dataset((final_cost, final_tour, total_duration, outputs, durations), output_filename)

        else:
            final_cost, final_tour, total_duration, outputs, durations = load_dataset(output_filename)

        return final_cost, final_tour, total_duration
    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_salesman(directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize, runs=10):

    problem_filename = os.path.join(directory, "{}.salesman{}.pctsp".format(name, runs))
    output_filename = os.path.join(directory, "{}.salesman{}.pkl".format(name, runs))

    try:
        # May have already been run
        if not os.path.isfile(output_filename):
            write_pctsp(problem_filename, depot, loc, penalty, deterministic_prize, name=name)

            start = time.time()

            random.seed(1234)
            pctsp = Pctsp()
            pctsp.load(problem_filename, float_to_scaled_int(1.))
            s = solution.random(pctsp, start_size=int(len(pctsp.prize) * 0.7))
            s = ilocal_search(s, n_runs=runs)

            output = (s.route[:s.size], s.quality)

            duration = time.time() - start

            save_dataset((output, duration), output_filename)
        else:
            output, duration = load_dataset(output_filename)

        # Now parse output
        tour = output[0][:]
        assert tour[0] == 0, "Tour should start with depot"
        assert tour[-1] != 0, "Tour should not end with depot"
        tour = tour[1:]  # Strip off depot

        total_cost = calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour)
        assert (float_to_scaled_int(total_cost) - output[1]) / float(output[1]) < 1e-5
        return total_cost, tour, duration
    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_gurobi(directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize,
                 disable_cache=False, timeout=None, gap=None):
    # Lazy import so we do not need to have gurobi installed to run this script
    from .pctsp_gurobi import solve_euclidian_pctsp as solve_euclidian_pctsp_gurobi

    try:
        problem_filename = os.path.join(directory, "{}.gurobi{}{}.pkl".format(
            name, "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))

        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()

            # Must collect 1 or the sum of the prices if it is less then 1.
            cost, tour = solve_euclidian_pctsp_gurobi(
                depot, loc, penalty, deterministic_prize, min(sum(deterministic_prize), 1.),
                threads=1, timeout=timeout, gap=gap
            )
            duration = time.time() - start  # Measure clock time
            save_dataset((cost, tour, duration), problem_filename)

        # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
        assert tour[0] == 0
        tour = tour[1:]

        total_cost = calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour)
        assert abs(total_cost - cost) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration

    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we can retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def solve_ortools(directory, name, depot, loc, penalty, deterministic_prize, stochastic_prize,
                  sec_local_search=0, disable_cache=False):
    # Lazy import so we do not require ortools by default
    from .pctsp_ortools import solve_pctsp_ortools

    try:
        problem_filename = os.path.join(directory, "{}.ortools{}.pkl".format(name, sec_local_search))
        if os.path.isfile(problem_filename) and not disable_cache:
            objval, tour, duration = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()
            objval, tour = solve_pctsp_ortools(depot, loc, deterministic_prize, penalty,
                                               min(sum(deterministic_prize), 1.), sec_local_search=sec_local_search)
            duration = time.time() - start
            save_dataset((objval, tour, duration), problem_filename)
        assert tour[0] == 0, "Tour must start with depot"
        tour = tour[1:]
        total_cost = calc_pctsp_cost(depot, loc, penalty, deterministic_prize, tour)
        assert abs(total_cost - objval) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration
    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def calc_pctsp_total(vals, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    return np.array(vals)[np.array(tour) - 1].sum()


def calc_pctsp_length(depot, loc, tour):
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def calc_pctsp_cost(depot, loc, penalty, prize, tour):
    # With some tolerance we should satisfy minimum prize
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert calc_pctsp_total(prize, tour) >= 1 - 1e-5 or len(tour) == len(prize), \
        "Tour should collect at least 1 as total prize or visit all nodes"
    # Penalty is only incurred for locations not visited, so charge total penalty minus penalty of locations visited
    return calc_pctsp_length(depot, loc, tour) + np.sum(penalty) - calc_pctsp_total(penalty, tour)


def write_pctsp(filename, depot, loc, penalty, prize, name="problem"):
    coord = [depot] + loc
    return write_pctsp_dist(filename, distance_matrix(coord, coord), penalty, prize)


def float_to_scaled_int_str(v):  # Program only accepts ints so scale everything by 10^7
    return str(float_to_scaled_int(v))


def float_to_scaled_int(v):
    return int(v * 10000000 + 0.5)


def write_pctsp_dist(filename, dist, penalty, prize):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "",
            " ".join([float_to_scaled_int_str(p) for p in [0] + list(prize)]),
            "",
            "",
            " ".join([float_to_scaled_int_str(p) for p in [0] + list(penalty)]),
            "",
            "",
            *(
                " ".join(float_to_scaled_int_str(d) for d in d_row)
                for d_row in dist
            )
        ]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        help="Name of the method to evaluate, 'pctsp', 'salesman' or 'stochpctsp(first|half|all)'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:

        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "pctsp", dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.path.join(results_dir, "{}{}{}-{}{}".format(
                dataset_basename,
                "offs{}".format(opts.offset) if opts.offset is not None else "",
                "n{}".format(opts.n) if opts.n is not None else "",
                opts.method, ext
            ))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        if method in ("pctsp", "salesman", "gurobi", "gurobigap", "gurobit", "ortools") or method[:10] == "stochpctsp":

            target_dir = os.path.join(results_dir, "{}-{}".format(
                dataset_basename,
                opts.method
            ))
            assert opts.f or not os.path.isdir(target_dir), \
                "Target dir already exists! Try running with -f option to overwrite."

            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            dataset = load_dataset(dataset_path)

            if method[:6] == "gurobi":
                use_multiprocessing = True  # We run one thread per instance

                def run_func(args):
                    return solve_gurobi(*args, disable_cache=opts.disable_cache,
                                        timeout=runs if method[6:] == "t" else None,
                                        gap=float(runs) if method[6:] == "gap" else None)
            elif method == "pctsp":
                executable = get_pctsp_executable()
                use_multiprocessing = False

                def run_func(args):
                    return solve_pctsp_log(executable, *args, runs=runs)
            elif method == "salesman":
                use_multiprocessing = True

                def run_func(args):
                    return solve_salesman(*args, runs=runs)
            elif method == "ortools":
                use_multiprocessing = True

                def run_func(args):
                    return solve_ortools(*args, sec_local_search=runs, disable_cache=opts.disable_cache)
            else:
                assert method[:10] == "stochpctsp"
                append = method[10:]
                assert append in ('first', 'half', 'all')
                use_multiprocessing = True

                def run_func(args):
                    return solve_stochastic_pctsp_log(executable, *args, runs=runs, append=append)

            results, parallelism = run_all_in_pool(
                run_func,
                target_dir, dataset, opts, use_multiprocessing=use_multiprocessing
            )

        else:
            assert False, "Unknown method: {}".format(opts.method)

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        save_dataset((results, parallelism), out_file)
