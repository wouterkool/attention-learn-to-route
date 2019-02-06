import argparse
import numpy as np
import os
import time
from datetime import timedelta
from scipy.spatial import distance_matrix
from utils import run_all_in_pool
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output, CalledProcessError
from problems.vrp.vrp_baseline import get_lkh_executable
import torch
from tqdm import tqdm
import re


def solve_gurobi(directory, name, loc, disable_cache=False, timeout=None, gap=None):
    # Lazy import so we do not need to have gurobi installed to run this script
    from problems.tsp.tsp_gurobi import solve_euclidian_tsp as solve_euclidian_tsp_gurobi

    try:
        problem_filename = os.path.join(directory, "{}.gurobi{}{}.pkl".format(
            name, "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))

        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()

            cost, tour = solve_euclidian_tsp_gurobi(loc, threads=1, timeout=timeout, gap=gap)
            duration = time.time() - start  # Measure clock time
            save_dataset((cost, tour, duration), problem_filename)

        # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
        total_cost = calc_tsp_length(loc, tour)
        assert abs(total_cost - cost) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration

    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def solve_concorde_log(executable, directory, name, loc, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.tsp".format(name))
    tour_filename = os.path.join(directory, "{}.tour".format(name))
    output_filename = os.path.join(directory, "{}.concorde.pkl".format(name))
    log_filename = os.path.join(directory, "{}.log".format(name))

    # if True:
    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, name=name)

            with open(log_filename, 'w') as f:
                start = time.time()
                try:
                    # Concorde is weird, will leave traces of solution in current directory so call from target dir
                    check_call([executable, '-s', '1234', '-x', '-o',
                                os.path.abspath(tour_filename), os.path.abspath(problem_filename)],
                               stdout=f, stderr=f, cwd=directory)
                except CalledProcessError as e:
                    # Somehow Concorde returns 255
                    assert e.returncode == 255
                duration = time.time() - start

            tour = read_concorde_tour(tour_filename)
            save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_lkh_log(executable, directory, name, loc, runs=1, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.lkh{}.vrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.lkh{}.tour".format(name, runs))
    output_filename = os.path.join(directory, "{}.lkh{}.pkl".format(name, runs))
    param_filename = os.path.join(directory, "{}.lkh{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.lkh{}.log".format(name, runs))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, name=name)

            params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": 1234}
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_tsplib(tour_filename)
            save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def write_tsplib(filename, loc, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # tsplib does not take floats
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")


def read_concorde_tour(filename):
    with open(filename, 'r') as f:
        n = None
        tour = []
        for line in f:
            if n is None:
                n = int(line)
            else:
                tour.extend([int(node) for node in line.rstrip().split(" ")])
    assert len(tour) == n, "Unexpected tour length"
    return tour


def read_tsplib(filename):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    return tour.tolist()


def calc_tsp_length(loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc)
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def _calc_insert_cost(D, prv, nxt, ins):
    """
    Calculates insertion costs of inserting ins between prv and nxt
    :param D: distance matrix
    :param prv: node before inserted node, can be vector
    :param nxt: node after inserted node, can be vector
    :param ins: node to insert
    :return:
    """
    return (
        D[prv, ins]
        + D[ins, nxt]
        - D[prv, nxt]
    )


def run_insertion(loc, method):
    n = len(loc)
    D = distance_matrix(loc, loc)

    mask = np.zeros(n, dtype=bool)
    tour = []  # np.empty((0, ), dtype=int)
    for i in range(n):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)
        if method == 'random':
            # Order of instance is random so do in order for deterministic results
            a = i
        elif method == 'nearest':
            if i == 0:
                a = 0  # order does not matter so first is random
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmin()] # node nearest to any in tour
        elif method == 'cheapest':
            assert False, "Not yet implemented" # try all and find cheapest insertion cost

        elif method == 'farthest':
            if i == 0:
                a = D.max(1).argmax()  # Node with farthest distance to any other node
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
        mask[a] = True

        if len(tour) == 0:
            tour = [a]
        else:
            # Find index with least insert cost
            ind_insert = np.argmin(
                _calc_insert_cost(
                    D,
                    tour,
                    np.roll(tour, -1),
                    a
                )
            )
            tour.insert(ind_insert + 1, a)

    cost = D[tour, np.roll(tour, -1)].sum()
    return cost, tour


def solve_insertion(directory, name, loc, method='random'):
    start = time.time()
    cost, tour = run_insertion(loc, method)
    duration = time.time() - start
    return cost, tour, duration


def calc_batch_pdist(dataset):
    diff = (dataset[:, :, None, :] - dataset[:, None, :, :])
    return torch.matmul(diff[:, :, :, None, :], diff[:, :, :, :, None]).squeeze(-1).squeeze(-1).sqrt()


def nearest_neighbour(dataset, start='first'):
    dist = calc_batch_pdist(dataset)

    batch_size, graph_size, _ = dataset.size()

    total_dist = dataset.new(batch_size).zero_()

    if not isinstance(start, torch.Tensor):
        if start == 'random':
            start = dataset.new().long().new(batch_size).zero_().random_(0, graph_size)
        elif start == 'first':
            start = dataset.new().long().new(batch_size).zero_()
        elif start == 'center':
            _, start = dist.mean(2).min(1)  # Minimum total distance to others
        else:
            assert False, "Unknown start: {}".format(start)

    current = start
    dist_to_startnode = torch.gather(dist, 2, current.view(-1, 1, 1).expand(batch_size, graph_size, 1)).squeeze(2)
    tour = [current]

    for i in range(graph_size - 1):
        # Mark out current node as option
        dist.scatter_(2, current.view(-1, 1, 1).expand(batch_size, graph_size, 1), np.inf)
        nn_dist = torch.gather(dist, 1, current.view(-1, 1, 1).expand(batch_size, 1, graph_size)).squeeze(1)

        min_nn_dist, current = nn_dist.min(1)
        total_dist += min_nn_dist
        tour.append(current)

    total_dist += torch.gather(dist_to_startnode, 1, current.view(-1, 1)).squeeze(1)

    return total_dist, torch.stack(tour, dim=1)


def solve_all_nn(dataset_path, eval_batch_size=1024, no_cuda=False, dataset_n=None, progress_bar_mininterval=0.1):
    import torch
    from torch.utils.data import DataLoader
    from problems import TSP
    from utils import move_to

    dataloader = DataLoader(
        TSP.make_dataset(filename=dataset_path, num_samples=dataset_n if dataset_n is not None else 1000000),
        batch_size=eval_batch_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")
    results = []
    for batch in tqdm(dataloader, mininterval=progress_bar_mininterval):
        start = time.time()
        batch = move_to(batch, device)

        lengths, tours = nearest_neighbour(batch)
        lengths_check, _ = TSP.get_costs(batch, tours)

        assert (torch.abs(lengths - lengths_check.data) < 1e-5).all()

        duration = time.time() - start
        results.extend(
            [(cost.item(), np.trim_zeros(pi.cpu().numpy(), 'b'), duration) for cost, pi in zip(lengths, tours)])

    return results, eval_batch_size


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        help="Name of the method to evaluate, 'nn', 'gurobi' or '(nearest|random|farthest)_insertion'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA (only for Tsiligirides)')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
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
            results_dir = os.path.join(opts.results_dir, "tsp", dataset_basename)
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

        match = re.match(r'^([a-z_]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        if method == "nn":
            assert opts.offset is None, "Offset not supported for nearest neighbor"

            eval_batch_size = opts.max_calc_batch_size

            results, parallelism = solve_all_nn(
                dataset_path, eval_batch_size, opts.no_cuda, opts.n,
                opts.progress_bar_mininterval
            )
        elif method in ("gurobi", "gurobigap", "gurobit", "concorde", "lkh") or method[-9:] == 'insertion':

            target_dir = os.path.join(results_dir, "{}-{}".format(
                dataset_basename,
                opts.method
            ))
            assert opts.f or not os.path.isdir(target_dir), \
                "Target dir already exists! Try running with -f option to overwrite."

            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            # TSP contains single loc array rather than tuple
            dataset = [(instance, ) for instance in load_dataset(dataset_path)]

            if method == "concorde":
                use_multiprocessing = False
                executable = os.path.abspath(os.path.join('problems', 'tsp', 'concorde', 'concorde', 'TSP', 'concorde'))

                def run_func(args):
                    return solve_concorde_log(executable, *args, disable_cache=opts.disable_cache)

            elif method == "lkh":
                use_multiprocessing = False
                executable = get_lkh_executable()

                def run_func(args):
                    return solve_lkh_log(executable, *args, runs=runs, disable_cache=opts.disable_cache)

            elif method[:6] == "gurobi":
                use_multiprocessing = True  # We run one thread per instance

                def run_func(args):
                    return solve_gurobi(*args, disable_cache=opts.disable_cache,
                                        timeout=runs if method[6:] == "t" else None,
                                        gap=float(runs) if method[6:] == "gap" else None)
            else:
                assert method[-9:] == "insertion"
                use_multiprocessing = True

                def run_func(args):
                    return solve_insertion(*args, opts.method.split("_")[0])

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
