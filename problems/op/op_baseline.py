import argparse
import os
import numpy as np
from utils import run_all_in_pool
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output
import tempfile
import time
from datetime import timedelta
from problems.op.opga.opevo import run_alg as run_opga_alg
from tqdm import tqdm
import re

MAX_LENGTH_TOL = 1e-5


# Run install_compass.sh to install
def solve_compass(executable, depot, loc, demand, capacity):
    with tempfile.TemporaryDirectory() as tempdir:
        problem_filename = os.path.join(tempdir, "problem.oplib")
        output_filename = os.path.join(tempdir, "output.tour")
        param_filename = os.path.join(tempdir, "params.par")

        starttime = time.time()
        write_oplib(problem_filename, depot, loc, demand, capacity)
        params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
        write_compass_par(param_filename, params)
        output = check_output([executable, param_filename])
        result = read_oplib(output_filename, n=len(demand))
        duration = time.time() - starttime
        return result, output, duration


def solve_compass_log(executable, directory, name, depot, loc, prize, max_length, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.oplib".format(name))
    tour_filename = os.path.join(directory, "{}.tour".format(name))
    output_filename = os.path.join(directory, "{}.compass.pkl".format(name))
    log_filename = os.path.join(directory, "{}.log".format(name))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_oplib(problem_filename, depot, loc, prize, max_length, name=name)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename],
                           stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_oplib(tour_filename, n=len(prize))
            if not calc_op_length(depot, loc, tour) <= max_length:
                print("Warning: length exceeds max length:", calc_op_length(depot, loc, tour), max_length)
            assert calc_op_length(depot, loc, tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
            save_dataset((tour, duration), output_filename)

        return -calc_op_total(prize, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def calc_op_total(prize, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    return np.array(prize)[np.array(tour) - 1].sum()


def calc_op_length(depot, loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def write_compass_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
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


def read_oplib(filename, n):
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

            if line.startswith("NODE_SEQUENCE_SECTION"):
                started = True
    
    assert len(tour) > 0, "Unexpected length"
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_oplib(filename, depot, loc, prize, max_length, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "OP"),
                ("DIMENSION", len(loc) + 1),
                ("COST_LIMIT", int(max_length * 10000000 + 0.5)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # oplib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("NODE_SCORE_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + prize)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def solve_opga(directory, name, depot, loc, prize, max_length, disable_cache=False):
    problem_filename = os.path.join(directory, "{}.opga.pkl".format(name))
    if os.path.isfile(problem_filename) and not disable_cache:
        (prize, tour, duration) = load_dataset(problem_filename)
    else:
        # 0 = start, 1 = end so add depot twice
        start = time.time()
        prize, tour, duration = run_opga_alg(
            [(*pos, p) for p, pos in zip([0, 0] + prize, [depot, depot] + loc)],
            max_length, return_sol=True, verbose=False
        )
        duration = time.time() - start  # Measure clock time
        save_dataset((prize, tour, duration), problem_filename)

    # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
    assert tour[0][3] == 0
    assert tour[-1][3] == 1
    return -prize, [i - 1 for x, y, p, i, t in tour[1:-1]], duration


def solve_gurobi(directory, name, depot, loc, prize, max_length, disable_cache=False, timeout=None, gap=None):
    # Lazy import so we do not need to have gurobi installed to run this script
    from problems.op.op_gurobi import solve_euclidian_op as solve_euclidian_op_gurobi

    try:
        problem_filename = os.path.join(directory, "{}.gurobi{}{}.pkl".format(
            name, "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))

        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()

            cost, tour = solve_euclidian_op_gurobi(
                depot, loc, prize, max_length, threads=1, timeout=timeout, gap=gap
            )
            duration = time.time() - start  # Measure clock time
            save_dataset((cost, tour, duration), problem_filename)

        # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
        assert tour[0] == 0
        tour = tour[1:]
        assert calc_op_length(depot, loc, tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
        total_cost = -calc_op_total(prize, tour)
        assert abs(total_cost - cost) <= 1e-4, "Cost is incorrect"
        return total_cost, tour, duration

    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def solve_ortools(directory, name, depot, loc, prize, max_length, sec_local_search=0, disable_cache=False):
    # Lazy import so we do not require ortools by default
    from problems.op.op_ortools import solve_op_ortools

    try:
        problem_filename = os.path.join(directory, "{}.ortools{}.pkl".format(name, sec_local_search))
        if os.path.isfile(problem_filename) and not disable_cache:
            objval, tour, duration = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()
            objval, tour = solve_op_ortools(depot, loc, prize, max_length, sec_local_search=sec_local_search)
            duration = time.time() - start
            save_dataset((objval, tour, duration), problem_filename)
        assert tour[0] == 0, "Tour must start with depot"
        tour = tour[1:]
        assert calc_op_length(depot, loc, tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
        assert abs(-calc_op_total(prize, tour) - objval) <= 1e-5, "Cost is incorrect"
        return -calc_op_total(prize, tour), tour, duration
    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def run_all_tsiligirides(
        dataset_path, sample, num_samples, eval_batch_size, max_calc_batch_size, no_cuda=False, dataset_n=None,
        progress_bar_mininterval=0.1, seed=1234):
    import torch
    from torch.utils.data import DataLoader
    from utils import move_to, sample_many
    from problems.op.tsiligirides import op_tsiligirides
    from problems.op.problem_op import OP
    torch.manual_seed(seed)

    dataloader = DataLoader(
        OP.make_dataset(filename=dataset_path, num_samples=dataset_n if dataset_n is not None else 1000000),
        batch_size=eval_batch_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")
    results = []
    for batch in tqdm(dataloader, mininterval=progress_bar_mininterval):
        start = time.time()
        batch = move_to(batch, device)

        with torch.no_grad():
            if num_samples * eval_batch_size > max_calc_batch_size:
                assert eval_batch_size == 1
                assert num_samples % max_calc_batch_size == 0
                batch_rep = max_calc_batch_size
                iter_rep = num_samples // max_calc_batch_size
            else:
                batch_rep = num_samples
                iter_rep = 1
            sequences, costs = sample_many(
                lambda inp: (None, op_tsiligirides(inp, sample)),
                OP.get_costs,
                batch, batch_rep=batch_rep, iter_rep=iter_rep)
            duration = time.time() - start
            results.extend(
                [(cost.item(), np.trim_zeros(pi.cpu().numpy(),'b'), duration) for cost, pi in zip(costs, sequences)])
    return results, eval_batch_size


if __name__ == "__main__":
    executable = os.path.abspath(os.path.join('problems', 'op', 'compass', 'compass'))

    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Name of the method to evaluate, 'compass', 'opga' or 'tsili'")
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
            results_dir = os.path.join(opts.results_dir, "op", dataset_basename)
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

        if method == "tsili" or method == "tsiligreedy":
            assert opts.offset is None, "Offset not supported for Tsiligirides"

            if method == "tsiligreedy":
                sample = False
                num_samples = 1
            else:
                sample = True
                num_samples = runs

            eval_batch_size = max(1, opts.max_calc_batch_size // num_samples)

            results, parallelism = run_all_tsiligirides(
                dataset_path, sample, num_samples, eval_batch_size, opts.max_calc_batch_size, opts.no_cuda, opts.n,
                opts.progress_bar_mininterval
            )
        elif method in ("compass", "opga", "gurobi", "gurobigap", "gurobit", "ortools"):

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
            elif method == "compass":
                use_multiprocessing = False

                def run_func(args):
                    return solve_compass_log(executable, *args, disable_cache=opts.disable_cache)
            elif method == "opga":
                use_multiprocessing = True

                def run_func(args):
                    return solve_opga(*args, disable_cache=opts.disable_cache)
            else:
                assert method == "ortools"
                use_multiprocessing = True

                def run_func(args):
                    return solve_ortools(*args, sec_local_search=runs, disable_cache=opts.disable_cache)

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
