import argparse
import os
import numpy as np
from data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.ones(dataset_size) * CAPACITIES[vrp_size]  # Capacity, same for whole dataset
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Filename of the dataset to create")
    parser.add_argument("--problem", type=str, default='tsp', help="Problem, 'tsp' or 'vrp'")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_size', type=int, default=20, help="Size of problem instances")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")

    opts = parser.parse_args()

    assert opts.f or not os.path.isfile(check_extension(opts.dataset)), \
        "File already exists! Try running with -f option to overwrite."

    np.random.seed(opts.seed)
    if opts.problem == 'tsp':
        dataset = generate_tsp_data(opts.dataset_size, opts.graph_size)
    elif opts.problem == 'vrp':
        dataset = generate_vrp_data(opts.dataset_size, opts.graph_size)
    else:
        assert False, "Unknown problem: {}".format(opts.problem)

    print(dataset[0])
    filename = check_extension(opts.dataset)

    save_dataset(dataset, opts.dataset)
