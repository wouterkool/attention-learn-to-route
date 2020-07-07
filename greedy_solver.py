import os
import pandas as pd
from problems.vrp.problem_vrp import CVRP
import torch
from torch.utils.data import DataLoader
import math
shift_time = 1200

# Note, this function also updates the cur_time in cases of idle
def find_nearest(prev, coords, mask):
    min_dist, min_index = torch.ones(coords.shape[0], dtype=float) * 10000, torch.ones(coords.shape[0], dtype=int) * -1

    for j in range(coords.shape[0]):
        for i in range(coords.shape[1]):
            if ~mask.squeeze()[j][i]:

                dist_eu = math.sqrt((coords[j][prev[j]][0] - coords[j][i][0]) ** 2 + (coords[j][prev[j]][1] - coords[j][i][1]) ** 2)
                if dist_eu < min_dist[j]:  # and dist_eu < min_dist:
                    min_index[j] = i
                    min_dist[j] = dist_eu
        assert min_index[j] > -1, 'could not find any available node'
    return min_index


def get_route(input):

    sequences = []

    state = CVRP.make_state(input, visited_dtype=torch.uint8)
    state.initialize(input)

    # Perform decoding steps
    i, batch_size = 0, input['loc'].shape[0]
    # max_route_length = int(len(input['tos'][1]) * 1.5)
    prev = torch.zeros(batch_size, dtype=int) # start at the Depo
    while not (state.all_finished()):
        if i > 150:
            print('Too many iterations. i = {}. Breaking'.format(i))
            break # TODO: Shouldent happen - or allow Unassigned tasks

        mask = state.get_mask()

        selected = find_nearest(prev, state.coords, mask)  #
        state = state.update(selected)
        sequences.append(selected)
        prev = selected
        i += 1

    # Collected lists, return Tensor
    return sequences, state.get_final_cost()

def persist_results(pi_i, input_data, arrive_at, output_folder=None):
    l = [i[0].numpy().item(0) for i in pi_i]
    l.insert(0, 0)  # Insert the Depot at the beginning
    l.append(0)
    df = pd.DataFrame({'loc_id': l, 'lat': 0.0, 'lng': 0.0, 'drive_time': 0.0,
                       'arrive_at': 0.0, 'TOS': 0, 'tw_start': 0, 'tw_end': 0})

    for index, row in df.iterrows():
        if 'original_loc' in input_data.keys():
            cur_point = input_data['original_loc'][l[index]]
        else:
            cur_point = input_data['loc'][l[index] - 1]
        cur_tw = input_data['tw'][l[index] - 1]
        df.lng.at[index], df.lat.at[index] = cur_point[1], cur_point[0]
        df.TOS.at[index] = input_data['tos'][l[index]]
        if index < len(l) - 1:
            df.arrive_at.at[index] = float(arrive_at[index - 1])
        df.drive_time.at[index] = input_data['wmat'][l[index - 1]][l[index]]
        if l[index] > 0:
            df.tw_start.at[index], df.tw_end.at[index] = cur_tw[0], cur_tw[1]
    df.arrive_at[0] = 0.0
    cost = str(int(df.drive_time.sum()))
    if output_folder is not None:
        file = os.path.join(output_folder, 'input_table_' + cost + '.csv')
        df.to_csv(file)
        route = os.path.join(output_folder, 'greedy_map_' + cost + '.htm')

        # out = save_df_on_map(df, output_file=route, persist_map=True, is_jsprit=False)
    return df

# def get_cost(input, sequence):
#     lambda input, pi: self.problem.get_costs(input[0], pi)

def get_costs(dataset, pi):
    loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
    d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

    # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
    return (
                   (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                   + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                   + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
           ), None