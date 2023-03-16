"""
Used to organize multiple memories into offline datasets.
"""

import pickle
import os
import numpy as np


def read_one_data(fpath):
    sample_file = open(fpath, "rb")
    print("======= load samples for: ==", fpath)
    try:
        sample_set = []
        while True:
            sample_set += pickle.load(sample_file)
    except EOFError:
        print("======= load samples finished======")
        sample_file.close()

    return sample_set


def save_final_data(data, fpath):
    with open(fpath, "wb+") as f:
        pickle.dump(data, f, -1)


def re_organize_dict_memory(data):
    """
    Input: data, collected all the data [[],[],...,[]]
           [state, action, next-state,  final-reward, average-reward]
    Output: [state1:[], state2:[], state3:[], ...]
    """
    total_samples = len(data)
    state, next_state, action, pressure_reward, ql_reward = dict(), dict(), [], [], []
    # state_names = data[0][0].keys()
    # customized state-names
    state_names = ["new_phase", "lane_num_vehicle_in", "lane_num_vehicle_out",
                   "lane_queue_vehicle_in", "lane_queue_vehicle_out",
                   "traffic_movement_pressure_queue_efficient",
                   "lane_run_in_part", "lane_queue_in_part", "num_in_seg"]
    for feat_name in state_names:
        state[feat_name] = []
        next_state[feat_name] = []

    for i in range(total_samples):
        tmp_state, tmp_action, tmp_next_state, _, _ = data[i]
        for feat_name in state_names:
            state[feat_name].append(tmp_state[feat_name])
            next_state[feat_name].append(tmp_next_state[feat_name])
        action.append(tmp_action)
        # final_reward.append(tmp_f_r)
        #average_reward.append(tmp_a_r)
    pressure_reward = list(-np.absolute(np.sum(next_state["traffic_movement_pressure_queue_efficient"], axis=-1)/4))
    ql_reward = list(-np.sum(next_state["lane_queue_vehicle_in"], axis=-1)/4)
    return [state, action, next_state, pressure_reward, ql_reward]


def collect_one_flow_data(path0, flow_path='JN', flow_path2='1', index=0):
    methods_path = [["Expert", "Random", "Cycle"][index]]  # ["FT"]# ["Efficient_MP", "M_QL", "Random"]

    total_samples = []

    for path1 in methods_path:
        path2 = flow_path
        if path2 == "JN":
            total_inters = 12
        elif path2 == "HZ":
            total_inters = 16
        path3 = flow_path2

        for inter_id in range(total_inters):
            sample_path = "total_samples_inter_{0}.pkl".format(inter_id)

            tmp_full_path = os.path.join(path0, path1, path2, path3, sample_path)

            tmp_sample_set = read_one_data(tmp_full_path)
            total_samples.extend(tmp_sample_set)
    print("======== collect data finshed =========")
    new_all = re_organize_dict_memory(total_samples)

    print("======== organize data finished =======")
    return new_all


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def mix_data(name='expert'):
    paths = [name+'_jn_1.pkl', name+'_jn_2.pkl', name+'_jn_3.pkl', name+'_hz_1.pkl', name+'_hz_2.pkl']
    # read_data

    data = []
    for path in paths:
        tmp_data = load_data(path)
        data.append(tmp_data)

    # organize
    state, next_state, action, pressure_reward, ql_reward = dict(), dict(), [], [], []
    # state_names = data[0][0].keys()
    # customized state-names
    state_names = ["new_phase", "lane_num_vehicle_in", "lane_num_vehicle_out",
                   "lane_queue_vehicle_in", "lane_queue_vehicle_out",
                   "traffic_movement_pressure_queue_efficient",
                   "lane_run_in_part", "lane_queue_in_part", "num_in_seg"]
    for feat_name in state_names:
        state[feat_name] = []
        next_state[feat_name] = []

    for i in range(5):
        _s, _a, _n_s, _p, _q = data[i]

        for feat_name in state_names:
            state[feat_name].extend(_s[feat_name][-28800:])
            next_state[feat_name].extend(_n_s[feat_name][-28800:])
        action.extend(_a[-28800:])
        pressure_reward.extend(_p[-28800:])
        ql_reward.extend(_q[-28800:])
    return [state, action, next_state, pressure_reward, ql_reward]


if __name__ == '__main__':
    # examples of collect expert data from JN-1
    data1 = collect_one_flow_data("./", index=0)
    save_final_data(data1, "./expert_jn_1.pkl")
    data2 = collect_one_flow_data("./", index=0, flow_path2=2)
    save_final_data(data2, "./expert_jn_2.pkl")
    data3 = collect_one_flow_data("./", index=0, flow_path2=3)
    save_final_data(data3, "./expert_jn_3.pkl")
    data4 = collect_one_flow_data("./", index=0, flow_path="HZ", flow_path2=1)
    save_final_data(data4, "./expert_hz_1.pkl")
    data5 = collect_one_flow_data("./", index=0, flow_path="HZ", flow_path2=1)
    save_final_data(data5, "./expert_hz_2.pkl")
    # collect them together

    data_6 = mix_data()
    save_final_data(data_6, 'expert_mix.pkl')


