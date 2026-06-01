import functools
import os
from collections import Counter

import numpy as np
import torch


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "TEST": 31,
    }
    return "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def set_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def ini_client_domain(domains_list, parti_num, rand_domain_select=False):
    if parti_num < len(domains_list):
        raise ValueError("parti_num must be at least the number of domains.")

    selected_domain_list = list(domains_list)
    remaining = parti_num - len(domains_list)

    if rand_domain_select and remaining > 0:
        selected_domain_list.extend(np.random.choice(domains_list, size=remaining, replace=True).tolist())
    else:
        base, extra = divmod(remaining, len(domains_list))
        for index, domain in enumerate(domains_list):
            repeat = base + (1 if index < extra else 0)
            selected_domain_list.extend([domain] * repeat)

    selected_domain_list = np.random.permutation(selected_domain_list).tolist()
    print(log_msg(selected_domain_list))
    print(log_msg(Counter(selected_domain_list)))
    return selected_domain_list


def cal_client_weight(online_clients_list, client_domain_list, freq):
    client_weight = {}
    for index, item in enumerate(online_clients_list):
        client_domain = client_domain_list[item]
        client_freq = freq[index]
        client_weight[str(item) + ":" + client_domain] = round(client_freq, 3)
    return client_weight


def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = row[offset : offset + new_size]
        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size
