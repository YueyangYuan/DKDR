import copy

import numpy as np
import torch

from Sever.utils.sever_methods import SeverMethod


class DKDRSever(SeverMethod):
    NAME = 'DKDRSever'

    def __init__(self, args, cfg):
        super(DKDRSever, self).__init__(args, cfg)
        self.domain_model = []
        self.domain_w = []
        self.clusters = []

    def sever_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)
        fed_aggregation = kwargs['fed_aggregation']

        ns = fed_aggregation.weight_calculate(
            online_clients_list=online_clients_list,
            priloader_list=priloader_list
        )

        with torch.no_grad():
            all_delta = []
            w = []
            for client_id in online_clients_list:
                net_all_delta = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[client_id].state_dict()[name]
                    delta = param1.detach() - param0.detach()
                    net_all_delta.append(delta.view(-1).cpu())

                w.append(copy.deepcopy(nets_list[client_id].state_dict()))

                net_all_delta = torch.cat(net_all_delta, dim=0).numpy()
                net_all_delta = net_all_delta / (np.linalg.norm(net_all_delta) + 1e-12)
                all_delta.append(net_all_delta)

            all_delta = np.array(all_delta, dtype=np.float64)
        updates = all_delta
        U, S, Vh = np.linalg.svd(updates, full_matrices=False)

        density = 0.2
        new_rank = max(1, int(len(S) * density))
        U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
        res = U @ np.diag(S) @ Vh

        svd_list = res

        self.clusters = finch_first_partition(svd_list)

        self.domain_w = []
        self.domain_model = []

        ns = np.array(ns, dtype=np.float64)
        prop = torch.tensor(ns, dtype=torch.float32)
        prop = prop / torch.sum(prop)

        total_ns = np.sum(ns)

        for cluster in self.clusters:
            cluster_ns = ns[cluster]
            cluster_sum = np.sum(cluster_ns)

            cluster_prop = prop[cluster]
            cluster_prop = cluster_prop / torch.sum(cluster_prop)

            self.domain_w.append(cluster_sum / total_ns)

            w_domain = copy.deepcopy(w[cluster[0]])
            for k in w_domain.keys():
                w_domain[k] = w_domain[k] * cluster_prop[0]

            for idx_in_cluster, client_idx in enumerate(cluster[1:], start=1):
                for k in w_domain.keys():
                    w_domain[k] += w[client_idx][k] * cluster_prop[idx_in_cluster]

            self.domain_model.append(copy.deepcopy(w_domain))

        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for i in range(1, len(w)):
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * prop[i]

        global_net.load_state_dict(w_avg)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

        return self.domain_model, self.domain_w, self.clusters


def finch_first_partition(X):

    n = X.shape[0]

    if n == 0:
        return []
    if n == 1:
        return [[0]]

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    sim = X @ X.T

    np.fill_diagonal(sim, -np.inf)

    first_neighbors = np.argmax(sim, axis=1)

    F = np.zeros((n, n), dtype=np.int32)
    F[np.arange(n), first_neighbors] = 1

    A = np.eye(n, dtype=np.int32) + F + F.T + (F @ F.T)
    A = (A > 0).astype(np.int32)

    clusters = find_clusters(A)
    return clusters


def find_clusters(graph):
    visited = [False] * len(graph)
    clusters = []

    for node in range(len(graph)):
        if not visited[node]:
            cluster = []
            dfs(node, graph, visited, cluster)
            clusters.append(cluster)

    return clusters


def dfs(node, graph, visited, cluster):
    visited[node] = True
    cluster.append(node)
    for i in range(len(graph)):
        if (not visited[i]) and (graph[node][i] != 0):
            dfs(i, graph, visited, cluster)
