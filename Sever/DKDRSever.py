import copy

import numpy as np
import torch

from Sever.utils.sever_methods import SeverMethod

from utils.utils import row_into_parameters

class DKDRSever(SeverMethod):
    NAME = 'DKDRSever'  # 定义服务器类名称

    def __init__(self, args, cfg):
        super(DKDRSever, self).__init__(args, cfg)  # 调用父类的初始化方法
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

        ns = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        with torch.no_grad():
            all_delta = []
            w = []
            for i in online_clients_list:
                net_all_delta = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[i].state_dict()[name]

                    delta = (param1.detach() - param0.detach())

                    net_all_delta.append(copy.deepcopy(delta.view(-1)))
                w.append(nets_list[i].state_dict())
                net_all_delta = torch.cat(net_all_delta, dim=0).cpu().numpy()
                net_all_delta /= (np.linalg.norm(net_all_delta) + 1e-5)
                all_delta.append(net_all_delta)
            all_delta = np.array(all_delta)

        updates = all_delta
        U, S, Vh = np.linalg.svd(updates, full_matrices=False)
        density = 0.2
        new_rank = int(len(S) * density)
        U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
        res = U @ np.diag(S) @ Vh

        svd_list = res
        prop = torch.tensor(ns, dtype=torch.float)
        prop /= torch.sum(prop)
        w_avg = copy.deepcopy(w[0])

        clients_dist = np.zeros((len(w), len(w)))
        for i in range(len(w)):
            for j in range(len(w)):
                dot_product = np.dot(svd_list[i], svd_list[j])
                norm_A = np.linalg.norm(svd_list[i])
                norm_B = np.linalg.norm(svd_list[j])
                similarity = dot_product / (norm_A * norm_B)
                clients_dist[i][j] = similarity

        if_nei = np.zeros((len(w), len(w)))

        for i in range(len(w)):
            maxd = -2
            for j in range(len(w)):
                if clients_dist[i][j] > maxd and i != j:
                    maxd = clients_dist[i][j]

            for j in range(len(w)):
                if clients_dist[i][j] == maxd:
                    if_nei[i][j] = 1
                    if_nei[j][i] = 1
        self.clusters = find_clusters(if_nei)
        self.domain_w = []
        self.domain_model = []

        for cluster in self.clusters:
            new_ns = []
            for i in cluster:
                new_ns.append(ns[i])

            new_prop = torch.tensor(new_ns, dtype=torch.float)
            new_prop /= torch.sum(new_prop)
            self.domain_w.append(sum(new_ns)/sum(ns))
            w_domain = copy.deepcopy(w[cluster[0]])

            for k in w_domain.keys():
                w_domain[k] = w_domain[k] * prop[cluster[0]]/torch.sum(new_prop)

            for k in w_domain.keys():
                first = 1
                for i in cluster:
                    if first:
                        first = 0
                        continue
                    w_domain[k] += w[i][k] * prop[i]/torch.sum(new_prop)
            self.domain_model.append(copy.deepcopy(w_domain))

        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * prop[i]

        global_net.load_state_dict(w_avg)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
        return self.domain_model ,self.domain_w, self.clusters

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
        if not visited[i] and graph[node][i]:
            dfs(i, graph, visited, cluster)