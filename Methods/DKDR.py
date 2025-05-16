from Aggregations import get_fed_aggregation
from Methods.utils.meta_methods import FederatedMethod

import copy
import wandb


class DKDR(FederatedMethod):
    NAME = 'DKDR'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(DKDR, self).__init__(nets_list, client_domain_list, args, cfg)
        self.w_domain = []
        self.domain_model = []
        self.clusters = []


    def ini(self):
        super().ini()

    def local_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()

        self.local_model.loc_update(online_clients_list=total_clients,
                                    nets_list=self.nets_list, global_net=self.global_net,
                                    priloader_list=priloader_list,domain_model=self.domain_model,w_domain=self.w_domain, clusters=self.clusters)

    def sever_update(self, priloader_list):
        self.domain_model, self.w_domain, self.clusters = self.sever_model.sever_update(fed_aggregation=self.fed_aggregation, online_clients_list=self.online_clients_list,
                                                                     priloader_list=priloader_list, client_domain_list=self.client_domain_list,
                                                                     global_net=self.global_net, nets_list=self.nets_list)
