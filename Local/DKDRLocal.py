import copy
import torch
from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
class DKDR_Loss(nn.Module):

    def __init__(self, num_classes=10, tau=1, beta=1, mu=0.5):
        super(DKDR_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta
        self.mu = mu
    def forward(self, logits, dg_logits):

        fkd_loss= self.fkd(logits, dg_logits)

        loss = self.beta * fkd_loss

        return loss

    def get_gam(self,logits, dg_logits):

        dg_logits = torch.masked_fill(dg_logits, torch.isinf(dg_logits), 0).to(torch.float32)
        logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)

        dg_probs = F.softmax(dg_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()
        re_dg_probs, idx = dg_probs.sort(dim=-1, descending=True)
        re_student_probs = student_probs.gather(dim=-1, index=idx)

        errors = torch.abs(re_dg_probs - re_student_probs)
        if errors.sum() == 0:
            return 0.5,0.5

        cum_sum = torch.cumsum(re_dg_probs, dim=-1)
        mask = cum_sum > self.mu
        mask[:, 0] = False
        s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
        s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)
        sum_s12 = s1 + s2
        s1 = s1 / sum_s12
        s2 = s2 / sum_s12
        s1 = torch.mean(s1, dim=0)
        s2 = torch.mean(s2, dim=0)

        return s1,s2

    def fkd(self, logits, dg_logits):

        pred_probs_f = F.log_softmax(logits / self.tau, dim=1)
        pred_probs_r = torch.softmax(logits / self.tau, dim=1)

        with torch.no_grad():
            dg_probs_f = torch.softmax(dg_logits / self.tau, dim=1)
            dg_probs_r = F.log_softmax(dg_logits / self.tau, dim=1)

        gam11,gam12 = self.get_gam(logits, dg_logits)
        DKDR_loss = (self.tau ** 2) * self.KLDiv(pred_probs_f, dg_probs_f) * gam11 + (self.tau ** 2) * self.KLDiv(
            dg_probs_r, pred_probs_r) * gam12
        return DKDR_loss


class DKDRLocal(LocalMethod):
    NAME = 'DKDRLocal'

    def __init__(self, args, cfg):
        super(DKDRLocal, self).__init__(args, cfg)
        self.tau = cfg.Local[self.NAME].tau
        self.beta = cfg.Local[self.NAME].beta
        self.criterion = DKDR_Loss(num_classes=self.cfg.DATASET.n_classes, tau=self.tau, beta=self.beta)

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        domain_model = kwargs['domain_model']
        w_domain = kwargs['w_domain']
        clusters = kwargs['clusters']
        for i in online_clients_list:
            self.train_net(i, nets_list[i], global_net, priloader_list[i],domain_model,w_domain,clusters)


    def train_net(self, index, net, global_net, train_loader, dm,w_domain,clusters):
        net = net.to(self.device)
        net.train()
        domain_model = []
        m = copy.deepcopy(global_net)
        for i in range(len(dm)):
            m.load_state_dict(dm[i])
            domain_model.append(copy.deepcopy(m).to(self.device))
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        self.criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        CE = nn.CrossEntropyLoss()
        for _ in iterator:
            i = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                i+=1
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = net(images)
                loss = CE(logits, labels)
                if len(domain_model) == 0:
                    w_domain.append(1)
                    domain_model.append(global_net)
                for i in range(len(domain_model)):
                    with torch.no_grad():
                        model_logits = domain_model[i](images)
                    iloss = self.criterion(logits, model_logits)/len(domain_model)
                    loss += iloss
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                optimizer.step()

