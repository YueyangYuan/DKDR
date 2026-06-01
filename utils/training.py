import copy

import numpy as np

from Methods.utils.meta_methods import FederatedMethod
from utils.logger import CsvWriter
from utils.utils import log_msg

try:
    import torch
except ImportError:  # pragma: no cover - optional during file-only checks
    torch = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


def cal_top_one_five(net, test_dl, device):
    net.eval()
    total, top1, top5 = 0.0, 0.0, 0.0
    for images, labels in test_dl:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)
            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
    net.train()
    return round(100 * top1 / total, 2), round(100 * top5 / total, 2)


def global_in_evaluation(optimizer: FederatedMethod, test_loader, in_domain_list):
    in_domain_accs = []
    for in_domain in in_domain_list:
        global_net = optimizer.global_net
        global_net.eval()
        top1acc, _ = cal_top_one_five(global_net, test_loader[in_domain], optimizer.device)
        in_domain_accs.append(top1acc)
        global_net.train()
    return in_domain_accs, round(np.mean(in_domain_accs, axis=0), 3)


def fill_blank(net_cls_counts, classes):
    expected = list(range(classes))
    for _, class_count in net_cls_counts.items():
        for class_index in expected:
            class_count.setdefault(class_index, 0)
    return net_cls_counts


def train(fed_method, private_dataset, args, cfg, client_domain_list) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, cfg)

    if hasattr(fed_method, "ini"):
        fed_method.ini()

    if args.task == "label_skew":
        mean_in_domain_acc_list = []
        fed_method.net_cls_counts = fill_blank(private_dataset.net_cls_counts, cfg.DATASET.n_classes)
    else:
        in_domain_accs_dict = {}
        mean_in_domain_acc_list = []
        performance_variance_list = []

    for epoch_index in range(cfg.DATASET.communication_epoch):
        fed_method.epoch_index = epoch_index
        fed_method.test_loader = private_dataset.test_loader
        fed_method.local_update(private_dataset.train_loaders)
        fed_method.nets_list_before_agg = copy.deepcopy(fed_method.nets_list)
        fed_method.sever_update(private_dataset.train_loaders)

        if args.task == "label_skew":
            top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.test_loader, fed_method.device)
            mean_in_domain_acc_list.append(top1acc)
            print(log_msg(f"The {epoch_index} Epoch: Acc:{top1acc}", "TEST"))
            if wandb is not None:
                wandb.log({"acc": top1acc}, step=epoch_index)
            continue

        domain_accs, mean_in_domain_acc = global_in_evaluation(
            fed_method,
            private_dataset.test_loader,
            private_dataset.domain_list,
        )
        perf_var = np.var(domain_accs, ddof=0)
        performance_variance_list.append(perf_var)
        mean_in_domain_acc_list.append(mean_in_domain_acc)

        for index, in_domain in enumerate(private_dataset.domain_list):
            in_domain_accs_dict.setdefault(in_domain, []).append(domain_accs[index])

        print(
            log_msg(
                f"The {epoch_index} Epoch: Mean Acc: {mean_in_domain_acc} "
                f"Method: {args.method} Per Var: {perf_var}",
                "TEST",
            )
        )

        if args.csv_log and hasattr(fed_method, "weight_dict"):
            csv_writer.write_weight(fed_method.weight_dict, epoch_index, client_domain_list)

        if wandb is not None:
            domain_metrics = {domain: in_domain_accs_dict[domain][-1] for domain in in_domain_accs_dict}
            wandb.log(domain_metrics, step=epoch_index)
            wandb.log({"mean_in_domain_acc": mean_in_domain_acc}, step=epoch_index)
