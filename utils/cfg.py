from yacs.config import CfgNode as CN

from utils.utils import log_msg


def simplify_cfg(args, cfg):
    dump_cfg = CN()
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    dump_cfg[args.method] = cfg[args.method]
    dump_cfg[args.task] = cfg[args.task]

    if cfg[args.method].global_method in cfg["Sever"]:
        dump_cfg["Sever"] = CN()
        dump_cfg["Sever"][cfg[args.method].global_method] = cfg["Sever"][cfg[args.method].global_method]

    if cfg[args.method].local_method in cfg["Local"]:
        dump_cfg["Local"] = CN()
        dump_cfg["Local"][cfg[args.method].local_method] = cfg["Local"][cfg[args.method].local_method]

    return dump_cfg


def show_cfg(args, cfg, method):
    dump_cfg = CN()
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    dump_cfg[method] = cfg[method]
    dump_cfg[args.task] = cfg[args.task]
    print(log_msg(f"CONFIG:\n{dump_cfg.dump()}", "INFO"))
    return dump_cfg


CFG = CN()

CFG.DATASET = CN()
CFG.DATASET.dataset = "fl_cifar10"
CFG.DATASET.communication_epoch = 2
CFG.DATASET.n_classes = 10
CFG.DATASET.parti_num = 4
CFG.DATASET.online_ratio = 1.0
CFG.DATASET.domain_ratio = 1.0
CFG.DATASET.train_eval_domain_ratio = 0.01
CFG.DATASET.backbone = "resnet18"
CFG.DATASET.pretrained = False
CFG.DATASET.aug = "weak"
CFG.DATASET.beta = 0.5

CFG.label_skew = CN()
CFG.domain_skew = CN()

CFG.OPTIMIZER = CN()
CFG.OPTIMIZER.type = "SGD"
CFG.OPTIMIZER.momentum = 0.9
CFG.OPTIMIZER.weight_decay = 1e-5
CFG.OPTIMIZER.local_epoch = 2
CFG.OPTIMIZER.local_train_batch = 64
CFG.OPTIMIZER.local_test_batch = 64
CFG.OPTIMIZER.val_batch = 64
CFG.OPTIMIZER.local_train_lr = 1e-3

CFG.Sever = CN()

CFG.Local = CN()
CFG.Local.DKDRLocal = CN()
CFG.Local.DKDRLocal.tau = 1.0
CFG.Local.DKDRLocal.beta = 1.0

CFG.FedAVG = CN()
CFG.FedAVG.local_method = "BaseLocal"
CFG.FedAVG.global_method = "BaseSever"

CFG.DKDR = CN()
CFG.DKDR.local_method = "DKDRLocal"
CFG.DKDR.global_method = "DKDRSever"
