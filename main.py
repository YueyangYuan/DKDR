import argparse
import datetime
import os
import socket
import uuid
from argparse import ArgumentParser

from Aggregations import Aggregation_NAMES
from Backbones import get_private_backbones
from Datasets.federated_dataset.multi_domain import (
    get_multi_domain_dataset,
    multi_domain_dataset_name,
)
from Datasets.federated_dataset.single_domain import (
    get_single_domain_dataset,
    single_domain_dataset_name,
)
from Methods import Fed_Methods_NAMES, get_fed_method
from utils.cfg import CFG as cfg, show_cfg, simplify_cfg
from utils.conf import config_path, set_random_seed
from utils.training import train
from utils.utils import ini_client_domain

try:
    import setproctitle
except ImportError:  # pragma: no cover - optional runtime dependency
    setproctitle = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


SUPPORTED_TASKS = ("label_skew", "domain_skew")
SUPPORTED_DATASETS = {
    "label_skew": ("fl_cifar10", "fl_cifar100"),
    "domain_skew": ("Office31", "OfficeHome"),
}


def parse_args():
    parser = ArgumentParser(description="Federated Learning", allow_abbrev=False)
    parser.add_argument("--device_id", type=int, default=1, help="The device id for experiments.")
    parser.add_argument(
        "--task",
        type=str,
        default="domain_skew",
        choices=SUPPORTED_TASKS,
        help="The paper setting to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Office31",
        help="Dataset for the selected paper setting.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="DKDR",
        choices=tuple(Fed_Methods_NAMES),
        help="Federated method name.",
    )
    parser.add_argument(
        "--averaging",
        type=str,
        default="Weight",
        choices=tuple(Aggregation_NAMES),
        help="Aggregation strategy.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--csv_log", action="store_true", default=False, help="Enable csv logging.")
    parser.add_argument("--csv_name", type=str, default=None, help="Optional csv run name.")
    parser.add_argument("--save_checkpoint", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    supported_datasets = SUPPORTED_DATASETS[args.task]
    if args.dataset not in supported_datasets:
        raise ValueError(
            f"{args.task} only supports {supported_datasets} in the simplified public repository, "
            f"but got {args.dataset}."
        )


def init_wandb(args, run_cfg):
    if wandb is None:
        return
    wandb.init(config=run_cfg, project="NIPS2025", name=f"{args.method}-{args.dataset}")


def set_process_title(args):
    if setproctitle is None:
        return
    title = f"{args.method}_{args.task}_{args.dataset}"
    if args.csv_name is not None:
        title = f"{title}_{args.csv_name}"
    setproctitle.setproctitle(title)


def build_private_dataset(args, run_cfg):
    if args.dataset in multi_domain_dataset_name:
        return get_multi_domain_dataset(args, run_cfg)
    if args.dataset in single_domain_dataset_name:
        return get_single_domain_dataset(args, run_cfg)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def prepare_data(args, run_cfg, private_dataset):
    if args.task == "label_skew":
        private_dataset.get_data_loaders()
        return None

    client_domain_list = ini_client_domain(private_dataset.domain_list, run_cfg.DATASET.parti_num)
    private_dataset.get_data_loaders(client_domain_list)
    return client_domain_list


def main(args=None):
    if args is None:
        args = parse_args()
    else:
        validate_args(args)

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    cfg_dataset_path = os.path.join(config_path(), args.task, args.dataset, "Default.yaml")
    cfg.merge_from_file(cfg_dataset_path)

    cfg_method_path = os.path.join(config_path(), args.dataset, args.method + ".yaml")
    if os.path.exists(cfg_method_path):
        cfg.merge_from_file(cfg_method_path)

    cfg.merge_from_list(args.opts)
    run_cfg = simplify_cfg(args, cfg)

    show_cfg(args, run_cfg, args.method)
    init_wandb(args, run_cfg)

    if args.seed is not None:
        set_random_seed(args.seed)

    private_dataset = build_private_dataset(args, run_cfg)
    client_domain_list = prepare_data(args, run_cfg, private_dataset)

    priv_backbones = get_private_backbones(run_cfg)
    fed_method = get_fed_method(priv_backbones, client_domain_list, args, run_cfg)

    set_process_title(args)
    train(fed_method, private_dataset, args, run_cfg, client_domain_list)


if __name__ == "__main__":
    main()
