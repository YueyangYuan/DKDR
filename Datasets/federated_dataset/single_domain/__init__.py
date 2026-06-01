from Datasets.federated_dataset.single_domain.cifar10 import FedLeaCIFAR10
from Datasets.federated_dataset.single_domain.cifar100 import FedLeaCIFAR100
from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset


single_domain_dataset_name = {
    FedLeaCIFAR10.NAME: FedLeaCIFAR10,
    FedLeaCIFAR100.NAME: FedLeaCIFAR100,
}


def get_single_domain_dataset(args, cfg) -> SingleDomainDataset:
    if args.dataset not in single_domain_dataset_name:
        raise ValueError(f"Unsupported single-domain dataset: {args.dataset}")
    return single_domain_dataset_name[args.dataset](args, cfg)
