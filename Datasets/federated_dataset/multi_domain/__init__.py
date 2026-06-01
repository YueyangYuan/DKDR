from Datasets.federated_dataset.multi_domain.office31 import FLOffice31
from Datasets.federated_dataset.multi_domain.officehome import FLOfficeHome
from Datasets.federated_dataset.multi_domain.utils.multi_domain_dataset import MultiDomainDataset


multi_domain_dataset_name = {
    FLOffice31.NAME: FLOffice31,
    FLOfficeHome.NAME: FLOfficeHome,
}


def get_multi_domain_dataset(args, cfg) -> MultiDomainDataset:
    if args.dataset not in multi_domain_dataset_name:
        raise ValueError(f"Unsupported multi-domain dataset: {args.dataset}")
    return multi_domain_dataset_name[args.dataset](args, cfg)
