import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicRepoScopeTest(unittest.TestCase):
    def test_public_repo_only_exposes_paper_single_domain_datasets(self):
        dataset_dir = REPO_ROOT / "Datasets" / "federated_dataset" / "single_domain"
        dataset_files = {path.stem for path in dataset_dir.glob("*.py") if not path.stem.startswith("__")}
        self.assertEqual(dataset_files, {"cifar10", "cifar100"})

    def test_public_repo_only_exposes_paper_multi_domain_datasets(self):
        dataset_dir = REPO_ROOT / "Datasets" / "federated_dataset" / "multi_domain"
        dataset_files = {path.stem for path in dataset_dir.glob("*.py") if not path.stem.startswith("__")}
        self.assertEqual(dataset_files, {"office31", "officehome"})

    def test_public_repo_only_exposes_paper_backbones(self):
        backbones_init = (REPO_ROOT / "Backbones" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('"fedavg_cifar": FedAvgNetCIFAR', backbones_init)
        self.assertIn('"resnet10": resnet10', backbones_init)
        self.assertIn('"resnet18": resnet18', backbones_init)
        self.assertNotIn("simple_cnn", backbones_init)
        self.assertNotIn("resnet34", backbones_init)
        self.assertNotIn("resnet50", backbones_init)
        self.assertNotIn("fedavg_mnist", backbones_init)

    def test_default_cfg_drops_non_paper_attack_and_ood_sections(self):
        cfg_source = (REPO_ROOT / "utils" / "cfg.py").read_text(encoding="utf-8")
        self.assertNotIn("CFG.attack", cfg_source)
        self.assertNotIn("CFG.OOD", cfg_source)
        self.assertIn("CFG.label_skew = CN()", cfg_source)
        self.assertIn("CFG.domain_skew = CN()", cfg_source)


if __name__ == "__main__":
    unittest.main()
