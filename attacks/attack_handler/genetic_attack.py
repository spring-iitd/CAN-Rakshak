import os

from ..Genetic_algorithm.Adversarial_DoS import AdversarialDosAttack
from ..Genetic_algorithm.Adversarial_Fuzzy import AdversarialFuzzyAttack
from ..Genetic_algorithm.Adversarial_Spoof import AdversarialSpoofAttack


class GeneticAdvAttack:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self):
        cfg         = self.cfg
        attack_name = cfg['attack_mode'].lower()
        print(f"  Attack         : Genetic ({attack_name.upper()})")

        model_path = os.path.join(
            cfg['dir_path'], "..", "models",
            cfg['model'] + "_" + cfg['model_name'] + ".h5"
        )
        file_path = os.path.join(
            cfg['dir_path'], "..", "datasets", cfg['dataset_name'],
            "test", cfg['test_dataset_dir'],
            cfg['file_name'][:-4] + "_test_data.npz"
        )
        print("File Path : ", file_path)

        ATTACK_REGISTRY = {
            "dos":   AdversarialDosAttack,
            "fuzzy": AdversarialFuzzyAttack,
            "spoof": AdversarialSpoofAttack,
        }

        AttackClass = ATTACK_REGISTRY.get(attack_name)
        if AttackClass is None:
            raise ValueError(f"Unknown attack mode: {attack_name}")

        attack = AttackClass(
            model_path=model_path,
            file_path=file_path,
            population_size=100,
            max_generations=20,
            mutation_rate=0.4,
        )

        return attack.apply(cfg)
