from common_imports import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import attacks.attack_handler as attack_handler


def get_attack(cfg):
    attack_name = cfg.get('adversarial_perturbation', {}).get('evasion_attack')
    if attack_name is None:
        return None

    adv_section = cfg.get('adversarial_perturbation', {})
    # Global fields: everything outside adversarial_perturbation
    global_fields = {k: v for k, v in cfg.items() if k != 'adversarial_perturbation'}
    # Shared scalars from adversarial_perturbation (e.g. surrogate_model, target_model)
    adv_globals = {k: v for k, v in adv_section.items() if not isinstance(v, dict)}
    attack_cfg = {**global_fields, **adv_globals, **adv_section.get(attack_name, {})}

    for attack_class in attack_handler.__all_classes__:
        if attack_class.__name__.lower() == attack_name.lower():
            print(f"  Attack handler : {attack_class.__name__}")
            return attack_class(attack_cfg).apply()

    raise Exception(f"{attack_name} not yet implemented")
