import os
import sys
import warnings
import yaml

# Suppress TensorFlow and absl warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add paths BEFORE any pipeline imports
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def build_config(yaml_cfg):
    """
    Build a flat config dict directly from YAML values.
    Replaces the former populate_config_module() pattern — no intermediary
    config.py or attack_config.py modules are needed.
    All pipeline functions receive this dict as their 'cfg' parameter.
    """
    dp = yaml_cfg.get('dataset_processing', {})
    tr = yaml_cfg.get('training', {})
    ts = yaml_cfg.get('testing', {})
    ap = yaml_cfg.get('adversarial_perturbation', {})
    ad = yaml_cfg.get('adversarial_defense', {})
    da = yaml_cfg.get('data_augmentation', {})
    run_steps = yaml_cfg.get('run_steps', {})

    # Only treat an evasion attack as "active" if Stage 4 is actually enabled.
    _adv_active   = run_steps.get('adversarial_perturbation', False)
    _adv_attack   = ap.get('evasion_attack') if _adv_active else None
    _adv_sub      = ap.get(ap.get('evasion_attack') or '', {}) if _adv_active else {}
    _adv_type     = _adv_sub.get('attack_mode') if _adv_active else None
    



    return {
        # Base path (equivalent to DIR_PATH in former config.py — points to src/)
        'dir_path': SRC_DIR,
        'dataset_path' : os.path.join(SRC_DIR, "..","datasets"),
        # Top-level
        'dataset_name': yaml_cfg.get('dataset_name', ''),
        'file_name':    yaml_cfg.get('file_name',    'can_data_logs.csv'),

        # dataset_processing
        'preprocess':         dp.get('preprocess',        False),
        'split':              dp.get('split',             False),
        'split_mode':         dp.get('split_mode',        'default'),
        'split_ratio':        dp.get('split_ratio',       0.2),
        'feature_extraction': dp.get('feature_extraction', True),
        'feature_extractor':  dp.get('feature_extractor', 'FrameBuilder'),

        # training
        'train_model':       tr.get('model', None),
        'train_model_name':  tr.get('model_name',None),
        'epochs':            tr.get('epochs',3),
        'train_dataset_dir': tr.get('train_dataset_dir', ''),

        # testing
        'test_model':        ts.get('model', None),
        'test_model_name':   ts.get('model_name', None),
        'test_dataset_dir':  ts.get('test_dataset_dir',  ''),

        # adversarial_perturbation — normalize 'evasion_attack' (YAML) → 'adv_attack' (dispatch key)
        'adversarial_perturbation': {**ap, 'adv_attack': _adv_attack},
        'adv_attack':        _adv_attack,
        'adv_attack_type':   _adv_type,
        'evasion_attack':    _adv_attack,
        'surrogate_model': ap.get('surrogate_model', None),
        'target_model': ap.get('target_model', None),
        'attack_mode': ap[ap['evasion_attack']].get('attack_mode',None) if ap.get('evasion_attack', None) else None,
        'model_path': ap[ap['evasion_attack']].get('model_path',None) if ap.get('evasion_attack', None) else None,

        # 'original_tracksheet': ap[ap['evasion_attack']].get('original_tracksheet',None) if ap.get('evasion_attack', None) else None,
        # 'output_dir': ap[ap['evasion_attack']].get('output_dir',None) if ap.get('evasion_attack', None) else None,
        # 'decoded_output_dir': ap[ap['evasion_attack']].get('decoded_output_dir',None) if ap.get('evasion_attack', None) else None,
        # 'prediction_output_dir': ap[ap['evasion_attack']].get('prediction_output_dir',None) if ap.get('evasion_attack', None) else None,
        # 'tracksheet_dir': ap[ap['evasion_attack']].get('tracksheet_dir',None) if ap.get('evasion_attack', None) else None,
        'rounds': ap['BitFlipAttack'].get('rounds',None),
        'original_tracksheet': ap['BitFlipAttack']['attack']['original_tracksheet'],
        'output_dir' : ap['BitFlipAttack']['attack']['output_dir'],
        'decoded_output_dir' : ap['BitFlipAttack']['decode']['decoded_output_dir'],
        'prediction_output_dir': ap['BitFlipAttack']['evaluate']['prediction_output_dir'],
        'tracksheet_dir': ap['BitFlipAttack']['update']['tracksheet_dir'],
        
        # adversarial_defense
        'defense_method':    ad.get('defense_method',    'AdversarialTraining'),
        'adv_examples_path': ad.get('adv_examples_path', None),
        'adv_samples':       ad.get('adv_samples',       800),

        # data_augmentation
        'augmentation': {
            'attack_type':    da.get('attack_type',    'dos'),
            'input_path':     da.get('input_path',     ''),
            'output_path':    da.get('output_path',    ''),
            'payload_mode':   da.get('payload_mode',   'random'),
            'target_id':      da.get('target_id',      None),
            'max_injections': da.get('max_injections', 10),
            'predictor':      da.get('predictor',      {}),
        },
    }


def run_pipeline(yaml_cfg):
    """Build config and run enabled pipeline stages."""
    cfg = build_config(yaml_cfg)

    from preprocessing import preprocess
    from get_extractor import get_extractor
    from get_splitter import get_splitter
    from get_augmentor import get_augmentor
    from train import train_model
    from test import test_model
    from get_attack import get_attack
    from retraining import adversarial_retraining

    run_steps = yaml_cfg.get('run_steps', {})
    dataset_path = os.path.join(PROJECT_ROOT, "datasets", cfg['dataset_name'])

    print()
    print("=" * 55)
    print("  CAN Rakshak Pipeline")
    print("=" * 55)
    print(f"  Dataset       : {cfg['dataset_name']}")
    print(f"  File          : {cfg['file_name']}")
    print(f"  Attack        : {cfg.get('evasion_attack', 'N/A')}")
    print(f"  Extractor     : {cfg['feature_extractor']}")
    print("=" * 55)
    print()

    # Stage 1: Dataset Processing
    if run_steps.get('dataset_processing', False):
        print("[Stage 1/7] Dataset Processing")
        print("-" * 40)
        preprocess(dataset_path)
        extractor = get_extractor(cfg['feature_extractor'], cfg) if cfg.get('feature_extraction', False) else None
        if cfg.get('split', False):
            get_splitter(dataset_path, mode=cfg['split_mode'], feature_extractor=extractor, cfg=cfg)
        print("[Stage 1/7] Done")
        print()

    # Stage 2: Data Augmentation
    if run_steps.get('data_augmentation', False):
        print("[Stage 2/7] Data Augmentation")
        print("-" * 40)
        aug_cfg     = cfg['augmentation']
        input_path  = os.path.join(PROJECT_ROOT, aug_cfg['input_path'])
        output_path = os.path.join(PROJECT_ROOT, aug_cfg['output_path'])
        augmentor   = get_augmentor(aug_cfg['attack_type'], aug_cfg)
        augmentor.augment(input_path, output_path)
        print("[Stage 2/7] Done")
        print()

    # Stage 3: Dataset Analysis
    if run_steps.get('run_analysis', False):
        print("[Stage 3/7] Dataset Analysis")
        print("-" * 40)
        from run_analysis import run_dataset
        run_dataset(cfg['dataset_name'], 'modified_dataset', False)
        print("[Stage 3/7] Done")
        print()

    # Stage 4: Training
    if run_steps.get('training', False):
        print("[Stage 4/7] Training")
        print("-" * 40)
        model_name = cfg['train_model'] + "_" + cfg['train_model_name'] + ".h5"
        model_path = os.path.join(PROJECT_ROOT, "models", model_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        train_model(cfg['train_model'], model_path, cfg)
        print("[Stage 4/7] Done")
        print()

    # Stage 5: Testing
    if run_steps.get('testing', False):
        print("[Stage 5/7] Testing")
        print("-" * 40)
        model_name = cfg['test_model'] + "_" + cfg['test_model_name'] + ".h5"
        model_path = os.path.join(PROJECT_ROOT, "models", model_name)
        test_model(cfg['test_model'], model_path, cfg, adv_attack=cfg['evasion_attack'])
        print("[Stage 5/7] Done")
        print()

    # Stage 6: Adversarial Perturbation
    if run_steps.get('adversarial_perturbation', False):
        print("[Stage 6/7] Adversarial Perturbation")
        print("-" * 40)
        get_attack(cfg)
        print("[Stage 6/7] Done")
        print()

    # Stage 7: Robust Training
    if run_steps.get('robust_training', False):
        print("[Stage 7/7] Robust Training")
        print("-" * 40)
        adv_examples_path = cfg.get('adv_examples_path')
        model_name = cfg['train_model'] + "_" + cfg['train_model_name'] + ".h5"
        model_path = os.path.join(PROJECT_ROOT, "models", model_name)
        adversarial_retraining(model_path, adv_examples_path, cfg, adversarial_samples_limit=cfg['adv_samples'])
        print("[Stage 7/7] Done")
        print()

    print("=" * 55)
    print("  Pipeline Complete")
    print("=" * 55)


def main():
    yaml_path = os.path.join(SRC_DIR, "config.yaml")
    yaml_cfg = load_yaml_config(yaml_path)
    run_pipeline(yaml_cfg)


if __name__ == "__main__":
    main()
