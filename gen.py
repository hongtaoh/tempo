from sim_engine import generate
import numpy as np
import json
import os
import yaml
import argparse
import glob

def load_config():
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def convert_np_types(obj):
    """Convert numpy types in a nested dictionary to Python standard types."""
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_np_types(obj.tolist())
    else:
        return obj

def dir_has_data(path):
    """Return True if directory exists and contains at least one CSV file."""
    return os.path.isdir(path) and len(glob.glob(os.path.join(path, "*.csv"))) > 0

def generate_split(data_save_dir, params_file, config, cwd):
    """Generate all experiments for one split (train or test)."""
    EXPERIMENT_NAMES = config['EXPERIMENT_NAMES']
    is_train = 'train' in os.path.basename(data_save_dir)
    gen_seed = config['GEN_SEED_TRAIN'] if is_train else config['GEN_SEED_TEST']
    N_VARIANTS = config['N_VARIANTS_train'] if is_train else config['N_VARIANTS_test']

    for i, exp_name in enumerate(EXPERIMENT_NAMES):
        OUTPUT_DIR = os.path.join(cwd, f"{data_save_dir}/exp{i + 1}")

        if dir_has_data(OUTPUT_DIR):
            print(f"  Skipping {OUTPUT_DIR} (already exists)")
            continue

        print(f"  Generating {OUTPUT_DIR} ...")
        true_order_and_stages_dicts = generate(
            experiment_name=exp_name,
            params_file=params_file,
            js=config['JS'],
            rs=config['RS'],
            num_of_datasets_per_combination=N_VARIANTS,
            output_dir=OUTPUT_DIR,
            seed=gen_seed,
            keep_all_cols=False,
        )

        with open(f"{OUTPUT_DIR}/true_order_and_stages.json", "w") as f:
            json.dump(true_order_and_stages_dicts, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument(
        '--mode', choices=['lowdim', 'highdim', 'all'], default='all',
        help="Which data to generate (default: all)"
    )
    args = parser.parse_args()

    config = load_config()
    cwd = os.path.dirname(__file__)

    MODES = {
        'lowdim': {
            'params_file': 'adni_params_ucl_gmm.json',
            'train_dir': './train',
            'test_dir': './test',
        },
        'highdim': {
            'params_file': 'high_dimensional.json',
            'train_dir': './train_highdim',
            'test_dir': './test_highdim',
        },
    }

    to_run = ['lowdim', 'highdim'] if args.mode == 'all' else [args.mode]

    for mode in to_run:
        m = MODES[mode]
        print(f"\n=== Generating {mode} data (params: {m['params_file']}) ===")
        for split_dir in [m['train_dir'], m['test_dir']]:
            split_label = 'train' if 'train' in split_dir else 'test'
            print(f"\n-- {split_label} split -> {split_dir} --")
            generate_split(split_dir, m['params_file'], config, cwd)

    print("\nDone.")
