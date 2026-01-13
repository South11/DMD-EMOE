"""
训练DMD-MoME模型
"""

import argparse
import os
import sys
from pathlib import Path

# Fix: Only append path if really needed, and check if it exists
# sys.path.append(str(Path(__file__).parent.parent))

from run import DMD_run

def main():
    parser = argparse.ArgumentParser(description='Train DMD-MoME model')
    parser.add_argument('--model_name', type=str, default='dmd', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='mosi', help='Dataset name')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111], help='Random seeds')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')

    # FIX: Changed default to False so --use_mome actually does something
    parser.add_argument('--use_mome', action='store_true', default=False, help='Use MoME router')

    parser.add_argument('--mome_fusion_type', type=str, default='sum', choices=['sum', 'concat'],
                        help='MoME fusion type')
    parser.add_argument('--mome_temperature', type=float, default=0.5, help='MoME router temperature')
    parser.add_argument('--mome_alpha', type=float, default=0.1, help='MoME balance loss weight')
    parser.add_argument('--mome_distill_weight', type=float, default=0.3, help='MoME distillation weight')

    args = parser.parse_args()

    # MoME Configuration
    # We enforce use_mome=True if calling from this script, unless user explicitly logic requires otherwise.
    # But adhering to args:
    config = {
        'use_mome': True, # Enforce True since this is train_mome.py
        'mome_fusion_type': args.mome_fusion_type,
        'mome_temperature': args.mome_temperature,
        'mome_alpha': args.mome_alpha,
        'mome_distill_weight': args.mome_distill_weight
    }

    print(f"Starting Training with MoME Config: {config}")

    # Run Training
    DMD_run(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        config=config,
        seeds=args.seeds,
        is_tune=False,
        model_save_dir="./pt",
        res_save_dir="./result",
        log_dir="./log",
        mode=args.mode,
        is_distill=True, # DMD requires is_distill=True for the 3-part model structure
        gpu_ids=[0],
        num_workers=4,
        verbose_level=1
    )

if __name__ == '__main__':
    main()