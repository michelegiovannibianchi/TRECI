import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
from franken.autotune import autotune
from franken.config import MaceBackboneConfig, GaussianRFConfig, DatasetConfig, SolverConfig, HPSearchConfig, AutotuneConfig, MultiscaleGaussianRFConfig
import os
os.environ["SLURM_NTASKS_PER_NODE"] = "4"

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--Potential_value",  # name on the CLI - potential values for which the training has to be performed
  type=str
)

CLI.add_argument(
  "--RF_number",  # name on the CLI - number of Random Features to be used
  type=int
)


# parse the command line
args = CLI.parse_args()

V = args.Potential_value
n_features = args.RF_number

gnn_config = MaceBackboneConfig(
    path_or_id="MACE-L0", 
    interaction_block=2,
)

rf_config = GaussianRFConfig(
    num_random_features=n_features,
    length_scale=HPSearchConfig(start=4.0, stop=10.0, num=4, scale='linear'),
    rng_seed=42,  # for reproducibility
)

solver_cfg = SolverConfig(
    l2_penalty=HPSearchConfig(start=-10, stop=-7, num=2, scale='log'),  # equivalent of numpy.logspace
    force_weight=HPSearchConfig(start=0.5, stop=0.9, num=2, scale='linear'),  # equivalent of numpy.linspace
)

print(f"###### Start training for: {V} ######\n")

dataset_cfg = DatasetConfig(name=f"Cu_H2O_{V}", train_path=f"Dataset/Train_{V}_union_with_head.xyz", val_path=f"Dataset/Valid_{V}_union_with_head.xyz")

autotune_cfg = AutotuneConfig(
dataset=dataset_cfg,
solver=solver_cfg,
backbone=gnn_config,
rfs=rf_config,
seed=42,
jac_chunk_size=8,
run_dir=f"./results_multi_potential/case_{V}_RF_{n_features}",
)

run_path = autotune(autotune_cfg)
print(f"###### End training for: {V} ######\n")