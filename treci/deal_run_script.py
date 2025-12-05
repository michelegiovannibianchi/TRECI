# Import 
from deal import DataConfig, DEALConfig, FlareConfig, DEAL
import argparse
import os

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--label",  # Sequence of labels associated to the case
  type=str
)

CLI.add_argument(
  "--cutoffs",  # cutoffs for FLARE  
  type=str
)

CLI.add_argument(
  "--deal_thresholds",  # selection threshold for DEAL 
  type=str
)

# parse the command line
args = CLI.parse_args()


if args.label is None:
    raise ValueError("Please provide the labels of the case with --label") 
V_vector_labels = args.label.split(",")

if args.cutoffs is None:  
    raise ValueError("Please provide the cutoffs with --cutoffs")   
cutoffs = [float(c) for c in args.cutoffs.split(",")]

if args.deal_thresholds is None:
    raise ValueError("Please provide the DEAL thresholds with --deal_thresholds")
deal_thresholds = [float(t) for t in args.deal_thresholds.split(",")]

for V in V_vector_labels:
    for cutoff in cutoffs:
        for deal_threshold in deal_thresholds:
          print(f"DEAL for case {V}")
          print(f"  - cutoff={cutoff}, deal_threshold={deal_threshold}")  
          path_DEAL=f"{V}/"
          run_folder=path_DEAL+f"threshold-{deal_threshold:.3f}/cutoff-{cutoff}/"
          os.makedirs(run_folder, exist_ok=True)
          
          # Define Config (uses defaults where not provided)
          data_cfg = DataConfig(files=path_DEAL+"All_Data.xyz")
          deal_cfg = DEALConfig(
              threshold=deal_threshold,
              output_prefix=run_folder+"deal",    
          )
          flare_cfg = FlareConfig(cutoff=cutoff)

          # Instantiate DEAL class
          deal = DEAL(data_cfg, deal_cfg, flare_cfg)

          # Run 
          deal.run()
