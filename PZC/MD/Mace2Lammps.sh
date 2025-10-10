# source env

source ~/.bashrc

# Activate the conda environment with MACE installed
ENV_PATH=$1
mamba activate "$ENV_PATH"

# Convert the MACE model to LAMMPS format
python $ENV_PATH/lib/python3.13/site-packages/mace/cli/create_lammps_model.py $2
 
mamba deactivate

echo "LAMMPS model created"