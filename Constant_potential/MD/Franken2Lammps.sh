# source env

source ~/.bashrc

# Activate the conda environment with MACE installed
# Make sure to set the correct path to your Franken environment
ENV_PATH=$1
mamba activate "$ENV_PATH"

# Convert the MACE model to LAMMPS format

#echo "Creating LAMMPS model in $2"

python $ENV_PATH/lib/python3.13/site-packages/franken/calculators/lammps_calc.py --model_path=$2

# Deactivate the conda environment
mamba deactivate

#echo "LAMMPS model created"