import torch
import os
a=torch.cuda.is_available()
#print('Cuda avail=',a)
b=torch.cuda.device_count()
#print('cuda device=',b)

import numpy as np
import argparse
from ase.io import read, write
from ase import Atoms
from mace.calculators import MACECalculator

# == Compute standard deviation of the forces from an ensamble of 4 ML models 

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--DUMP_NAME",  # PATH name of the MD dump file
  type=str
)

CLI.add_argument(
  "--PATH_FF",  # PATH where there are the ML-FFs
  type=str
)

CLI.add_argument(
  "--PREFIX_FF",  # Prefix of the ML-FF files
  type=str
)

# parse the command line
args = CLI.parse_args()

if args.DUMP_NAME is None:
    raise ValueError("Please provide the name of the MD dump file with --DUMP_NAME") 
else:
    dump_name=args.DUMP_NAME

if args.PATH_FF is None:
    raise ValueError("Please provide the path to the ML-FFs with --PATH_FF") 
else:
    path_FF=args.PATH_FF

if args.PREFIX_FF is None:
    raise ValueError("Please provide the prefix of the ML-FF files with --PREFIX_FF") 
else:
    prefix_FF=args.PREFIX_FF
    
# Read the MD trajectory
MD=read(dump_name,index=':', format='lammps-dump-text')
num_frame=len(MD)
n_atom=len(MD[0])

# initialize arrays
# for standard deviation on the force components
std_dev_fx=np.zeros([n_atom,len(MD)])
std_dev_fy=np.zeros([n_atom,len(MD)])
std_dev_fz=np.zeros([n_atom,len(MD)])

# matrices to store the forces from the 4 MLFFs
# 1st index on atoms, 2nd on MLFF number, 3rd on configurations
mat_forcex=np.zeros([n_atom,4,len(MD)])
mat_forcey=np.zeros([n_atom,4,len(MD)])
mat_forcez=np.zeros([n_atom,4,len(MD)])

# Load the 4 ML models
path_FF="../../Training/"
calculator0 = MACECalculator(model_paths=path_FF+f"MLFF1/{prefix_FF}_1_stagetwo_compiled.model", device='cuda',default_dtype="float32")
calculator1 = MACECalculator(model_paths=path_FF+f"MLFF2/{prefix_FF}_2_stagetwo_compiled.model", device='cuda',default_dtype="float32")
calculator2 = MACECalculator(model_paths=path_FF+f"MLFF3/{prefix_FF}_3_stagetwo_compiled.model", device='cuda',default_dtype="float32")
calculator3 = MACECalculator(model_paths=path_FF+f"MLFF4/{prefix_FF}_4_stagetwo_compiled.model", device='cuda',default_dtype="float32")

file=f'{prefix_FF}_with_std_dev.xyz'

for i in range(0,len(MD)):
    frame=MD[i]
    n_atom=len(frame)
    symbol=frame.get_chemical_symbols()
    
    # calculate forces with the 4 MLFFs
    frame.calc=calculator0 
    force0=frame.get_forces()
    energy=frame.get_total_energy()
    
    frame.calc=calculator1 
    force1=frame.get_forces()

    frame.calc=calculator2 
    force2=frame.get_forces()

    frame.calc=calculator3 
    force3=frame.get_forces()

    #Create matrices of forces for the 4 MLFFs and compute standard deviation

    #x-component

    mat_forcex[:,0,i]=force0[:,0]
    mat_forcex[:,1,i]=force1[:,0]
    mat_forcex[:,2,i]=force2[:,0]
    mat_forcex[:,3,i]=force3[:,0]

    std_dev_fx[:,i]=np.std(mat_forcex[:,:,i], axis=1)

    #y-component
    
    mat_forcey[:,0,i]=force0[:,1]
    mat_forcey[:,1,i]=force1[:,1]
    mat_forcey[:,2,i]=force2[:,1]
    mat_forcey[:,3,i]=force3[:,1]

    std_dev_fy[:,i]=np.std(mat_forcey[:,:,i], axis=1)

    #z-component
    
    mat_forcez[:,0,i]=force0[:,2]
    mat_forcez[:,1,i]=force1[:,2]
    mat_forcez[:,2,i]=force2[:,2]
    mat_forcez[:,3,i]=force3[:,2]

    std_dev_fz[:,i]=np.std(mat_forcez[:,:,i], axis=1)
    
    # Save the frame with the std_dev arrays
    #frame.info['energy']=energy
    frame.calc.results['forces']=force0
    frame.set_array('std_dev_fx',std_dev_fx[:,i])
    frame.set_array('std_dev_fy',std_dev_fy[:,i])
    frame.set_array('std_dev_fz',std_dev_fz[:,i])
    
    write(file,frame,format="extxyz",append=True)