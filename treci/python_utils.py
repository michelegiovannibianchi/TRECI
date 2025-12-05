from ase.io import read, write
import numpy as np
import math
from glob import glob
import os
import shutil
from ase import Atoms

def generate_dataset_partition(fraction_4_model, fraction_val):
    """
    This script creates 4 different training/validation partitions of the dataset.
    Each partition will be used to train a different ML-FF.
    Each model will be validated on a fraction the selected partition.
    The training and validation sets are saved in the folders "MLFF1", "MLFF2" etc...
    
    Inputs:
    - fraction_4_model: float, fraction of the dataset that is used for a specific ML-FF
    - fraction_val: float, fraction of the dataset for a specific ML-FF that is used for validation
    """
    # Create 4 different training/validation partitions of the dataset
    # Each partition will be used to train a different ML-FF
   
    train_all=0
    val_all=0

    if fraction_4_model is None:
        #Each model will be trained on a random partition of the dataset, equal to the 70% of the total dataset
        fraction_4_model=0.7
    if fraction_val is None:
        #Each model will be validated on the 15 % the selected partition
        fraction_val=0.15

    for fname in sorted(glob("*.xyz")): 
        #print(fname)

        # Dataset for MLFF1
        data=read(fname, index=':',format='extxyz')

        # indeces of the structures to be extracted from the dataset to create a partition for a specific model
        index_extract=np.random.choice(len(data),size=int(fraction_4_model*len(data)),replace=False)
        extract=[]

        for i in range (0, len(data)):
            if i in index_extract:
                extract.append(data[i])

        # split the extracted structures into training and validation sets
        index_validation = np.random.choice(len(extract),size=int(fraction_val*len(extract)),replace=False)
        index_training = list(set(range(len(extract)))-set(index_validation))

        train=[]
        val=[]
        for i in range (0, len(extract)):
            if i in index_validation:
                val.append(extract[i])
            else:
                train.append(extract[i])
                
        #print('val',len(val),'train',len(train))
        train_all=train_all+len(train)
        val_all=val_all+len(val)
        
        write('train_model_1.xyz',train, format='extxyz',append=True)
        write('val_model_1.xyz',val, format='extxyz',append=True)

        ################################################

        #Repeat the same procedure for MLFF2
        data=read(fname, index=':',format='extxyz')

        index_extract=np.random.choice(len(data),size=int(fraction_4_model*len(data)),replace=False)
        extract=[]

        for i in range (0, len(data)):
            if i in index_extract:
                extract.append(data[i])

        index_validation = np.random.choice(len(extract),size=int(fraction_val*len(extract)),replace=False)
        index_training = list(set(range(len(extract)))-set(index_validation))

        train=[]
        val=[]
        for i in range (0, len(extract)):
            if i in index_validation:
                val.append(extract[i])
            else:
                train.append(extract[i])
                
        #print('val',len(val),'train',len(train))
        write('train_model_2.xyz',train, format='extxyz',append=True)
        write('val_model_2.xyz',val, format='extxyz',append=True)

        ################################################

        #Repeat the same procedure for MLFF3
        data=read(fname, index=':',format='extxyz')

        index_extract=np.random.choice(len(data),size=int(fraction_4_model*len(data)),replace=False)
        extract=[]

        for i in range (0, len(data)):
            if i in index_extract:
                extract.append(data[i])

        index_validation = np.random.choice(len(extract),size=int(fraction_val*len(extract)),replace=False)
        index_training = list(set(range(len(extract)))-set(index_validation))

        train=[]
        val=[]
        for i in range (0, len(extract)):
            if i in index_validation:
                val.append(extract[i])
            else:
                train.append(extract[i])
                
        #print('val',len(val),'train',len(train))
        write('train_model_3.xyz',train, format='extxyz',append=True)
        write('val_model_3.xyz',val, format='extxyz',append=True)

        ################################################

        #Repeat the same procedure for MLFF4
        data=read(fname, index=':',format='extxyz')

        index_extract=np.random.choice(len(data),size=int(fraction_4_model*len(data)),replace=False)
        extract=[]

        for i in range (0, len(data)):
            if i in index_extract:
                extract.append(data[i])

        index_validation = np.random.choice(len(extract),size=int(fraction_val*len(extract)),replace=False)
        index_training = list(set(range(len(extract)))-set(index_validation))

        train=[]
        val=[]
        for i in range (0, len(extract)):
            if i in index_validation:
                val.append(extract[i])
            else:
                train.append(extract[i])
                
        #print('val',len(val),'train',len(train))
        write('train_model_4.xyz',train, format='extxyz',append=True)
        write('val_model_4.xyz',val, format='extxyz',append=True)

    print(f'Number of configurations for each ML model:\n')
    print(f"Train: {train_all}")
    print(f"Validation: {val_all}\n\n")

    ####################################################################

    # Move the training and validation sets to the corresponding folders of the different ML potentials

    print("Moving the training and validation sets\n")
    # From ASE version 3.23.0b1, using energy_key 'energy' and forces_key 'forces' is no longer safe when 
    # communicating between MACE and ASE
    # We replace 'energy' with 'DFT_energy' and 'forces' with 'DFT_forces'

    #MLFF1
    os.system("sed -i 's/energy/DFT_energy/g' train_model_1.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' train_model_1.xyz")
    os.system("sed -i 's/energy/DFT_energy/g' val_model_1.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' val_model_1.xyz")
    os.system('mv train_model_1.xyz ../MLFF1')
    os.system('mv val_model_1.xyz ../MLFF1')

    #MLFF2
    os.system("sed -i 's/energy/DFT_energy/g' train_model_2.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' train_model_2.xyz")
    os.system("sed -i 's/energy/DFT_energy/g' val_model_2.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' val_model_2.xyz")
    os.system('mv train_model_2.xyz ../MLFF2')
    os.system('mv val_model_2.xyz ../MLFF2')

    #MLFF3
    os.system("sed -i 's/energy/DFT_energy/g' train_model_3.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' train_model_3.xyz")
    os.system("sed -i 's/energy/DFT_energy/g' val_model_3.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' val_model_3.xyz")
    os.system('mv train_model_3.xyz ../MLFF3')
    os.system('mv val_model_3.xyz ../MLFF3')

    #MLFF4
    os.system("sed -i 's/energy/DFT_energy/g' train_model_4.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' train_model_4.xyz")
    os.system("sed -i 's/energy/DFT_energy/g' val_model_4.xyz")
    os.system("sed -i 's/forces/DFT_forces/g' val_model_4.xyz")
    os.system('mv train_model_4.xyz ../MLFF4')
    os.system('mv val_model_4.xyz ../MLFF4')

    print("Terminated !")
    return

def create_mace_train_scripts(num_models, template_script, output_prefix):
    """
    This script creates the sbatch scripts for the training of the ML potential with MACE.
    The sbatch scripts are saved in the folder "MLFF1", "MLFF2" etc...

    Inputs:
    - num_models: int, number of ML models to be trained (default: 4)
    - template_script: str, name of the template script (default: sbatch_train_mace_template)
    - output_prefix: str, prefix of the output script (default: sbatch_train_mace_model_)
    """
    if num_models is None:
        num_models=4    

    if template_script is None:
        template_script='sbatch_train_mace_template'

    if output_prefix is None:
        output_prefix='sbatch_train_mace_model_' 
        
    print("Creating the sbatch scripts for the training of the ML potentials:\n")   
    for i in range(1, num_models+1):
        seed = 240 + 60*i
        print(f"Model {i} with seed {seed}")

        # Read the template
        with open(template_script, 'r') as f:
            content = f.read()

        # Replace placeholders
        content = content.replace('NUM_MODEL', str(i))
        content = content.replace('SEED_NUMBER', str(seed))
        
        # Create output directory
        out_dir = f'MLFF{i}'
        os.makedirs(out_dir, exist_ok=True)

        # Write the customized file
        out_file = os.path.join(out_dir, f'{output_prefix}{i}')
        with open(out_file, 'w') as f:
            f.write(content)
    return

def compute_solvent_density_profile(MD_file,format,sigma,z_surf,n_bin):
    """
    This script computes the density profile of the water solvent along the z direction via Kernel Density Estimation (KDE).
    The density profile is computed for each atomic species present in the system and with respect to the surface of the metal.
    
    Inputs:
        - MD_file: string, file containing the MD trajectory    
        - format: str, ase format of the MD trajectory 
        - sigma: float, the standard deviation of the Gaussian kernel used for the KDE
        - z_surf: float, z position of the metal surface with respect to the center of the slab (in Angstrom)   
        - n_bin: int, number of grid points
 
    Outputs:
        - bin_center: array, centers of each bin of the density profile   
        - density_O_g_cm3: array, density profile of oxygen in g/cm^3
        - density_H_g_cm3: array, density profile of hydrogen in g/cm^3
        """     
    
    if MD_file is None:
        raise ValueError("MD_file not provided")
    if format is None:
        raise ValueError("format not provided")
    if sigma is None:
        sigma=0.06
    if z_surf is None:
        raise Exception("z_surf not provided, density profile will be computed with respet to the center of the slab")
    if n_bin is None:
        n_bin=200

    #Read MD
    MD=read(MD_file, index=':',format=format)

    n_atoms=len(MD[-1])
    n_frame=len(MD)
    
    box=MD[1].get_cell()
    a_cell=box[0,0]
    b_cell=box[1,1]
    c=box[2,2]#side of the domain in z direction
    
    count_O_kde=np.zeros(n_bin-1)
    count_H_kde=np.zeros(n_bin-1)
    count_Cu_kde=np.zeros(n_bin-1)

    for snap in MD:
        snap.wrap()
        pos=snap.get_positions()

        # Transform positions to have the center slab in z=0
        pos=pos-[0,0,c/2]*np.ones((len(pos),1))
        
        # I have double symmetric interfaces, so I exploit the symmetry to double statistics
        pos=abs(pos)
        # Transform positions to have the surface of Cu in z=0 
        pos=pos-[0,0,z_surf]*np.ones((len(pos),1))
        
        # Count number of atoms of each species
        symbol=snap.get_chemical_symbols()
        atomic_number=snap.get_atomic_numbers()
        n_O=len(np.argwhere(atomic_number==8))
        #n_Cu=len(np.argwhere(atomic_number==29))
        n_H=len(np.argwhere(atomic_number==1))
        
        zO=np.zeros(n_O)
        iO=0
        zH=np.zeros(n_H)
        iH=0
                

        for i in range(0,len(snap)):
                if symbol[i]=='O':
                    zO[iO]=pos[i,2]
                    iO=iO+1
                if symbol[i]=='H':
                    zH[iH]=pos[i,2]
                    iH=iH+1
                    
        
        hist,center=kde_histogram(zO,[0,c/2-z_surf],nbins=n_bin,sigma=sigma)
        count_O_kde=count_O_kde+hist
        
        hist,center=kde_histogram(zH,[0,c/2-z_surf],nbins=n_bin,sigma=sigma)
        count_H_kde=count_H_kde+hist

    # Compute volumes
    len_bin=center[1]-center[0] #width of each bin
    #print('spatial resolution:',len_bin,'A')
        
    Volume_bin_kde=a_cell*b_cell*(center[1]-center[0])   

    # For oxygen
    count_O_kde=(count_O_kde/n_frame)
    density_O_kde=count_O_kde/(2*Volume_bin_kde)# molecule/A^3, factor 2 stands for considering that I am considering at the same time the solvnet in the rion above and below the slab
    mass_O=16/6.022e23  #g atomic mass of Oxygen
    density_O_g_cm3_kde=density_O_kde*mass_O/((1e-8)**3)# from molecule/A^3 to g/cm3

    # For hydrogen
    count_H_kde=(count_H_kde/n_frame)
    density_H_kde=count_H_kde/(2*Volume_bin_kde)#molecule/A^3
    mass_H=1.007/6.022e23  # g atomic mass of protium
    density_H_g_cm3_kde=density_H_kde*mass_H/((1e-8)**3)# from molecule/A^3 to g/cm3

    return center, density_O_g_cm3_kde, density_H_g_cm3_kde
    
def kde_histogram(x,limits,nbins,sigma=None):
    """ This function computes a histogram of the data in x using a Gaussian kernel density estimation (KDE) approach.
    Inputs: 
        - x: data to be histogrammed
        - limits: limits of the histogram [min,max]     
        - nbins: number of bins
        - sigma: standard deviation of the Gaussian kernel"""
    if sigma==None:
        sigma=(limits[1]-limits[0])/nbins/2
    grid = np.linspace(limits[0],limits[1],nbins)
    centers = (grid[1:]+grid[:-1])/2
    
    hist = np.zeros_like(centers)
    
    for xx in x:
        hist+=np.exp(-(xx-centers)**2/(2*sigma**2))#/(np.sqrt(2*math.pi*sigma**2))
    norm=sum(hist)#scipy.integrate.trapezoid(hist,centers)  
    
    return len(x)*hist/norm,centers

def create_DFT_scripts(num_configurations, case, template_script, path_poscar, python_calc_file, customize_potential,V_case=None,extra_elec=None,C_guess=None):
    """
    This script creates the sbatch scripts for the DFT single-points.
    The sbatch scripts are saved in folders "case_0', 'case_1' etc...

    Inputs:
        - num_configurations: int, number of configurations to label
        - case: string, it identifies the case (e.g., "PZC", "m05" etc...)
        - template_script: string, name of the template script (default: sbatch_vasp_ase_template)
        - path_poscar: string, path to the POSCAR files for the DFT single-points
        - python_calc_file: string, path to the python file containing the VASP calculator
        - customize_potential: bool, logical switch to specify extra parameters for constant-potential calculation (default: False)
        - V_case: array, of target potential values that will be computed for the same geometry via double reference method (required only if customize_potential=True)
        - extra_elec: float, initial guess for the extra electron charge [e] to add to the system (required only if customize_potential=True)
        - C_guess:    float, initial guess for the capacitance [e/(V A^2)] (required only if customize_potential=True) 
    """
    if num_configurations is None:
        raise ValueError("num_configurations not provided")    
    
    if case is None:
        raise ValueError("case not provided")

    if template_script is None:
        template_script='sbatch_vasp_ase_template'

    if path_poscar is None:
        raise ValueError("path_poscar not provided") 
        
    if python_calc_file is None:
        raise ValueError("python_calc_file not provided")
    
    if customize_potential is None:
        customize_potential=False

    if customize_potential is True:
        if V_case is None:
            raise ValueError("V_case not provided")
        
    
    for i in range(0, num_configurations):
        
        # Read the template
        with open(template_script, 'r') as f:
            content = f.read()

        # Replace placeholders
        content = content.replace('NUMBER', str(i))
        content = content.replace('VCASE',str(case))
        content = content.replace('POSCAR_PATH', path_poscar)
        
        # Create output directory
        out_dir = f'{case}/case_{i}/'
        os.makedirs(out_dir, exist_ok=True)

        # Write the customized file
        out_file = os.path.join(out_dir, 'sbatch_vasp_ase')
        with open(out_file, 'w') as f:
            f.write(content)

        # Read the python file with the VASP calculator
        with open(python_calc_file, 'r') as f:
            content = f.read()

        # Customize setting for constant-potential calculation, if needed
        if customize_potential is True:
                content = content.replace('V_VECTOR', str(V_case))
                content = content.replace('EXTRA_ELEC',str(extra_elec))
                content = content.replace('C_START', str(C_guess))

        
        # Write the python file in the output directory
        # Extract name of the pyton file from the full path provided
        last_folder = os.path.basename(python_calc_file)
        out_file = os.path.join(out_dir, python_calc_file)
        with open(out_file, 'w') as f:
            f.write(content)
    return

def create_Franken_train_scripts(V_cases,N_RF, backbone_path, template_sbatch_script, template_python_API):
    """   
    This script creates the sbatch script for the training via Franken transfer learning.
    It also creates the python script to run the training via the Franken API

    Inputs:
    	- V_cases: array, target potentials to be trained, e.g. V_cases=[“m05”,”m075”,”m1”]
    	- N_RF: array, number of Random Features to be used for the training
    	- backbone_path: str, path to the backbone model to be used for the transfer learning (default: MACE-L0)
    	- template_sbatch_script: str, name of the template sbatch script (default: sbatch_Franken_multiple_pot_train_template)
 	    - template_python_API: str, name of the template python script for the training via the Franken API (default: run_train_Franken_API_template.py)
    """
    
    if V_cases is None:
        raise ValueError("V_cases not provided")  
    
    if N_RF is None:
        raise ValueError("N_RF not provided")

    if backbone_path is None:
        backbone_path="MACE-L0"
        print(f"backbone_path not provided, using default: {backbone_path}")

    if template_sbatch_script is None:
        template_script='sbatch_Franken_multiple_pot_train_template'

    if template_python_API is None:
        template_python_API='run_train_Franken_API_template.py'
        
    print("Creating the scripts for the training of the ML potentials:\n")   
    
    V_labels=convert_V_to_label(V_cases)
    # Read the template
    with open(template_sbatch_script, 'r') as f:
            content = f.read()

    V_sequence=' '.join(map(str, V_labels))
    rf_sequence=' '.join(map(str, N_RF))
    
    # Replace placeholders
    content = content.replace('V_CASES', V_sequence)
    content = content.replace('RF_CASES', rf_sequence)
        
    # Write the customized file
    out_file = "sbatch_Franken_multiple_pot_train"
    with open(out_file, 'w') as f:
        f.write(content)

    # Read the template of the python script for the training via the Franken API
    with open(template_python_API, 'r') as f:
            content = f.read()  

    # Replace placeholders
    content = content.replace('BACKBONE_PATH', str(backbone_path))

    # Write the customized file
    out_file = "run_train_Franken_API.py"   
    with open(out_file, 'w') as f:
        f.write(content)
    
    print("Done")
    return
    

def prepare_multiple_potential_dataset(V_vector,fraction_val):
    """
    This script creates the ext-xyz files for the training and validation set of Franken multiple potential.
    The files are created in the "Training" folder.

    Inputs:
        - V_vector: array, value of target potential to be trained, e.g. [-0.5,-0.75,-1.0]
        - fraction_val: float, fraction of the dataset used for validation (default: 0.15)
    """
    if V_vector is None:
        raise ValueError("Target potentials not provided")
    
    if fraction_val is None:
        fraction_val=0.15
    
    # Convert the potential value in label
    V_vector_labels=convert_V_to_label(V_vector)
    
    for V,head_folder in zip(V_vector,V_vector_labels):

        print(f"Creating xyz files for V = {V} V\n")
        train=[]
        valid=[]
        
        path=head_folder+"/"

        for fname in sorted(glob(path+"*.xyz")): 

                traj = read(fname,index=':',format='extxyz') 

                # Add tag with info on the head name
                for frame in traj:
                    frame.info["head"]=head_folder
                            
                # split the extracted structures into training and validation sets
                index_validation = np.random.choice(len(traj),size=int(fraction_val*len(traj)),replace=False)
                index_training = list(set(range(len(traj)))-set(index_validation))

                train.extend([traj[i] for i in index_training])
                valid.extend([traj[i] for i in index_validation])

        print("Train_{}_union_with_head.xyz".format(head_folder),'Configurations: ', len(train))
        write("Train_{}_union_with_head.xyz".format(head_folder),train,format='extxyz') 
        print("Valid_{}_union_with_head.xyz".format(head_folder),'Configurations: ', len(valid),'\n\n') 
        write("Valid_{}_union_with_head.xyz".format(head_folder),valid,format='extxyz')
         
    return

def create_lammps_files(case, potential_file, initial_geometry_file, sbatch_template, input_template):
    """
    This script creates the sbatch script and the LAMMPS input script to run a MD simulation with LAMMPS.
    The scripts are saved in the folder {case}.
    
    Inputs:
    	- case: str, label of the case (e.g. "PZC", "m05", "m075" ...)
    	- potential_file: str, path to the potential file of the ML-FF
    	- initial_geometry_file: str, path to the initial geometry file in LAMMPS format
    	- sbatch_template: str, name of the template sbatch script (default: sbatch_lammps_mace_template)
        - input_template: str, name of the template LAMMPS input script (default: MACE_Cu_111_H2O_VCASE_lammps.in)
    """
    if case is None:
        raise ValueError("case not provided")    

    if potential_file is None:
        raise ValueError("potential_file not provided") 

    if initial_geometry_file is None:
        raise ValueError("initial_geometry_file not provided") 

    if sbatch_template is None:
        sbatch_template='sbatch_lammps_template'

    if input_template is None:
        input_template='MACE_Cu_111_H2O_VCASE_lammps.in' 

    print(f"Creating the scripts for the MD simulation of case {case}\n")   

    # Read the template of the sbatch script
    with open(sbatch_template, 'r') as f:
        content = f.read()

    # Replace placeholders
    content = content.replace('INPUT_TEMPLATE', input_template)
    content = content.replace('VCASE', str(case))
    
        
    # Create output directory
    out_dir = f'{case}/'
    os.makedirs(out_dir, exist_ok=True)

    # Write the customized sbatch file
    out_file = os.path.join(out_dir, 'sbatch_lammps_mace')
    with open(out_file, 'w') as f:
        f.write(content)

    # Read the template of the LAMMPS input script
    with open(input_template, 'r') as f:
        content = f.read()  

    # Replace placeholders
    content = content.replace('POTENTIAL_FILE', potential_file)
    content = content.replace('VCASE', str(case))

    # Write the customized LAMMPS input file
    name_input_file = input_template.replace('VCASE', str(case))
    out_file = os.path.join(out_dir, name_input_file)
    
    with open(out_file, 'w') as f:
        f.write(content)

    # Copy the initial geometry file to the output directory
    os.system(f"cp {initial_geometry_file} {out_dir}/Cu_H2O_{case}.data")

    return

def convert_V_to_label(V_vector):

    """ Function to convert the numeric format of the applied potential to a label.
        Convert the sign of the potentials to a single letter prefix:
        V > 0 -> 'p'
        V < 0 -> 'm'

        example V = -0.5 V => m05
        
        Input:
            -V_vector: list of potential values         
        Output:
            -V_vector_labels: list of associated labels"""

    n_V = len(V_vector)

    V_vector_labels=[]
    for V in V_vector:
        mantissa = str(abs(V)).split('.')[0]
        digits = str(abs(V)).split('.')[1]
        if digits == '0':
            digits = ''
        if V > 0:
            V_vector_labels.append('p'+mantissa+digits)
        else:
            V_vector_labels.append('m'+mantissa+digits)

    return V_vector_labels
    
def extract_data_from_multiple_potential_dataset(case,V_to_do,round,folder_input,custom_output_file,folder_output):
    """ 
        Function extracting the data from the different folders (case_xx) generated by the DFT, 
        separating the info of the different target potentials and moving xyz files in the dataset
        folders for the next round of active learning

        Inputs:
            - case: str, label identify the potential value folder of the DFT calculations
            - V_to_do: array, values of the target potential for which one wants to extract the data
            - round: int, number identifying the current round of active learning
            - folder_input: str, path of the folder where there are the DFT results
            - custom_output_file: str, attribute to add to the name of the xyz file
            - folder_output: str, path of the folder where the xyz will be saved
     """

    if case is None:
        raise ValueError("case not provided")

    if V_to_do is None:
        raise ValueError("V_to_do not provided")
    
    if round is None:
        raise ValueError("round not provided")
    
    if folder_input is None:
        raise ValueError("folder_input not provided")
    
    if folder_output is None:
        raise ValueError("folder_output not provided")
    
    if custom_output_file is None:
        raise ValueError("custom_output_file not provided")
    
    V_to_do_labels=convert_V_to_label(V_to_do)

    for V_to_do_label,V in zip(V_to_do_labels,V_to_do):

        print(f"Extract data for V = {V} V")
        #extract the numberic label that identify the computations
        # e.g. case_1 > 1 
        dataset=[]
        for path in sorted(glob(f'{folder_input}/case_*/case_*/')):
            
            try: 
                snap_charge=read(path+f'OUTCAR_{V_to_do_label}.xyz',format='extxyz') #read only last configurations
                n_atom=len(snap_charge)
                symbol=snap_charge.get_chemical_symbols()
                energy=snap_charge.get_total_energy()
                force=snap_charge.get_forces()
                charge=snap_charge.get_initial_charges()

                # Create a new atom objects with only the attribute needed for the training (energy, forces, bader charge)
                snap_new=Atoms(symbols=symbol,
                        positions=snap_charge.get_positions(),
                        cell=snap_charge.get_cell(),
                        calculator=snap_charge.get_calculator(),
                        pbc=[1, 1, 1])
                snap_new.calc.results['energy']=energy
                snap_new.calc.results["forces"]=force
                snap_new.arrays["initial_charges"]=charge
                dataset.append(snap_new)
                
            except FileNotFoundError:
                print(f"File at {V_to_do_label} does not exist in {path}\n")
        
        write(f"{folder_input}/dataset_{V_to_do_label}_geom_{case}.xyz",dataset,format="extxyz")
        print(f'Number of configurations  for {V_to_do_label}: {len(dataset)}')
        os.makedirs(f"{folder_output}/{V_to_do_label}/",exist_ok=True)
        shutil.copy(f"{folder_input}/dataset_{V_to_do_label}_geom_{case}.xyz", f"{folder_output}/{V_to_do_label}/dataset_{custom_output_file}_{V_to_do_label}_geom_{case}_round_{round}.xyz")

    # Repeat the same procedure to extract the data labeled at the PZC in the auxiliary system
    print(f"Extract data for PZC")
    dataset=[]
    for path in sorted(glob(f'{folder_input}/case_*/case_*/')):

        try: 
            snap=read(path+f'OUTCAR_neutral_no_vaccum.xyz',format='extxyz') #read only last configurations
            n_atom=len(snap)
            symbol=snap.get_chemical_symbols()
            energy=snap.get_total_energy()
            force=snap.get_forces()
            charge=snap.get_initial_charges()

            # Create a new atom objects with only the attribute needed for the training (energy, forces, bader charge)
            snap_new=Atoms(symbols=symbol,
                    positions=snap_charge.get_positions(),
                    cell=snap_charge.get_cell(),
                    calculator=snap_charge.get_calculator(),
                    pbc=[1, 1, 1])
            snap_new.calc.results['energy']=energy
            snap_new.calc.results["forces"]=force
            snap_new.arrays["initial_charges"]=charge
            dataset.append(snap_new)
            
        except FileNotFoundError:
            print(f"File at PZC does not exist in {path}\n")
    
    write(f"{folder_input}/dataset_PZC_geom_{case}.xyz",dataset,format="extxyz")
    print(f'Number of configurations for PZC: {len(dataset)}\n')
    os.makedirs(folder_output+"/PZC_label/", exist_ok=True)
    shutil.copy(f"{folder_input}/dataset_PZC_geom_{case}.xyz", f"{folder_output}/PZC_label/dataset_{custom_output_file}_PZC_geom_{case}_round_{round}.xyz")
    
    return 
