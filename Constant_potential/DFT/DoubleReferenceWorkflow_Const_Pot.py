from ase.io import read,write
from ase.calculators.vasp import Vasp
from ase.calculators.DoubleReferenceMethod.DoubleReferenceWorkflow_calc import DoubleReferenceWorkflow
import numpy as np

#################   Definition of the different ASE-VASP Calculators to compute the "Double Reference Method"   #################

#### 1) No extra charge + no vacuum
calc_neutral_no_vacuum=Vasp(directory='neutral',
            istart = 0,   #restart from scratch
            icharg = 2,
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1, 
            algo = 'fast',
            ediff = 1E-07,
            nelm = 160,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore  = 1,
            kpar=2,
            npar=2,
            lreal='Auto',
            laechg = True,
            lvhar   = True)

#### 2) No extra charge + vacuum            

####### 2.1) No extra charge + vacuum + no dipole corrections           
calc_neutral_vacuum_no_dipole=Vasp(directory='neutral_vacuum',
            istart = 0,   #restart from scratch
            icharg = 2,
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1, 
            algo = 'N',
            ediff = 1E-07,
            nelm = 400,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore  = 1,
            kpar=2,
            npar=2,
            lreal='Auto')
####### 2.2) No extra charge + vacuum + dipole corrections
calc_neutral_vacuum_dipole=Vasp(directory='neutral_vacuum',
            istart = 1,   #restart 
            icharg = 1,
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1, 
            algo = 'A',               
            ediff = 1E-07,
            nelm = 300,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore  = 1,
            kpar=2,
            npar=2,
            lreal='Auto',
            ldipol  = True, #swith on dipole correction
            idipol  = 3, #in 3rd direction
            dipol   = [0.5, 0.5, 0.5], 
            lvhar   = True, # to print only ionic and hartree potential
            lvacpotav=True )

#### 3) Extra charge + no vacuum
calc_charge=Vasp(directory='charge', 
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1, 
            algo = 'fast',
            ediff = 1E-07,
            nelm = 160,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore = 1,
            kpar=2,
            npar=2,
            lreal='Auto',
            laechg = True,
            lvhar   = True # to print only ionic and hartree potential
            )

#################   Start calculation of the "Double Reference Method" workflow   #################

snap=read('POSCAR',format='vasp')

#    """Function implementing the Double Reference Workflow
#    
#    Input: 
#        - snap: ase atoms, 
#            atomic geometry
#
#        - external_bias_vector: list, 
#            values of applied potential for which the Double Reference Method will be applied
#
#        - calc_neutral_no_vacuum: ase calculator, 
#            calculator for system without a vacuum region and no extra charge
#
#        - calc_neutral_vacuum_no_dipole: ase calculator, 
#            calculator for system with a vaccum region, but not dipole corrections
#
#        - calc_neutral_vacuum_dipole: ase calculator, 
#            calculator for system with a vaccum region, and with dipole corrections
#
#        - calc_charge: ase calculator, 
#            calculator for system without a vacuum region and extra charge
#
#        - guess_extra_electrons: int, 
#            initial guess of extra electrons to add to the system to the first point of the external_bias_vector 
#            default:  0, i.e. start from the neutral system
#
#        - C_guess: float, 
#            initial guess for the capacitance of the interface
#            default: 1/80 e/(V A^2) (same defalt value of the FCP2rm calculator)
#
#        - restart: bool, 
#            if True the calculation will start from the calculation of the system with a vaccum region, and with dipole corrections
#            (suppose that the calculations of the neutral system without a vacuum region and system with a vaccum region, but not dipole corrections have been already performed and the charge densities are available)
#            default: False
#
#        - V_SHE: float, 
#            value of the standard hydrogen electrode potential
#            default: 4.44 V
#        """

DoubleReferenceWorkflow(
                        snap=snap,
                        external_bias_vector=V_VECTOR,
                        calc_neutral_no_vacuum=calc_neutral_no_vacuum,
                        calc_neutral_vacuum_no_dipole=calc_neutral_vacuum_no_dipole,
                        calc_neutral_vacuum_dipole=calc_neutral_vacuum_dipole,
                        calc_charge=calc_charge,
                        guess_extra_electrons=EXTRA_ELEC,
                        C_guess=C_START,
                        restart=False
                        )
print("Calculation terminated")