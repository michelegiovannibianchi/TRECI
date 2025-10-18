from ase.io import read,write
from ase.calculators.vasp import Vasp
from ase.calculators.DoubleReferenceMethod.DoubleReferenceWorkflow_calc import DoubleReferenceWorkflow_PZC
import numpy as np

#################   Definition of the ASE-VASP Calculator to compute the PZC single-point  ###########

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


#################   Start calculation of the PZC workflow   #################

snap=read('POSCAR',format='vasp')

DoubleReferenceWorkflow_PZC(snap, calc_neutral_no_vacuum)

print("Calculation terminated")