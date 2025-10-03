# DoubleReference-MLFF
Workflow for developing ML force-fields for metal/water interface at “constant potential” via Transfer Learning.

This repository contains the code needed to reproduce the workflow presented in:

xxxxxxxxxxxxxxx

# Introduction

# Contents
```text
├── PZC                   # Workflow to develop a ML force-field for metal/water interface at PZC 
│   ├── Train
│   ├── MD
│   ├── DEAL
│   └── DFT
├── Constant_potential    # Workflow to develop ML force-fields for metal/water interface at "constant potential" 
│   ├── Train
│   ├── MD
|   ├── DEAL
│   └── DFT
└── workflow_utils        # python modules containing the necessary functions
```

# Requirements
The following software and versions have been used:
- MACE (>=v0.3.13 from [ACEsuit/MACE](https://github.com/ACEsuit/mace) with multi-GPU training support)
- Franken (>=v0.4.0 from [CSML-IIT-UCL/franken](https://github.com/CSML-IIT-UCL/franken/))
- LAMMPS with support of [MACE](https://github.com/ACEsuit/lammps)
- DEAL (from [luigibonati/DEAL/new_DEAL](https://github.com/luigibonati/DEAL/tree/new_deal?tab=readme-ov-file))
- FLARE (v1.3.3-fix from [luigibonati/flare](https://github.com/luigibonati/flare/tree/1.3.3-fix) → modified FLARE version with small fixes)
- VASP [package](https://www.vasp.at/)
- DoubleReferenceMethod suite (from [michelegiovannibianchi/FCP-calculator-DoubleReferenceMethod](https://github.com/michelegiovannibianchi/FCP-calculator-DoubleReferenceMethod))
Optional:
- Fessa colormap (from 

# Citing
If you find this library useful, please cite our work using the folowing bibtex entry:
