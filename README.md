# Molecular-Dynamics-Molecular-Cages
> Data analysis methods and part of the raw data for the paper "Supramolecular Proton Conductors Self-Assembled by Organic Cages".

## Methods
The analysis methods are described in the [md_ana.ipynb](https://github.com/Lei-Lei-alpha/Molecular-Dynamics-Molecular-Cages/blob/main/md_ana.ipynb) file.

## Results
The analysis results are available in the cage1 and cage2 folders. The files in the above folders are:
- `sys-pos-1.xyz` the AIMD raw trajectory file generated by CP2K code run on Archer2.
- `cage*_500K.xyz` the trajectory file calibrated by fix the centre of mass of the system.
- `water_com_cage*.xlsx` the centre of masses of water molecules
- `MSD_D_H2O_cage*.xlsx` the mean square displacement and diffusion coefficients of water molecules.
- `hbonds_mic.json` the hydrogen bond analysis results, including the number of H-bonds in each frame, the donor, acceptor, the donor-acceptor separation, the D-H-A angle.

## Raw data
The raw `.xyz` trajectory files are large (several hundreds of MB). Part of the trajectory files (50000 AIMD steps, 5000 frames) have been uploaded into the cage1 and cage2 folders. The raw trajectory files can be provided in reasonable request to reproduce our results.

## Contacts
Please contact [Lei Lei](mailto:Lei.Lei@nottingham.ac.uk) or [Sanliang Ling](mailto:sanliang.ling@nottingham.ac.uk) for further computation details of the simulation results.
