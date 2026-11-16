# When can nanoparticles lower the thermal conductivity of epoxy? Code and data

Finite element code (FEniCSx + Gmsh) and data for the article
*"When can nanoparticles lower the thermal conductivity of epoxy? Interfacial
resistance, particle arrangement and hollow fillers"* (O.M. Pinto, J.C.V. Ferreira,
G.A. Libaneo, O.M. Berengue; submitted to the International Journal of Heat and
Mass Transfer).

The code computes the effective thermal conductivity of a matrix with spherical
particles in periodic unit cells, with a temperature jump (Kapitza resistance
R_K) at the particle surface, and compares it with the Hasselman–Johnson model.
Comments and variable names in the scripts are in Portuguese.

## Licences

- Code (all `.py` and `.sh` files): MIT, see `LICENSE`.
- Data (all `.csv` files and the figures in `16_figuras_artigo/figuras`):
  Creative Commons Attribution 4.0 International (CC BY 4.0).

## Environment

FEniCSx/DOLFINx 0.11, Gmsh 4.14, PETSc 3.24, mpi4py, petsc4py, NumPy, SciPy and
Matplotlib (Ubuntu under WSL, packages from the `fenics-packages/fenics` PPA).
The theory scripts (folders 14, 17, 18, 19 and the `teoria*.py` files) need only
NumPy, SciPy and Matplotlib. Meshes are generated with one Gmsh thread
(`--threads 1`) so that results are reproducible. Each script reads and writes
files relative to its own folder; the header of each script gives its usage.

## Contents and article sections

| Folder | Content | Article |
| --- | --- | --- |
| `04_duas_particulas` | Two-field solver, mesh convergence, single sphere vs. Hasselman–Johnson, critical radius (462 cases), particle pairs | Sections 3, 4, 5.1, 6.1 |
| `05_N_particulas` | Clusters of 3 to 8 spheres | Section 6.2 |
| `06_dispersao_aleatoria` | Random dispersions (N = 8, 27, 64), high concentration | Section 6.3, Tables 5, 5b |
| `07_resistencia_interface` | AMM/DMM interfacial resistance | Section 5.2, Table 3 |
| `09_particulas_ocas` | Hollow particles, critical wall thickness, gas rarefaction | Sections 7.1, 7.4, 7.5 |
| `11_muitas_particulas` | Cells with 80 random spheres at the reference loading | Sections 6.3, 7.1–7.3 |
| `12_nanocristalino` | Nanocrystalline and granular particles | Sections 7.2, 7.3 |
| `14_reforcos_teoria` | Analytical supplements: wall size effect, Monte Carlo, size distribution, second-order coefficient | Supplementary S1–S4 |
| `15_bidispersa` | Bidisperse cells (FEM check of the mixture rule) | Supplementary S3 |
| `16_figuras_artigo` | Script that draws the article figures from the CSV files | Figures |
| `17_tc_alfa` | Critical wall thickness against α (theory and FEM) | Section 7.4, Fig. 14 |
| `18_vacancias` | Callaway estimate of vacancy scattering in γ-Al₂O₃ | Supplementary S5 |
| `19_espumas_sinteticas` | Comparison with measurements on syntactic foams | Section 7.1, Table 5c |

Meshes and other large binary files are not stored; every script regenerates them.

## Citation

See `CITATION.cff`. A DOI for each release is provided by Zenodo.
