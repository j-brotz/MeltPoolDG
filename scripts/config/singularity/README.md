# Build a Singularity container of MeltPoolDG

This projects provides the utilities for creating a Singularity container of the current master of [MeltPoolDG](https://github.com/MeltPoolDG/MeltPoolDG-dev) with all its dependencies.

## Installation

You need to have [Singularity](https://singularity.lbl.gov/all-releases) installed. Then, by running
```
sudo \
SINGULARITYENV_MAKE_JOBS="31" \
SINGULARITYENV_DELETE_CACHE=true \
singularity build MeltPoolDG.simg meltpooldg.def
```
a (ready-only) container `MeltPoolDG.simg` is created. The arguments `SINGULARITYENV_MAKE_JOBS` and `SINGULARITYENV_DELETE_CACHE` are optional, defining the number of make jobs and if you would like to delete cache files (`true` or `false` valid).
The installation will take some time since all components (`Trilinos`, `deal.II`, `adaflo`, `MeltPoolDG`) will be freshly installed. Alternatively, you may use the recipe file `meltpooldg_dealii_docker.def`, where the docker-image of the current `deal.II`-master is used.

If you would like to modify/update components of your container, you may create a writable container using the `--sandbox` argument
```
sudo \
SINGULARITYENV_MAKE_JOBS="31" \
SINGULARITYENV_DELETE_CACHE=true \
singularity build --sandbox MeltPoolDG.img meltpooldg.def
```
or simply convert your read-only container to a writable one by calling
```
sudo singularity build --sandbox MeltPoolDG.img MeltPoolDG.simg
```
Now you have your container ready to run your simulation `test.json` by executing
```
mpirun -np 10 singularity exec MeltPoolDG.simg meltpooldg test.json
```
on the Linux device of your choice (e.g. HPC).
