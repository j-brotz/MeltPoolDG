# ⬇️ Installation

## Prerequisites

### Download

Download the repository via
```bash
git clone git@github.com:MeltPoolDG/MeltPoolDG-dev.git
```

### Installation of CMake

Make sure that you install CMake >= 3.17.0 either via the package manager of your linux distribution or downloading from https://cmake.org.

### Installation of numdiff

For our testing infrastructure we rely on the tool `numdiff`. Usually, it can installed via the package manager of your linux distribution, e.g. for Ubuntu systems
```bash
sudo apt-get install numdiff
```

### Installation of clang-format

We rely on clang-format 11. In case, your OS does not support this version, it can be installed manually by executing

```bash
bash path_to_dealii/dealii/contrib/utilities/download_clang_format
```
You could add it to your `~/.bashrc` file as follows
```bash
export PATH="$PATH:path_to_dealii/dealii/contrib/utilities/programs/clang-11/bin"
```

## Full installation of MeltPoolDG including dependencies (recommended)
To perform a full installation of MeltPoolDG including the dependencies, simply execute
```bash
cd MeltPoolDG-dev
bash scripts/config/install.sh
```
and follow the command line instructions. In the default case, the following folder structure will be generated:
```bash
external_libs/
├─ dealii-build/
├─ p4est-install/
├─ trilinos-install/
├─ adaflo/
│  ├─ build_release/
│  ├─ build_debug/
│  ├─ include/
MeltPoolDG-dev/
├─ build_release/
```
MeltPoolDG will be built in release mode into `MeltPoolDG-dev/build_release`.

## Manual installation

### Installation of dependencies

To install the dependencies (p4est, Trilinos, deal.II and adaflo), first change to your target folder:
```bash
cd ..
mkdir -p external_libs
cd external_libs
```
Then, run the installtion script by passing the number of processes as command line arguments:
```bash
# install the dependencies with 8 processes
bash ../MeltPoolDG-dev/scripts/config/download_and_install_dependencies.sh 8
```
### Building MeltPoolDG

Given a full installation of dependencies into e.g. `external_libs`, MeltPoolDG can be built in release mode by executing `scripts/config/install_meltpooldg.sh`

```bash
# install MeltPoolDG with e.g. 8 processes
scripts/config/install_meltpooldg.sh 8 external_libs/dealii-build external_libs/adaflo/include external_libs/adaflo/build_release
```

This installs MeltPoolDG into a build folder `build_release`. Alternatively, one can execute the following commands manually. First, configure CMake

```bash
mkdir build
cd build
cmake \
	-DADAFLO_LIB=external_libs/adaflo/build_release \ # change to external_libs/adaflo/build_debug for debug mode
	-DADAFLO_INCLUDE=external_libs/adaflo/include \
	-DCMAKE_BUILD_TYPE=Release \ # change to Debug for debug mode
	-DDEAL_II_DIR=external_libs/dealii-build \
	-DBUILD_SHARED_LIBS=true \
	-DCMAKE_CXX_COMPILER=mpicxx \
	..
```
and subsequently, build MeltPoolDG with e.g. 8 processes by executing
```bash
make -j8
```
