# ⬇️ Installation

## Prerequisites

### Download the repository

Clone the repository via:
```bash
git clone git@github.com:MeltPoolDG/MeltPoolDG-dev.git
```

### Installation of CMake

Make sure that you install CMake >= 3.17.0, either via the package manager of your linux distribution or by downloading from https://cmake.org.

### Installation of numdiff

Our testing infrastructure relies on the tool `numdiff`. It is typically available through the package manager of your Linux distribution. For Ubuntu, install it using:
```bash
sudo apt-get install numdiff
```

### Installation of clang-format

We require clang-format 16. If your OS does not support this version, you can install it manually by running:

```bash
bash path_to_dealii/dealii/contrib/utilities/download_clang_format
```
You could add it to your `~/.bashrc` file as follows
```bash
export PATH="$PATH:path_to_dealii/dealii/contrib/utilities/programs/clang-16/bin"
```

## Full installation of MeltPoolDG including dependencies (recommended)
To install MeltPoolDG along with its dependencies, simply run:
```bash
cd MeltPoolDG-dev
bash scripts/config/install.sh
```
Follow the command-line instructions. By default, the following directory tree will be created:
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
MeltPoolDG is compiled in release mode in `MeltPoolDG-dev/build_release` and in debug mode in `MeltPoolDG-dev/build_debug`.

## Manual installation

### Installation of dependencies

To manually install the dependencies (p4est, Trilinos, deal.II and adaflo), first create the target directory:
```bash
cd ..
mkdir -p external_libs
cd external_libs
```
Then, run the installation script with the desired number of processes and build configuration (`Debug`, `Release` or `DebugRelease` [default]):
```bash
# install the dependencies with 8 processes and DebugRelease configuration
bash path_to_meltpooldg/MeltPoolDG-dev/scripts/config/download_and_install_dependencies.sh 8 DebugRelease
```
### Building MeltPoolDG

Once all dependencies are installed (e.g. in `external_libs`), MeltPoolDG can be built in DebugRelease mode using:

```bash
# build MeltPoolDG with 8 processes in DebugRelease configuration and create the build folders insource, indicated by the last argument
cd path_to_meltpooldg/MeltPoolDG-dev
scripts/config/install_meltpooldg.sh 8 external_libs/dealii-build external_libs/adaflo/include external_libs/adaflo/build_release DebugRelease .
```
Alternatively, you can manually configure and build the project using CMake.
Create a build directory and configure CMake:

```bash
cd path_to_meltpooldg/MeltPoolDG-dev
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
Then compile MeltPoolDG using e.g. 8 processes by executing
```bash
make -j8
```
Happy computing :-).
