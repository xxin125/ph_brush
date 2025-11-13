
# MC – Monte Carlo GPU Implement

## Requirements

* NVIDIA CUDA Toolkit (e.g., CUDA 11 or newer)
* `g++` with C++17 support
* `make`
* cuFFT library (included with CUDA)

---

## Build

Run the following command in the project root:

```bash
make
```

The executable will be created at:

```
bin/MC
```

### Optional build options

```bash
make ARCH=sm_86           # specify GPU architecture
make PRECISION=double     # use double/single precision
make debug                # build with debug info
make clean                # remove build files
```

---

## Run

After building, run the program with:

```bash
./bin/MC MC_input.data params.in
```

* `MC_input.data` – data file (lammps format)
* `params.in` – parameter file

Example:

```bash
./bin/MC MC_input.data params.in
```

---
