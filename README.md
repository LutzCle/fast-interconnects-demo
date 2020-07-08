Fast Interconnects Demo
=======================

## What are fast interconnects?

Fast interconnects are a new class of technology that connects GPUs (and other
co-processors) to the system. These interconnects include Compute Express Link,
Infinity Fabric, and NVLink 2.0+. These interconnects have in common that they
provide high bandwidth (more than 50 GB/s) and cache-coherence.

### Ok, so what is this demo about?

In this demo, we show how to use a fast interconnect with CUDA. As a bonus, we
also show how to launch GPU kernels with Rust. After the demo, you should know
the advantages and limitations of all currently available memory allocation
techniques for CUDA.

**IMPORTANT**: Note that this demo requires that the GPU has cache-coherent
access to main-memory. _Some programs are intended to segfault!_

### CUDA

For the CUDA demo, ensure that a recent version of CUDA is installed on you
system.  You can follow the [instructions on the Nvidia
website](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### Rust

For the Rust demo, ensure that a recent version of Rust is installed on your
system. You can install an up-to-date version of Rust using
[rustup](https://rustup.rs):
```sh
curl https://sh.rustup.rs -sSf | sh
```

## Usage

The CUDA programs can be compiled with:

```
nvcc -O3 -o cpu-demo cpu-demo.cu
nvcc -O3 -o ccgpu-demo ccgpu-demo.cu
nvcc -O3 -lnuma -o ccnuma-demo ccnuma-demo.cu
nvcc -O3 -o cuda-demo cuda-demo.cu
```

The Rust program can be compiled with:

```
cargo run --package rust-demo
```
