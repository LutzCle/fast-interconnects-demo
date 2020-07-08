use cc;

use std::env;

fn main() {
    // Set a host compiler version that is supported by CUDA
    env::set_var("CXX", "g++-7");

    // Build and link the CUDA kernel as a static library
    cc::Build::new()
        .cuda(true) // Build with NVCC
        .flag("-fatbin") // Tell NVCC to build a fatbin
        .file("cudautils/vadd.cu") // Add the CUDA file
        .debug(false) // Debug enabled slows down mem latency by 10x
        .opt_level(3) // Set -O3
        .cargo_metadata(false) // Don't link target as a library
        .no_default_flags(true) // Don't automatically add any other flags
        .compile("vadd.fatbin");

    // Get output directory from Cargo
    let out_dir = env::var("OUT_DIR").unwrap();

    // Define path of CUDA binary for compiler
    println!("cargo:rustc-env=CUDA_VADD_PATH={}/cudautils/vadd.o", out_dir);

    // Link the CUDA runtime library
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
