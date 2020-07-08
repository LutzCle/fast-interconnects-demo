use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use rustacuda::launch;
use rustacuda::memory::DevicePointer;

fn main() -> Result<(), Box<dyn Error>> {
    // Set the vector length.
    const LEN: i32 = 100_000; 

    // Set the GPU kernel parameters.
    let grid: u32 = 160;
    let block: u32 = 1024;
    let shared_memory: u32 = 0;

    // Initialize CUDA.
    let _context = rustacuda::quick_init()?;

    // Load the CUDA module containing the 'vadd' function.
    let module_path = CString::new(env!("CUDA_VADD_PATH"))?;
    let module = Module::load_from_file(&module_path)?;

    // Create a CUDA stream.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate a vector.
    let mut data: Vec<i32> = (0..LEN).collect();

    unsafe {
        // Wrap regular pointer with a DevicePointer.
        // This type transformation allows us to pass host-allocated memory to the GPU.
        let dptr = DevicePointer::wrap(data.as_mut_ptr());

        // Launch the `vadd` GPU kernel.
        launch!(module.vadd<<<grid, block, shared_memory, stream>>>(
            dptr,
            1_u32,
            LEN as usize
        ))?;
    }

    // Wait for the GPU kernel to finish execution.
    stream.synchronize()?;

    // Verify that the result is correct.
    let sum: i64 = data.iter().fold(0, |sum, &x| sum + (x as i64));
    let long_len = LEN as i64;

    assert_eq!(sum, (long_len * (long_len + 1)) / 2);

    Ok(())
}
