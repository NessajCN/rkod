use std::{
    env,
    ffi::CString,
    io::{self, Result},
    path::Path,
    ptr::null_mut,
};

use crate::{read_data_from_file, rknn_context, rknn_init};
use libc::{c_char, c_void};
use tracing::error;

pub fn init_model(path: &str) -> Result<()> {
    let mut ctx: rknn_context = 0;
    let model_raw = Box::new(0 as c_char);
    let model = Box::into_raw(Box::new(Box::into_raw(model_raw)));

    // Find absolute path of the model
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = format!("{}", Path::new(&dir).join(path).display());
    let model_path = CString::new(path).unwrap();

    // Load RKNN Model
    let model_len = unsafe { read_data_from_file(model_path.as_ptr(), model) };

    if model_len < 0 {
        error!("Failed to load rknn model");
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Failed to load rknn model",
        ));
    }

    // let model_ptr: *mut c_void = &mut model_path as *mut _ as *mut c_void;
    let c = unsafe {
        rknn_init(
            &mut ctx,
            *model as *mut c_void,
            model_len as u32,
            0,
            null_mut(),
        )
    };

    if c < 0 {
        error!("Failed to init rknn");
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Failed to init rknn",
        ));
    }
    println!("hello: {model_len}, c: {c}");

    Ok(())
}
