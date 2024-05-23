// use cc;
// use bindgen;

use std::{env, path::Path};

fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo::rerun-if-changed=librknn_api/aarch64");
    println!("cargo::rerun-if-changed=src/utiles");
    // Link to librknnrt lib
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("librknn_api/aarch64").display());
    // println!("cargo:rustc-link-search=native=/home/nessaj/Projects/rkod/librknn_api/aarch64");
    println!("cargo:rustc-link-lib=dylib=rknnrt");
    cc::Build::new()
        .file("src/utils/file_utils.c")
        .compile("file_utils");

    // Auto generate ffi bindings.
    // let bindings = bindgen::Builder::default()
    //     .header("wrapper.h")
    //     .generate()
    //     .expect("Unable to generate bindings");
    // let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // bindings
    //     .write_to_file(out_path.join("bindings.rs"))
    //     .expect("Couldn't write bindings!");
}