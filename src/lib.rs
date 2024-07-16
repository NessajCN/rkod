#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!("./bindings.rs");

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use tracing::info;

pub mod od;
pub mod cv;
pub mod upload;

pub fn dump_tensor_attr(attr: &rknn_tensor_attr) {
    let mut name: Vec<u8> = Vec::new();
    for &n in attr.name.iter() {
        if n != 0 {
            name.push(n as u8);
        }
    }
    let name = std::str::from_utf8(&name).expect("Error while parsing input attr name");
    let fmt_string = match attr.fmt {
        _rknn_tensor_format_RKNN_TENSOR_NCHW => "NCHW",
        _rknn_tensor_format_RKNN_TENSOR_NHWC => "NHWC",
        _rknn_tensor_format_RKNN_TENSOR_NC1HWC2 => "NC1HWC2",
        _rknn_tensor_format_RKNN_TENSOR_UNDEFINED => "UNDEFINED",
        _ => "UNKNOWN",
    };

    let type_string = match attr.type_ {
        _rknn_tensor_type_RKNN_TENSOR_FLOAT32 => "FP32",
        _rknn_tensor_type_RKNN_TENSOR_FLOAT16 => "FP16",
        _rknn_tensor_type_RKNN_TENSOR_INT8 => "INT8",
        _rknn_tensor_type_RKNN_TENSOR_UINT8 => "UINT8",
        _rknn_tensor_type_RKNN_TENSOR_INT16 => "INT16",
        _rknn_tensor_type_RKNN_TENSOR_UINT16 => "UINT16",
        _rknn_tensor_type_RKNN_TENSOR_INT32 => "INT32",
        _rknn_tensor_type_RKNN_TENSOR_UINT32 => "UINT32",
        _rknn_tensor_type_RKNN_TENSOR_INT64 => "INT64",
        _rknn_tensor_type_RKNN_TENSOR_BOOL => "BOOL",
        _rknn_tensor_type_RKNN_TENSOR_INT4 => "INT4",
        _ => "UNKNOW",
    };

    let qnt_type = match attr.qnt_type {
        _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE => "NONE",
        _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP => "DFP",
        _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => "AFFINE",
        _ => "UNKNOW",
    };

    info!(
        "index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}], \
    n_elems={}, size={}, fmt={}, type={}, qnt_type={}, \
    zp={}, scale={}",
        attr.index,
        name,
        attr.n_dims,
        attr.dims[0],
        attr.dims[1],
        attr.dims[2],
        attr.dims[3],
        attr.n_elems,
        attr.size,
        fmt_string,
        type_string,
        qnt_type,
        attr.zp,
        attr.scale
    );
}

// The output is wrapped in a Result to allow matching on errors.
// Returns an Iterator to the Reader of the lines of the file.
pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
