use std::{
    env,
    ffi::CString,
    io::{self, Error, ErrorKind, Result},
    mem::size_of,
    path::Path,
    ptr::null_mut,
};

use crate::{
    _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR, _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
    _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR, _rknn_tensor_format_RKNN_TENSOR_NCHW,
    _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC, _rknn_tensor_type_RKNN_TENSOR_INT8,
    dump_tensor_attr, image_buffer_t, image_format_t, read_data_from_file, read_image,
    rknn_context, rknn_init, rknn_input_output_num, rknn_query, rknn_tensor_attr,
};
use libc::{c_char, c_void};
use tracing::{error, info};

#[derive(Debug, Clone)]
pub struct RknnAppContext {
    rknn_ctx: rknn_context,
    io_num: rknn_input_output_num,
    input_attrs: Vec<rknn_tensor_attr>,
    output_attrs: Vec<rknn_tensor_attr>,
    model_channel: i32,
    model_width: i32,
    model_height: i32,
    is_quant: bool,
}

impl RknnAppContext {
    pub fn new() -> Self {
        let rknn_ctx = 0u64;
        let io_num = rknn_input_output_num {
            n_input: 0u32,
            n_output: 0u32,
        };
        let (input_attrs, output_attrs) = (Vec::new(), Vec::new());
        let (model_channel, model_width, model_height) = (0i32, 0i32, 0i32);
        let is_quant = false;
        Self {
            rknn_ctx,
            io_num,
            input_attrs,
            output_attrs,
            model_channel,
            model_width,
            model_height,
            is_quant,
        }
    }

    pub fn init_model(&mut self, path: &str) -> Result<()> {
        let mut ctx: rknn_context = 0;
        let mut model_raw = 0 as c_char;
        let mut model_ptr = &mut model_raw as *mut _;
        let model = &mut model_ptr as *mut *mut _;
        // let model = Box::into_raw(Box::new(Box::into_raw(model_raw)));

        // Find absolute path of the model
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}", Path::new(&dir).join(path).display());
        let model_path = CString::new(path).unwrap();

        // Load RKNN Model
        let model_len = unsafe { read_data_from_file(model_path.as_ptr(), model) };

        if model_len < 0 {
            error!("Failed to load rknn model. Return code: {model_len}");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Failed to load rknn model",
            ));
        }

        // let model_ptr: *mut c_void = &mut model_path as *mut _ as *mut c_void;
        let ret = unsafe {
            rknn_init(
                &mut ctx,
                *model as *mut c_void,
                model_len as u32,
                0,
                null_mut(),
            )
        };

        if ret < 0 {
            error!("Failed to init rknn. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Failed to init rknn",
            ));
        }

        // Get Model Input Output Number
        let mut io_num = rknn_input_output_num {
            n_input: 0u32,
            n_output: 0u32,
        };

        let ret = unsafe {
            rknn_query(
                ctx,
                _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
                &mut io_num as *mut _ as *mut c_void,
                size_of::<rknn_input_output_num>() as u32,
            )
        };
        if ret < 0 {
            error!("Failed to query rknn. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Failed to query rknn",
            ));
        }
        info!(
            "Model input num: {}, output num: {}",
            io_num.n_input, io_num.n_output
        );

        // Get Model Input Info
        info!("Input tensors:");
        let mut input_attrs: Vec<rknn_tensor_attr> = Vec::new();

        for i in 0..io_num.n_input {
            let mut attr = rknn_tensor_attr {
                index: i,
                n_dims: 0,
                dims: [0; 16],
                name: [0; 256],
                n_elems: 0,
                size: 0,
                fmt: 0,
                type_: 0,
                qnt_type: 0,
                fl: 0,
                zp: 0,
                scale: 0.0,
                w_stride: 0,
                size_with_stride: 0,
                pass_through: 0,
                h_stride: 0,
            };
            let ret = unsafe {
                rknn_query(
                    ctx,
                    _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    size_of::<rknn_tensor_attr>() as u32,
                )
            };
            if ret != 0 {
                error!("Failed to query rknn. Error code: {ret}");
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Failed to query rknn",
                ));
            }
            dump_tensor_attr(&attr);
            input_attrs.push(attr);
        }

        // Get Model Output Info
        info!("Output tensor");
        let mut output_attrs: Vec<rknn_tensor_attr> = Vec::new();
        for i in 0..io_num.n_output {
            let mut attr = rknn_tensor_attr {
                index: i,
                n_dims: 0,
                dims: [0; 16],
                name: [0; 256],
                n_elems: 0,
                size: 0,
                fmt: 0,
                type_: 0,
                qnt_type: 0,
                fl: 0,
                zp: 0,
                scale: 0.0,
                w_stride: 0,
                size_with_stride: 0,
                pass_through: 0,
                h_stride: 0,
            };
            let ret = unsafe {
                rknn_query(
                    ctx,
                    _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    size_of::<rknn_tensor_attr>() as u32,
                )
            };
            if ret != 0 {
                error!("Failed to query rknn. Error code: {ret}");
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Failed to query rknn",
                ));
            }
            dump_tensor_attr(&attr);
            output_attrs.push(attr);
        }
        // Set to context
        self.rknn_ctx = ctx;
        if output_attrs[0].qnt_type == _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
            && output_attrs[0].type_ == _rknn_tensor_type_RKNN_TENSOR_INT8
        {
            self.is_quant = true;
        } else {
            self.is_quant = false;
        }

        self.io_num = io_num;
        self.input_attrs = input_attrs.clone();
        self.output_attrs = output_attrs;

        if input_attrs[0].fmt == _rknn_tensor_format_RKNN_TENSOR_NCHW {
            info!("model is NCHW input fmt");
            self.model_channel = input_attrs[0].dims[1] as i32;
            self.model_height = input_attrs[0].dims[2] as i32;
            self.model_width = input_attrs[0].dims[3] as i32;
        } else {
            info!("model is NHWC input fmt");
            self.model_height = input_attrs[0].dims[1] as i32;
            self.model_width = input_attrs[0].dims[2] as i32;
            self.model_channel = input_attrs[0].dims[3] as i32;
        }
        info!(
            "model input height={}, width={}, channel={}",
            self.model_height, self.model_width, self.model_channel
        );
        Ok(())
    }
}

pub fn load_image(img_path: &str) -> Result<image_buffer_t> {
    let mut src_image = image_buffer_t {
        width: 0,
        height: 0,
        width_stride: 0,
        height_stride: 0,
        format: 0,
        virt_addr: null_mut(),
        size: 0,
        fd: 0,
    };
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = format!("{}", Path::new(&dir).join(img_path).display());
    let img_path = CString::new(path).unwrap();
    let ret = unsafe { read_image(img_path.as_ptr(), &mut src_image as *mut _) };
    if ret != 0 {
        return Err(Error::new(ErrorKind::InvalidData, "Failed to load image"));
    }
    info!("Source image loaded: {src_image:?}");
    Ok(src_image)
}

// pub fn inference_model(self: &RknnAppContext, img: &image_buffer_t, od_results: )
