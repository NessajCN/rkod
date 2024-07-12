use std::{
    collections::HashSet,
    fs::File,
    io::{self, Read, Result},
    mem::size_of,
    ptr::null_mut,
};

use crate::{
    _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR, _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
    _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR, _rknn_tensor_format_RKNN_TENSOR_NCHW,
    _rknn_tensor_format_RKNN_TENSOR_NHWC, _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC,
    _rknn_tensor_type_RKNN_TENSOR_INT8, _rknn_tensor_type_RKNN_TENSOR_UINT8, dump_tensor_attr,
    rknn_context, rknn_init, rknn_input, rknn_input_output_num, rknn_inputs_set, rknn_output,
    rknn_outputs_get, rknn_outputs_release, rknn_query, rknn_run, rknn_tensor_attr,
};
use libc::c_void;
use tracing::{error, info};

// const OBJ_NAME_MAX_SIZE: u8 = 64;
const OBJ_NUMB_MAX_SIZE: i32 = 128;
const OBJ_CLASS_NUM: i32 = 2;
const NMS_THRESH: f32 = 0.45;
const BOX_THRESH: f32 = 0.25;
const PROB_THRESHOLD: f32 = 0.2;

// #[derive(Debug, Default, Clone, Copy)]
// struct ImageRect {
//     left: i32,
//     top: i32,
//     right: i32,
//     bottom: i32,
// }

#[derive(Debug, Default, Clone, Copy)]
struct ObjectDetection {
    // rect: ImageRect,
    prob: f32,
    cls_id: i32,
    f_box: [f32; 4],
}

#[derive(Debug, Clone, Default)]
pub struct ObjectDetectList {
    count: i32,
    results: Vec<ObjectDetection>,
}

impl ObjectDetectList {
    pub fn new(
        class_id: &Vec<i32>,
        obj_probs: &Vec<f32>,
        order: &Vec<usize>,
        filter_boxes: &Vec<[f32; 4]>,
    ) -> Result<Self> {
        if class_id.len() != obj_probs.len() || order.len() != class_id.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "class_id, obj_probs and order should be the same length.",
            ));
        }
        if PROB_THRESHOLD < 0. || PROB_THRESHOLD >= 1. {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PROB_THRESHOLD should be within range [0,1).",
            ));
        }
        let mut count = 0i32;
        let mut results: Vec<ObjectDetection> = Vec::new();
        for i in 0..class_id.len() {
            if count >= OBJ_NUMB_MAX_SIZE {
                break;
            }
            let n = order[i];
            if n == 0xffff {
                continue;
            }

            if obj_probs[n] < PROB_THRESHOLD {
                break;
            }

            let res = ObjectDetection {
                prob: obj_probs[n],
                cls_id: class_id[n],
                f_box: filter_boxes[n],
            };
            results.push(res);
            count += 1;
        }
        Ok(Self { count, results })
    }

    pub fn get_results(&self) -> Vec<(i32, f32, [f32; 4])> {
        self.results
            .iter()
            .map(|r| (r.cls_id, r.prob, r.f_box))
            .collect::<Vec<_>>()
    }

    pub fn get_count(&self) -> i32 {
        self.count
    }
}

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

    pub fn width(&self) -> u32 {
        self.model_width as _
    }

    pub fn height(&self) -> u32 {
        self.model_height as _
    }

    pub fn init_model(&mut self, path: &str) -> Result<()> {
        let mut ctx: rknn_context = 0;

        // let mut model_raw = 0 as c_char;
        // let mut model_ptr = &mut model_raw as *mut _;
        // let model = &mut model_ptr as *mut *mut _;
        // // let model = Box::into_raw(Box::new(Box::into_raw(model_raw)));

        // // Find absolute path of the model
        // let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        // let path = format!("{}", Path::new(&dir).join(path).display());
        // let model_path = CString::new(path).unwrap();

        // // Load RKNN Model
        // let model_len = unsafe { read_data_from_file(model_path.as_ptr(), model) };

        // if model_len < 0 {
        //     error!("Failed to load rknn model. Return code: {model_len}");
        //     return Err(io::Error::new(
        //         io::ErrorKind::InvalidData,
        //         "Failed to load rknn model",
        //     ));
        // }

        let mut model = File::open(path)?;
        let mut model_buf: Vec<u8> = Vec::new();

        let model_len = model.read_to_end(&mut model_buf)?;
        let model = model_buf.as_mut_ptr() as *mut c_void;

        let ret = unsafe { rknn_init(&mut ctx, model, model_len as u32, 0, null_mut()) };

        // info!("model_buf: {model_buf:?}");
        drop(model_buf);

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
        // info!("Output tensor");
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
            // dump_tensor_attr(&attr);
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

    pub fn inference_model(&self, img: &[u8]) -> Result<ObjectDetectList> {
        // let reader = ImageReader::open(img_path)?;
        // let img = match reader.decode() {
        //     Ok(m) => m,
        //     Err(e) => {
        //         return Err(Error::new(ErrorKind::InvalidInput, e.to_string()));
        //     }
        // };
        // let img = img.resize_to_fill(
        //     self.model_width as u32,
        //     self.model_height as u32,
        //     image::imageops::FilterType::Nearest,
        // );

        let img_buf = img.as_ptr() as *mut c_void;
        let mut inputs: Vec<rknn_input> = Vec::new();
        for n in 0..self.io_num.n_input {
            let input = rknn_input {
                index: n,
                size: (self.model_width * self.model_height * self.model_channel) as u32,
                type_: _rknn_tensor_type_RKNN_TENSOR_UINT8,
                fmt: _rknn_tensor_format_RKNN_TENSOR_NHWC,
                // pass_through - if 1 directly pass image buff to rknn node else if 0 do conversion first.
                pass_through: 0,
                buf: img_buf,
            };
            inputs.push(input);
        }

        // info!("Setting rknn inputs...n_input: {}", self.io_num.n_input);

        let ret =
            unsafe { rknn_inputs_set(self.rknn_ctx, self.io_num.n_input, inputs.as_mut_ptr()) };

        if ret < 0 {
            error!("Failed to set rknn input. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Failed to set rknn input",
            ));
        }

        // info!("Running rknn...");

        let ret = unsafe { rknn_run(self.rknn_ctx, null_mut()) };

        if ret < 0 {
            error!("Failed to run rknn. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Failed to run rknn",
            ));
        }

        let mut outputs: Vec<rknn_output> = Vec::new();
        for i in 0..self.io_num.n_output {
            let output = rknn_output {
                index: i,
                want_float: !self.is_quant as u8,
                is_prealloc: 0,
                size: 0,
                buf: null_mut() as *mut c_void,
            };
            outputs.push(output);
        }

        // info!("Generating outputs...");

        let ret = unsafe {
            rknn_outputs_get(
                self.rknn_ctx,
                self.io_num.n_output,
                outputs.as_mut_ptr(),
                null_mut(),
            )
        };
        if ret < 0 {
            error!("Failed to get rknn outputs. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Failed to get rknnoutputs",
            ));
        }

        // Post process
        // let mut valid_count = 0;

        // info!("Post process begins...");

        let mut filter_boxes: Vec<[f32; 4]> = Vec::new();
        let mut obj_probs: Vec<f32> = Vec::new();
        let mut class_id: Vec<i32> = Vec::new();

        let dfl_len = self.output_attrs[0].dims[1] / 4;
        let output_per_branch = self.io_num.n_output / 3;
        for i in 0..3 {
            let (score_sum, score_sum_zp, score_sum_scale) = if output_per_branch == 3 {
                (
                    outputs[i * 3 + 2].buf,
                    self.output_attrs[i * 3 + 2].zp,
                    self.output_attrs[i * 3 + 2].scale,
                )
            } else {
                (null_mut() as *mut c_void, 0, 1.0)
            };
            let box_idx = i * output_per_branch as usize;
            let score_idx = i * output_per_branch as usize + 1;

            let grid_h = self.output_attrs[box_idx].dims[2];
            let grid_w = self.output_attrs[box_idx].dims[3];
            let stride = self.model_height as u32 / grid_h;

            if self.is_quant {
                // process_i8
                let grid_len = (grid_h * grid_w) as usize;
                let score_thres_i8 = qnt_f32_to_affine(
                    BOX_THRESH,
                    self.output_attrs[score_idx].zp,
                    self.output_attrs[score_idx].scale,
                );
                let score_sum_thres_i8 =
                    qnt_f32_to_affine(BOX_THRESH, score_sum_zp, score_sum_scale);
                for m in 0..grid_h {
                    for n in 0..grid_w {
                        let offset = (m * grid_w + n) as usize;
                        let mut max_cls_id = -1;

                        // 通过 score sum 起到快速过滤的作用
                        if !score_sum.is_null() {
                            let buf_offset =
                                unsafe { *(score_sum.wrapping_add(offset) as *mut i8) };
                            if buf_offset < score_sum_thres_i8 {
                                continue;
                            }
                        }

                        let mut max_score = -self.output_attrs[score_idx].zp as i8;
                        for k in 0..OBJ_CLASS_NUM {
                            let buf_offset = unsafe {
                                *(outputs[score_idx]
                                    .buf
                                    .wrapping_add(offset + grid_len * k as usize)
                                    as *mut i8)
                            };
                            if buf_offset > score_thres_i8 && buf_offset > max_score {
                                max_score = buf_offset;
                                max_cls_id = k;
                            }
                            // info!("looping in: i: {i}, m: {m}, n: {n}, k:{k}");
                        }

                        // compute box
                        if max_score > score_thres_i8 {
                            // let mut offset = (m * grid_w + n) as usize ;
                            let mut before_dfl: Vec<f32> = Vec::new();
                            for k in 0..(dfl_len * 4) {
                                let box_tensor = unsafe {
                                    *(outputs[box_idx]
                                        .buf
                                        .wrapping_add(offset + grid_len * k as usize)
                                        as *mut i8)
                                };
                                let deqnt = (box_tensor as f32
                                    - self.output_attrs[box_idx].zp as f32)
                                    * self.output_attrs[box_idx].scale as f32;
                                before_dfl.push(deqnt);
                            }
                            let draw_box = compute_dfl(before_dfl, dfl_len as usize);

                            let x1 = (-draw_box[0] + n as f32 + 0.5) * stride as f32;
                            let y1 = (-draw_box[1] + m as f32 + 0.5) * stride as f32;
                            let x2 = (draw_box[2] + n as f32 + 0.5) * stride as f32;
                            let y2 = (draw_box[3] + m as f32 + 0.5) * stride as f32;
                            // let w = x2 - x1;
                            // let h = y2 - y1;

                            filter_boxes.push([x1, y1, x2, y2]);

                            // filter_boxes.push(x1);
                            // filter_boxes.push(y1);
                            // filter_boxes.push(w);
                            // filter_boxes.push(h);

                            let deqnt = (max_score as f32 - self.output_attrs[score_idx].zp as f32)
                                * self.output_attrs[score_idx].scale as f32;

                            obj_probs.push(deqnt);
                            class_id.push(max_cls_id);
                            // valid_count += 1;
                        }
                    }
                }
            } else {
                // process_fp32
                let grid_len = (grid_h * grid_w) as usize;
                for m in 0..grid_h {
                    for n in 0..grid_w {
                        let offset = (m * grid_w + n) as usize;
                        let mut max_cls_id = -1;

                        // 通过 score sum 起到快速过滤的作用
                        if !score_sum.is_null() {
                            let buf_offset =
                                unsafe { *(score_sum.wrapping_add(offset) as *mut i8) };
                            if (buf_offset as f32) < BOX_THRESH {
                                continue;
                            }
                        }

                        let mut max_score = 0.0f32;
                        for k in 0..OBJ_CLASS_NUM {
                            let buf_offset = unsafe {
                                *(outputs[score_idx]
                                    .buf
                                    .wrapping_add(offset + grid_len * k as usize)
                                    as *mut f32)
                            };
                            if buf_offset > BOX_THRESH && buf_offset > max_score {
                                max_score = buf_offset;
                                max_cls_id = k;
                            }
                            // offset += grid_len as usize;
                        }

                        // compute box
                        if max_score > BOX_THRESH {
                            // let mut offset = (m * grid_w + n) as usize ;
                            let mut before_dfl: Vec<f32> = Vec::new();
                            for k in 0..(dfl_len * 4) {
                                let box_tensor = unsafe {
                                    *(outputs[box_idx]
                                        .buf
                                        .wrapping_add(offset + grid_len * k as usize)
                                        as *mut f32)
                                };
                                let deqnt = (box_tensor - self.output_attrs[box_idx].zp as f32)
                                    * self.output_attrs[box_idx].scale as f32;
                                before_dfl.push(deqnt);
                            }
                            let draw_box = compute_dfl(before_dfl, dfl_len as usize);

                            let x1 = (-draw_box[0] + n as f32 + 0.5) * stride as f32;
                            let y1 = (-draw_box[1] + m as f32 + 0.5) * stride as f32;
                            let x2 = (draw_box[2] + n as f32 + 0.5) * stride as f32;
                            let y2 = (draw_box[3] + m as f32 + 0.5) * stride as f32;
                            // let w = x2 - x1;
                            // let h = y2 - y1;

                            filter_boxes.push([x1, y1, x2, y2]);

                            // filter_boxes.push(x1);
                            // filter_boxes.push(y1);
                            // filter_boxes.push(w);
                            // filter_boxes.push(h);

                            obj_probs.push(max_score);
                            class_id.push(max_cls_id);
                            // valid_count += 1;
                        }
                    }
                }
            }
        }

        if obj_probs.len() == 0 {
            // warn!("No object detected");
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "No object detected",
            ));
        }

        let class_set: HashSet<i32> = HashSet::from_iter(class_id.clone().into_iter());

        // info!("obj_probs: {obj_probs:?}");
        // info!("class_id: {class_id:?}");

        let mut order = (0..obj_probs.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| obj_probs[b].total_cmp(&obj_probs[a]));

        // nms

        for &c in class_set.iter() {
            nms(&filter_boxes, &class_id, &mut order, c);
        }

        // nms end

        let od_result = match ObjectDetectList::new(&class_id, &obj_probs, &order, &filter_boxes) {
            Ok(r) => r,
            Err(e) => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, e));
            }
        };

        // info!("Rknn running: context is now {}", self.rknn_ctx);
        let _ = unsafe {
            rknn_outputs_release(self.rknn_ctx, self.io_num.n_output, outputs.as_mut_ptr())
        };

        Ok(od_result)
    }
}

fn qnt_f32_to_affine(threshold: f32, score_zp: i32, score_scale: f32) -> i8 {
    let dst_val = (threshold / score_zp as f32) + score_scale as f32;
    match (dst_val <= -128.0, dst_val >= 127.0) {
        (true, _) => -128i8,
        (false, true) => 127i8,
        (false, false) => dst_val as i8,
    }
}

fn compute_dfl(tensor: Vec<f32>, dfl_len: usize) -> [f32; 4] {
    let mut draw_box = [0.0f32; 4];
    for b in 0..4 as usize {
        let mut exp_t: Vec<f32> = Vec::new();
        let mut exp_sum = 0.0f32;
        let mut acc_sum = 0.0f32;
        for i in 0..dfl_len {
            let expon = tensor[i + b * dfl_len].exp();
            exp_t.push(expon);
            exp_sum += expon;
        }

        for i in 0..dfl_len {
            acc_sum += exp_t[i] / exp_sum * (i as f32);
        }
        draw_box[b] = acc_sum;
    }
    draw_box
}

pub fn nms(
    filter_boxes: &Vec<[f32; 4]>,
    class_id: &Vec<i32>,
    order: &mut Vec<usize>,
    filter_id: i32,
) {
    for i in 0..class_id.len() {
        if (order[i] == 0xffff) || (class_id[i] != filter_id) {
            continue;
        }
        let n = order[i];
        for j in (i + 1)..class_id.len() {
            let m = order[j];
            if m == 0xffff || class_id[i] != filter_id {
                continue;
            }
            let iou = cal_overlap([
                filter_boxes[n][0],
                filter_boxes[n][1],
                filter_boxes[n][2],
                filter_boxes[n][3],
                filter_boxes[m][0],
                filter_boxes[m][1],
                filter_boxes[m][2],
                filter_boxes[m][3],
            ]);
            if iou > NMS_THRESH {
                order[j] = 0xffff;
            }
        }
    }
}

/// mxy: [xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1]
fn cal_overlap(mxy: [f32; 8]) -> f32 {
    let xmax = if mxy[2] >= mxy[6] { mxy[6] } else { mxy[2] };
    let xmin = if mxy[0] >= mxy[4] { mxy[0] } else { mxy[4] };
    let ymax = if mxy[3] >= mxy[7] { mxy[7] } else { mxy[3] };
    let ymin = if mxy[1] >= mxy[5] { mxy[1] } else { mxy[5] };
    let w = if xmax - xmin + 1. > 0. {
        xmax - xmin + 1.
    } else {
        0.
    };
    let h = if ymax - ymin + 1. > 0. {
        ymax - ymin + 1.
    } else {
        0.
    };
    let i = w * h;
    let u = (mxy[2] - mxy[0] + 1.) * (mxy[3] - mxy[1] + 1.)
        + (mxy[6] - mxy[4] + 1.) * (mxy[7] - mxy[5] + 1.)
        - i;
    if u <= 0. {
        0.
    } else {
        i / u
    }
}
