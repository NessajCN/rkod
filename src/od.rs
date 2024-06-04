pub const OBJ_NAME_MAX_SIZE: u8 = 64;
pub const OBJ_NUMB_MAX_SIZE: u8 = 128;
pub const OBJ_CLASS_NUM: i32 =  2;
pub const NMS_THRESH: f32 = 0.45;
pub const BOX_THRESH: f32 = 0.25;

// #[derive(Debug, Default, Clone, Copy)]
// struct ImageRect {
//     left: i32,
//     top: i32,
//     right: i32,
//     bottom: i32,
// }

// #[derive(Debug, Default, Clone, Copy)]
// struct ObjectDetection {
//     rect: ImageRect,
//     prop: f32,
//     cls_id: i32,
// }

// #[derive(Debug, Default, Clone)]
// pub struct ObjectDetectList {
//     id: i32,
//     count: i32,
//     results: Vec<ObjectDetection>,
// }