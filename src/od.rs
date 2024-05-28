pub const OBJ_NAME_MAX_SIZE: u8 = 64;
pub const OBJ_NUMB_MAX_SIZE: u8 = 128;
pub const OBJ_CLASS_NUM: u8 =  80;
pub const NMS_THRESH: f64 = 0.45;
pub const BOX_THRESH: f64 = 0.25;

struct ImageRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

struct ObjectDetection {
    rect: ImageRect,
    prop: f64,
    cls_id: i32,
}

pub struct ObjectDetectList {
    id: i32,
    count: i32,
    results: Vec<ObjectDetection>,
}

