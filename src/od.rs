pub const OBJ_NAME_MAX_SIZE: u8 = 64;
pub const OBJ_NUMB_MAX_SIZE: i32 = 128;
pub const OBJ_CLASS_NUM: i32 = 2;
pub const NMS_THRESH: f32 = 0.45;
pub const BOX_THRESH: f32 = 0.25;
const PROB_THRESHOLD: f32 = 0.5;

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
    ) -> Result<Self, String> {
        if class_id.len() != obj_probs.len() || order.len() != class_id.len() {
            return Err("class_id, obj_probs and order should be the same length.".to_string());
        }
        if PROB_THRESHOLD < 0. || PROB_THRESHOLD >= 1. {
            return Err("PROB_THRESHOLD should be within range [0,1).".to_string());
        }
        let mut count = 0i32;
        let mut results: Vec<ObjectDetection> = Vec::new();
        for i in 0..class_id.len() {
            if count >= OBJ_NUMB_MAX_SIZE {
                break;
            }
            let n = order[i];

            if obj_probs[n] < PROB_THRESHOLD {
                break;
            }

            if n == 0xffff {
                continue;
            }
            let res = ObjectDetection {
                prob: obj_probs[n],
                cls_id: class_id[n],
            };
            results.push(res);
            count += 1;
        }
        Ok(Self { count, results })
    }

    pub fn get_results(&self) -> Vec<(i32, f32)> {
        self.results
            .iter()
            .map(|r| (r.cls_id, r.prob))
            .collect::<Vec<_>>()
    }

    pub fn get_count(&self) -> i32 {
        self.count
    }
}
