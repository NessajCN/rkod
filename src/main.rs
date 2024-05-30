use rkod::{read_lines, yolov8::RknnAppContext};
use tracing::{error, info};
use std::io;
// use tracing_subscriber::fmt::time::ChronoLocal;

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .compact()
        // .with_timer(ChronoLocal::new(String::from("[%F %T]")))
        .without_time()
        .with_target(false)
        .init();

    let lines = read_lines("model/coco_80_labels_list.txt")?;
    let labels = lines.flatten().collect::<Vec<String>>();
    let mut app_ctx = RknnAppContext::new();
    app_ctx.init_model("model/yolov8.rknn")?;
    let class_set = app_ctx.inference_model("model/bus.jpg")?;
    if class_set.contains(&-1) || class_set.is_empty() {
        error!("class id error");
        return Err(io::Error::new(io::ErrorKind::InvalidData, "class id error"));
    }
    let class_names = class_set.into_iter().map(|i| labels[i as usize].clone()).collect::<Vec<String>>();
    info!("Image containing class: {class_names:?}");
    Ok(())
}
