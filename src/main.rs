use rkod::{read_lines, yolov8::RknnAppContext};
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
    app_ctx.inference_model("model/bus.jpg")?;
    Ok(())
}
