use rkod::yolov8::{init_model, RknnAppContext};
// use tracing_subscriber::fmt::time::ChronoLocal;

fn main() {
    tracing_subscriber::fmt()
        .compact()
        // .with_timer(ChronoLocal::new(String::from("[%F %T]")))
        .without_time()
        .with_target(false)
        .init();

    let mut app_ctx = RknnAppContext::new();
    let _ = init_model("model/yolov8.rknn", &mut app_ctx);
}
