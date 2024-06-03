use rkod::{read_lines, yolov8::RknnAppContext};
use tracing::{error, info};
use std::io;
use clap::Parser;
// use tracing_subscriber::fmt::time::ChronoLocal;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model to use
    #[arg(short, long)]
    model: String,

    /// Path to the Image for inference
    #[arg(short, long)]
    image: String,
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .compact()
        // .with_timer(ChronoLocal::new(String::from("[%F %T]")))
        .without_time()
        .with_target(false)
        .init();

    let args = Args::parse();
    let lines = read_lines("model/safety_hat.txt")?;
    let labels = lines.flatten().collect::<Vec<String>>();
    let mut app_ctx = RknnAppContext::new();
    app_ctx.init_model(&args.model)?;
    let class_set = app_ctx.inference_model(&args.image)?;
    if class_set.contains(&-1) || class_set.is_empty() {
        error!("class id error");
        return Err(io::Error::new(io::ErrorKind::InvalidData, "class id error"));
    }
    info!("Image containing class: {class_set:?}");
    // let class_names = class_set.into_iter().map(|i| labels[i as usize].clone()).collect::<Vec<String>>();
    // info!("Image containing class: {class_names:?}");
    Ok(())
}
