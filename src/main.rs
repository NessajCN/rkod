extern crate ffmpeg_the_third as ffmpeg;

use clap::Parser;
use image::ImageReader;
use rkod::{
    cv::FrameExtractor,
    od::RknnAppContext,
    read_lines,
    upload::{UpError, UploaderWorker},
};
use std::io::{self, Error, ErrorKind};
use tracing::{error, info};

use ffmpeg::{format, media};
// use tracing_subscriber::fmt::time::ChronoLocal;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model to use
    #[arg(short, long, default_value = "model/safety_hat.rknn")]
    model: String,

    /// Path to the input for inference. Could be image or video.
    #[arg(short, long)]
    input: String,

    /// Object detection results uploading switch.
    /// You don't need it unless you have deployed your own RESTful API.
    #[arg(short, long)]
    upload: bool,
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .compact()
        // .with_timer(ChronoLocal::new(String::from("[%F %T]")))
        .without_time()
        .with_target(false)
        .init();

    let args = Args::parse();
    let lines = read_lines("model/label_list.txt")?;
    let labels = lines.flatten().collect::<Vec<String>>();
    let mut app_ctx = RknnAppContext::new();
    app_ctx.init_model(&args.model)?;

    if args.input.starts_with("rtsp") {
        // Init ffmpeg
        ffmpeg::init()?;
        let mut ictx = format::input(&args.input)?;

        // Print detailed information about the input or output format,
        // such as duration, bitrate, streams, container, programs, metadata, side data, codec and time base.
        format::context::input::dump(&ictx, 0, Some(&args.input));

        // Find video stream from input context.
        let input = ictx
            .streams()
            .best(media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_stream_index = input.index();

        let mut frame_extractor = FrameExtractor::new(&input, [app_ctx.width(), app_ctx.height()])?;

        let upworker = if args.upload {
            let u = UploaderWorker::new();
            Some(u)
        } else {
            None
        };
        // let mut frame_count = 0 as usize;
        for packet_result in ictx.packets() {
            let (stream, packet) = packet_result?;
            
            // Detect objects from 1 frame every 64 extracted.
            // frame_count = frame_count.wrapping_add(1 as usize);
            // if frame_count % 32 != 0 {
            //     continue;
            // }

            // Find key frame
            if !packet.is_key() {
                continue;
            }
            if stream.index() == video_stream_index {
                frame_extractor.send_packet_to_decoder(&packet)?;
                if let Some(r) = frame_extractor.process_frames(&app_ctx)? {
                    let results = r
                        .into_iter()
                        .map(|(id, prob, f_box)| (labels[id as usize].clone(), prob, f_box))
                        .collect::<Vec<_>>();

                    if let Some(upload_worker) = upworker.as_ref() {
                        if let Err(UpError::ChannelError(e)) =
                            upload_worker.upload_odres(results.clone())
                        {
                            error!("Failed to send od result to UploadWorker channel: {e}");
                        }
                    }
                    info!("Object detected: {results:?}");
                } else {
                    // info!("No object deteced.");
                }
            }
        }
        frame_extractor.send_eof_to_decoder()?;
        frame_extractor.process_frames(&app_ctx)?;
    } else {
        // Read image raw bytes
        let reader = ImageReader::open(&args.input)?;
        let img = match reader.decode() {
            Ok(m) => m,
            Err(e) => {
                return Err(Error::new(ErrorKind::InvalidInput, e.to_string()));
            }
        };
        let img = img.resize_to_fill(
            app_ctx.width(),
            app_ctx.height(),
            image::imageops::FilterType::Nearest,
        );

        let od_results = app_ctx.inference_model(img.as_bytes())?;
        let results = od_results
            .get_results()
            .into_iter()
            .map(|(id, prob, f_box)| (labels[id as usize].clone(), prob, f_box))
            .collect::<Vec<_>>();

        info!("results: {results:?}");
    }
    Ok(())
}
