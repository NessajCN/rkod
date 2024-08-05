extern crate ffmpeg_next as ffmpeg;

use std::io::{self, Result};
// use std::path::Path;

use ffmpeg::format::{stream, Pixel};
// use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::{decoder, frame, picture, Packet};

use crate::od::RknnAppContext;

pub struct FrameExtractor {
    decoder: decoder::Video,
    scaler: Context,
}

impl FrameExtractor {
    /// frame_size: \[width, height\]
    pub fn new(ist: &stream::Stream, frame_size: [u32; 2]) -> Result<Self> {
        let decoder = ffmpeg::codec::context::Context::from_parameters(ist.parameters())?
            .decoder()
            .video()?;

        let scaler = Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            frame_size[0],
            frame_size[1],
            Flags::BILINEAR,
        )?;

        Ok(Self { decoder, scaler })
    }

    pub fn send_packet_to_decoder(&mut self, packet: &Packet) -> Result<()> {
        self.decoder.send_packet(packet)?;
        Ok(())
    }

    pub fn send_eof_to_decoder(&mut self) -> Result<()> {
        self.decoder.send_eof()?;
        Ok(())
    }

    pub fn process_frames(
        &mut self,
        app_ctx: &RknnAppContext,
    ) -> Result<Option<Vec<(i32, f32, [f32; 4])>>> {
        let mut frame = frame::Video::empty();
        while self.decoder.receive_frame(&mut frame).is_ok() {
            let timestamp = frame.timestamp();

            frame.set_pts(timestamp);
            frame.set_kind(picture::Type::None);
            let mut rgb_frame = Video::empty();
            self.scaler.run(&frame, &mut rgb_frame)?;

            let f = rgb_frame.data(0);

            let result = match app_ctx.inference_model(f) {
                Ok(r) => {
                    let od_result = r.get_results().into_iter().collect::<Vec<_>>();
                    Some(od_result)
                }
                Err(e) => {
                    if e.kind() != io::ErrorKind::NotFound {
                        return Err(e);
                    }
                    None
                }
            };
            return Ok(result);
        }
        Ok(None)
    }
}

// fn extract_frame<P: AsRef<Path> + ?Sized>(input_path: &P, frame_size: [u32; 2]) -> Result<()> {
//     ffmpeg::init()?;
//     if let Ok(mut ictx) = input(input_path) {
//         let input = ictx
//             .streams()
//             .best(Type::Video)
//             .ok_or(ffmpeg::Error::StreamNotFound)?;
//         let video_stream_index = input.index();

//         let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
//         let mut decoder = context_decoder.decoder().video()?;

//         let mut scaler = Context::get(
//             decoder.format(),
//             decoder.width(),
//             decoder.height(),
//             Pixel::RGB24,
//             frame_size[0],
//             frame_size[1],
//             Flags::BILINEAR,
//         )?;

//         let mut frame_index = 0;

//         let mut receive_and_process_decoded_frames =
//             |decoder: &mut ffmpeg::decoder::Video| -> Result<()> {
//                 let mut decoded = Video::empty();
//                 while decoder.receive_frame(&mut decoded).is_ok() {
//                     let mut rgb_frame = Video::empty();
//                     scaler.run(&decoded, &mut rgb_frame)?;
//                     // save_file(&rgb_frame, frame_index).unwrap();
//                     frame_index += 1;
//                 }
//                 Ok(())
//             };

//         for (stream, packet) in ictx.packets() {
//             if stream.index() == video_stream_index {
//                 decoder.send_packet(&packet)?;
//                 receive_and_process_decoded_frames(&mut decoder)?;
//             }
//         }
//         decoder.send_eof()?;
//         receive_and_process_decoded_frames(&mut decoder)?;
//     }
//     Ok(())
// }
