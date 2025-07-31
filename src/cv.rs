extern crate ffmpeg_the_third as ffmpeg;

use std::io::{self, Result};

use ffmpeg::format::{stream, Pixel};
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