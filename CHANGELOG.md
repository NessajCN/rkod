# Change log

All notable changes to the `rkod` project will be documented in this file.

## 0.2.0

已完成全部识别和抓帧的模块，并新加入上传识别结果到 api 接口的模块 `upload.rs`
- 视频帧抓取识别从固定间隔改为仅识别关键帧
- 修复 `od` 模块中 `fn cal_overlap(mxy: [f32; 8]) -> f32` 函数的一个符号错位。该错位会导致 `nms()` 过滤函数不起作用

## 0.1.0

用 Rust 写的视频帧识别程序。使用 yolov8 训练并用 rknn 转换的安全帽模型（详见 `rkconv` 项目）。主要特性：
- `od` 模组 —— 载入图片或视频帧并调用 rknn 模型进行识别处理
- `cv` 模组 —— 调用 ffmpeg 库读取视频流，抓帧，解码。再喂给 `od` 模组
- `bindings` 模组 —— 由 `bindgen` 将 rknn 相关的 C API 生成为 Rust ffi.
项目地址 https://github.com/NessajCN/rkod