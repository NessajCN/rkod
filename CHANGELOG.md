# Change log

All notable changes to the `rkod` project will be documented in this file.

## 0.1.0

用 Rust 写的视频帧识别程序。使用 yolov8 训练并用 rknn 转换的安全帽模型（详见 `rkconv` 项目）。主要特性：
- `od` 模组 —— 载入图片或视频帧并调用 rknn 模型进行识别处理
- `cv` 模组 —— 调用 ffmpeg 库读取视频流，抓帧，解码。再喂给 `od` 模组
- `bindings` 模组 —— 由 `bindgen` 将 rknn 相关的 C API 生成为 Rust ffi.
项目地址 https://github.com/NessajCN/rkod