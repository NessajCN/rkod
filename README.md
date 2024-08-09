It's the Utility of Rockchip's [RKNN](https://github.com/airockchip/rknn-toolkit2) C API on rk3588.
Written in Rust with FFI.
`src/bindings.rs` was generated by 
`bindgen wrapper.h -o src/bindings.rs`

This repo is actually a Rust port of the yolov8 example in [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/cpp/main.cc)

To run it:

- Download yolov8n.onnx
```bash
git clone https://github.com/NessajCN/rkod.git
cd rkod/model
bash ./download_model.sh
```

- Convert onnx to rknn model following [this instruction](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/README.md#4-convert-to-rknn). Install any dependencies necessary.

```bash
cd python
python convert.py <onnx_model> <TARGET_PLATFORM> <dtype(optional)> <output_rknn_path(optional)>

# such as: 
python convert.py ../model/yolov8n.onnx rk3588
```
- Move `yolov8.rknn` in `rkod/model`

- Fill `model/label_list.txt` with object name labels as you trained(one per line). Example could be found in `model/coco_80_labels_list.txt`.

- Change the const `OBJ_CLASS_NUM` in `src/od.rs` to the total number of entries you put in `model/labels_list.txt`. e.g. `const OBJ_CLASS_NUM: i32 = 80;` if you adopted `model/coco_80_labels_list.txt` as the labels_list.

- To detect objects in a single image:
```bash
cargo run -- -m <model/path> -i <image/path>
```
- To continuously detect the rtsp stream:
```bash
cargo run -- -m <model/path> -i rtsp://<rtsp-stream-path>
```

> Note: It is recommended to compile and install the Rockchip version of ffmpeg following [this guild](https://github.com/nyanmisaka/ffmpeg-rockchip/wiki/Compilation) if you intend to detect an rtsp streaming.