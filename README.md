# rknpu2-rs

Rust bindings for the [Rockchip RKNN API (rknpu2)](https://github.com/airockchip/rknn-toolkit2), targeting deployment of deep learning models on Rockchip NPUs.

> These bindings do **not** require redistributing the C headers from the Rockchip rknpu2 SDK. The sys crate contains bindgen bindings, but bindgen is not part of the build process.

## Requirements

- Rockchip NPU compatible with librknnrt or librknnmrt libraries.

## Features

- Safe and idiomatic Rust abstractions over the `librknn_api` C API
- Support for loading models, setting inputs, running inference, and reading outputs
- There will be Zero-copy input/output buffer support
- Includes both low-level `-sys` bindings and higher-level wrappers
- Based on the **rknpu2 SDK v2.3.2**

## Version Compatibility

These bindings are aligned with the **2.3.2** release of the RKNN API.
[rknn C API v2.3.2 PDF](https://github.com/airockchip/rknn-toolkit2/blob/42aa1d426c0a9e0869b6374edba009f7208a1926/doc/04_Rockchip_RKNPU_API_Reference_RKNNRT_V2.3.2_EN.pdf)


## Usage

Coming soon!
