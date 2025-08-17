# RKNN

Rust bindings for the Rockchip RKNN Runtime API (librknnrt.so), enabling deployment of deep learning models on Rockchip NPUs. This library is a key component of the rknpu2 SDK, providing efficient integration between Rust applications and Rockchip's neural processing unit for optimized AI model execution.

> These bindings do **not** require redistributing the C headers from the Rockchip rknpu2 SDK. The sys crate contains bindgen bindings, but bindgen is not part of the build process.

## Requirements

- Rockchip NPU compatible with `librknnrt` or `librknnmrt` libraries
- Rust 1.70+ (edition 2021)

## Features

- Safe and idiomatic Rust abstractions over the `librknnrt.so` C API
- Support for loading models, setting inputs, running inference, and reading outputs
- Zero-copy input/output buffer support (planned)
- Includes both low-level `-sys` bindings and higher-level wrappers
- Based on the **rknpu2 SDK v2.3.2**
- Comprehensive error handling with custom error types
- Memory-safe tensor operations and buffer management


## Project Structure

```
crates/
├── rknpu2/          # High-level Rust API wrapper
├── rknpu2-sys/      # Low-level FFI bindings
└── rktensor/        # Tensor operations and utilities
```

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
rknpu2 = { git = "https://github.com/CodeVoyager15/RKNN", package = "rknpu2" }
rktensor = { git = "https://github.com/CodeVoyager15/RKNN", package = "rktensor" }
```

## Usage

Coming soon! The library will provide a simple and intuitive API for:

- Loading RKNN models
- Setting input tensors
- Running inference
- Retrieving output results
- Managing memory and buffers efficiently

## Examples

Check out the examples in `crates/rknpu2/examples/` for usage demonstrations.

## Development Status

This project is actively maintained and follows Rust best practices. Contributions are welcome!

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
