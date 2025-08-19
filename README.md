# RKNN

Rust bindings for the Rockchip RKNN Runtime API (librknnrt.so), enabling deployment of deep learning models on Rockchip NPUs. This library is a key component of the rknpu2 SDK, providing efficient integration between Rust applications and Rockchip's neural processing unit for optimized AI model execution.

> These bindings do **not** require redistributing the C headers from the Rockchip rknpu2 SDK. The sys crate contains bindgen bindings, but bindgen is not part of the build process.

## Overview

This crate provides safe, idiomatic Rust abstractions over the Rockchip RKNN runtime API, enabling efficient deployment and inference of deep learning models on Rockchip NPU hardware. It's designed for production use in embedded systems, IoT devices, and edge computing applications where Rockchip NPUs are deployed.

## Requirements

- Rockchip NPU compatible with `librknnrt` or `librknnmrt` libraries
- Rust 1.70+ (edition 2021)
- Linux environment (primary target)
- Supported NPU families: RK35xx, RK3576, and others compatible with RKNN API v2.3.2

## Features

- **Safe Rust Abstractions**: Memory-safe wrappers around the C API with proper error handling
- **Model Management**: Load, query, and manage RKNN models with type-safe interfaces
- **Inference Pipeline**: Set inputs, run inference, and retrieve outputs with zero-copy buffer support
- **Hardware Optimization**: Configure NPU core usage and batch processing for optimal performance
- **Flexible API Loading**: Choose between linked and runtime-loaded RKNN libraries
- **Comprehensive Querying**: Access model metadata, tensor attributes, and performance information
- **Multi-format Support**: Handle various tensor formats (NCHW, NHWC) and data types
- **Production Ready**: Based on the stable **rknpu2 SDK v2.3.2**
- **Comprehensive Error Handling**: Custom error types with detailed error information
- **Memory-safe Operations**: Safe tensor operations and buffer management

## Project Structure

```
crates/
├── rknpu2/          # High-level Rust API wrapper
├── rknpu2-sys/      # Low-level FFI bindings
└── rktensor/        # Tensor operations and utilities
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rknpu2 = { version = "0.1", features = ["rk35xx"] }

# For runtime library loading (optional)
rknpu2 = { version = "0.1", features = ["rk35xx", "libloading"] }

# For RK3576 specific features
rknpu2 = { version = "0.1", features = ["rk3576"] }
```

## API Overview

### Core Types

- **`RKNN<A>`**: Main struct for model management and inference
- **`Input`**: Represents model input tensors with various data types and formats
- **`Output`**: Handles model output retrieval with flexible buffer management
- **`Query`**: Trait for querying model metadata and attributes

### Key Methods

- **`RKNN::new()`**: Create RKNN context from model data
- **`set_inputs()`**: Configure model inputs before inference
- **`run()`**: Execute model inference
- **`get_outputs()`**: Retrieve inference results
- **`query()`**: Get model information and attributes

### Supported Data Types

- **Input**: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f16`, `f32`
- **Output**: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f16`, `f32`
- **Tensor Formats**: NCHW, NHWC, NC1HWC2

## Examples

The crate includes several examples demonstrating different use cases:

- **MobileNet Classification**: Complete image classification pipeline
- **Model Querying**: How to inspect model properties and performance
- **Custom Input/Output**: Advanced buffer management techniques

Run examples with:

```bash
# Basic MobileNet example
cargo run --example mobilenet --features rk35xx

# With runtime library loading
cargo run --example mobilenet --features "rk35xx,libloading"
```

## Performance Considerations

- **Zero-copy Buffers**: Use preallocated buffers when possible for optimal performance
- **Core Selection**: Configure appropriate NPU cores based on your workload
- **Batch Processing**: Leverage batch inference for multiple inputs
- **Memory Management**: Reuse input/output buffers across inference calls

## Troubleshooting

### Common Issues

1. **Library Not Found**: Ensure `librknnrt.so` is in your library path
2. **Feature Flags**: Verify you're using the correct feature flags for your NPU
3. **Model Format**: Ensure your model is in the correct RKNN format
4. **Memory Alignment**: Check that input/output buffers meet alignment requirements

## Development Status

This project is actively maintained and follows Rust best practices. Contributions are welcome!

## Contributing

Contributions are welcome! Please see our contributing guidelines for:

- Code style and formatting
- Testing requirements
- Documentation standards
- Feature request process

## Related Projects

- [rknpu2-sys](crates/rknpu2-sys/): Low-level FFI bindings
- [rktensor](crates/rktensor/): Tensor operations and utilities
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2): Official Python toolkit

## Support

For issues and questions:

- Check the [examples](crates/rknpu2/examples/) directory
- Review the [API documentation](https://docs.rs/rknpu2)
- Open an issue on GitHub
- Consult the [RKNN API reference](https://github.com/airockchip/rknn-toolkit2)

## License

This project is licensed under either of:

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

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

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
