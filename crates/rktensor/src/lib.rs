//! rktensor has utilities for converting to flat tensor representations for
//! use with rknpu2.
//!
//! The main functions are `to_tensor` and `to_tensor_with_quant`.

pub use image;

pub mod implementation;
pub mod markers;
pub mod softmax;

pub use {
    implementation::{to_tensor, to_tensor_with_quant},
    softmax::{softmax_f16, softmax_f32},
};
