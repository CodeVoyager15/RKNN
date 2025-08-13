#![cfg_attr(feature = "docs", feature(doc_cfg))]
#![doc = include_str!("../README.md")]

/// Error type
pub mod error;

/// Query types
pub mod query;

/// Main RKNN struct
pub mod rknn;

/// Input and output types
pub mod io;

/// Tensor types
pub mod tensor;

/// Utility functions
pub mod utils;

pub use {error::Error, rknn::RKNN};

/// Trait and implementation for RKNN
#[allow(non_snake_case)]
pub mod api;

pub use {
    half::{bf16, f16},
    rknpu2_sys,
};
