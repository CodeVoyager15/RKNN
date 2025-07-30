#![cfg_attr(feature = "docs", feature(doc_cfg))]
#![doc = include_str!("../README.md")]

/// Error type
pub mod error;

/// Query types
pub mod query;

/// Main RKNN struct
pub mod rknn;

/// Tensor types
pub mod tensor;

pub use {error::Error, rknn::RKNN};

/// Trait and implementation for RKNN
#[allow(non_snake_case)]
pub mod api;

pub use rknpu2_sys;
