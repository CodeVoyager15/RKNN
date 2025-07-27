#![cfg_attr(feature = "docs", feature(doc_cfg))]

pub mod error;
pub mod query;
pub mod rknn;
pub mod tensor;

pub use {error::Error, rknn::RKNN};
