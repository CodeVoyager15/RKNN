#![cfg_attr(feature = "docs", feature(doc_cfg))]
#![allow(unused_imports)]
#![doc = include_str!("../README.md")]

#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
mod api;
#[allow(non_camel_case_types)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(non_snake_case)]
#[cfg(feature = "libloading")]
#[cfg_attr(feature = "docs", doc(cfg(feature = "libloading")))]
mod rt;

pub use api::*;
#[cfg(feature = "libloading")]
#[cfg_attr(feature = "docs", doc(cfg(feature = "libloading")))]
pub use rt::*;

#[cfg(not(feature = "libloading"))]
pub use api::functions::*;

#[cfg(not(feature = "libloading"))]
pub use api::functions_custom_ops::*;

#[cfg(not(feature = "libloading"))]
pub use api::functions_matmul::*;
