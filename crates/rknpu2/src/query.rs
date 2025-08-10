use std::fmt::Display;

/// Query types for the query associated function.
use rknpu2_sys::_rknn_query_cmd::Type;

pub trait Query: From<Self::Output> + Sized {
    const QUERY_TYPE: Type;

    type Output;
}

pub mod in_out_num;
pub mod input_attr;
pub mod native_input_attr;
pub mod native_nc1hwc2_input_attr;
pub mod native_nc1hwc2_output_attr;
pub mod native_nhwc_input_attr;
pub mod native_nhwc_output_attr;
pub mod native_output_attr;
pub mod output_attr;
pub mod sdk_version;

#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
#[cfg_attr(
    feature = "docs",
    doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
)]
pub mod perf_run;

#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
#[cfg_attr(
    feature = "docs",
    doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
)]
pub mod perf_detail;

use crate::tensor::{DataTypeKind, QuantTypeKind, TensorFormatKind};

pub use {
    in_out_num::InputOutputNum, native_input_attr::NativeInputAttr,
    native_nc1hwc2_input_attr::NativeNC1HWC2InputAttr,
    native_nc1hwc2_output_attr::NativeNC1HWC2OutputAttr,
    native_nhwc_input_attr::NativeNHWCInputAttr, native_nhwc_output_attr::NativeNHWCOutputAttr,
    native_output_attr::NativeOutputAttr, sdk_version::SdkVersion,
};

#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
#[cfg_attr(
    feature = "docs",
    doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
)]
pub use perf_run::PerfRun;

#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
#[cfg_attr(
    feature = "docs",
    doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
)]
pub use perf_detail::PerfDetail;

pub trait QueryWithInput: From<Self::Output> + Sized {
    const QUERY_TYPE: Type;

    type Output;

    type Input;

    fn prepare(input: Self::Input, output: &mut Self::Output);
}

pub use {input_attr::InputAttr, output_attr::OutputAttr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Io {
    Input,
    Output,
}

impl Display for Io {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Io::Input => write!(f, "Input"),
            Io::Output => write!(f, "Output"),
        }
    }
}

impl Io {
    pub fn is_input(&self) -> bool {
        matches!(self, Io::Input)
    }

    pub fn is_output(&self) -> bool {
        matches!(self, Io::Output)
    }
}

/// Tensor attribute view trait
pub trait TensorAttrView {
    fn io(&self) -> Io;

    /// Tensor index
    fn index(&self) -> u32;
    /// Tensor name
    fn name(&self) -> String;
    /// Tensor data type
    fn dtype(&self) -> DataTypeKind;

    /// Number of dimensions
    fn num_dims(&self) -> u32;

    /// Tensor dimensions
    fn dims(&self) -> &[u32];

    /// Tensor format
    fn format(&self) -> TensorFormatKind;

    /// Quantization type
    fn qnt_type(&self) -> QuantTypeKind;

    /// Number of elements
    fn num_elements(&self) -> u32;

    /// Scale factor
    fn scale(&self) -> f32;
    /// Zero point
    fn zero_point(&self) -> i32;
    /// Fractional length
    fn fl(&self) -> i8;
    /// pixels per row (width + padding), unit: pixel
    fn w_stride(&self) -> u32;

    /// rows per image (height + padding), unit: pixel; 0 => same as H
    fn h_stride(&self) -> u32;
    /// Size in bytes
    fn size(&self) -> u32;
    /// Size in bytes
    fn size_with_stride(&self) -> u32;
}
