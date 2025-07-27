use rknpu2_sys::_rknn_query_cmd::Type;

pub trait Query: From<Self::Output> + Sized {
    const QUERY_TYPE: Type;

    type Output;
}

pub mod in_out_num;
pub mod input_attr;
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

pub use {in_out_num::InputOutputNum, sdk_version::SdkVersion};

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
