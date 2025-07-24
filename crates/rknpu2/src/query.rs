use rknpu2_sys::_rknn_query_cmd::Type;

pub trait Query: From<Self::Output> + Sized {
    const QUERY_TYPE: Type;

    type Output;
}

pub mod in_out_num;
pub mod input_attr;
pub mod output_attr;
pub mod sdk_version;

pub use {in_out_num::InputOutputNum, sdk_version::SdkVersion};

pub trait QueryWithInput: From<Self::Output> + Sized {
    const QUERY_TYPE: Type;

    type Output;

    type Input;

    fn prepare(input: Self::Input, output: &mut Self::Output);
}

pub use input_attr::InputAttr;
