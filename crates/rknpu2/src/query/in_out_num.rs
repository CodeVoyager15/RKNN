use crate::query::Query;
/// # of inputs and outputs in a model.
use rknpu2_sys::{
    _rknn_query_cmd::{RKNN_QUERY_IN_OUT_NUM, Type},
    rknn_input_output_num,
};

/// Query the number of inputs and outputs in a model.
pub struct InputOutputNum {
    pub(crate) inner: rknn_input_output_num,
}

impl InputOutputNum {
    /// Number of inputs in the model.
    pub fn input_num(&self) -> u32 {
        self.inner.n_input
    }

    /// Number of outputs in the model.
    pub fn output_num(&self) -> u32 {
        self.inner.n_output
    }
}

impl Query for InputOutputNum {
    const QUERY_TYPE: Type = RKNN_QUERY_IN_OUT_NUM;

    type Output = rknn_input_output_num;
}

impl From<rknn_input_output_num> for InputOutputNum {
    fn from(value: rknn_input_output_num) -> Self {
        InputOutputNum { inner: value }
    }
}
