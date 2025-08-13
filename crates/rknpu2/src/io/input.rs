use {
    crate::{io::buffer::BufView, tensor::TensorFormatKind},
    rknpu2_sys::rknn_input,
};

#[derive(Debug)]
pub struct Input<'a> {
    pub index: u32,
    pub buffer: BufView<'a>,
    pub pass_through: bool,
    pub fmt: TensorFormatKind,
}

impl<'a> Input<'a> {
    pub fn new(index: u32, buffer: BufView<'a>, pass_through: bool, fmt: TensorFormatKind) -> Self {
        Input {
            index,
            buffer,
            pass_through,
            fmt,
        }
    }

    pub(crate) fn as_sys_input(&mut self) -> rknn_input {
        rknn_input {
            index: self.index,
            buf: self.buffer.as_mut_ptr(),
            size: self.buffer.num_bytes() as u32,
            pass_through: self.pass_through as u8,
            fmt: self.fmt.into(),
            type_: self.buffer.dtype().into(),
        }
    }
}

pub type Inputs<'a> = Vec<Input<'a>>;

pub trait IntoInputs<'a> {
    fn into_inputs(self) -> Inputs<'a>;
}

impl<'a> IntoInputs<'a> for Vec<Input<'a>> {
    fn into_inputs(self) -> Inputs<'a> {
        self
    }
}

impl<'a> IntoInputs<'a> for Input<'a> {
    fn into_inputs(self) -> Inputs<'a> {
        vec![self]
    }
}
