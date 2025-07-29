/// Model output attributes.
use {
    crate::{
        query::QueryWithInput,
        tensor::{DataTypeKind, QuantTypeKind, TensorFormatKind},
    },
    rknpu2_sys::rknn_tensor_attr,
    std::ffi::CStr,
};

pub struct OutputAttr {
    pub(crate) inner: rknn_tensor_attr,
}

impl QueryWithInput for OutputAttr {
    const QUERY_TYPE: rknpu2_sys::_rknn_query_cmd::Type =
        rknpu2_sys::_rknn_query_cmd::RKNN_QUERY_OUTPUT_ATTR;

    type Output = rknn_tensor_attr;
    type Input = u32;

    fn prepare(input: Self::Input, output: &mut Self::Output) {
        output.index = input;
    }
}

impl From<rknn_tensor_attr> for OutputAttr {
    fn from(attr: rknn_tensor_attr) -> Self {
        Self { inner: attr }
    }
}

impl OutputAttr {
    pub fn index(&self) -> u32 {
        self.inner.index
    }

    pub fn num_dims(&self) -> u32 {
        self.inner.n_dims
    }

    pub fn dims(&self) -> &[u32] {
        &self.inner.dims[..self.inner.n_dims as usize]
    }

    pub fn name(&self) -> String {
        let cstr = unsafe { CStr::from_ptr(self.inner.name.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    pub fn num_elements(&self) -> u32 {
        self.inner.n_elems
    }

    pub fn size(&self) -> u32 {
        self.inner.size
    }

    pub fn format(&self) -> TensorFormatKind {
        self.inner.fmt.into()
    }

    pub fn dtype(&self) -> DataTypeKind {
        self.inner.type_.into()
    }

    pub fn qnt_type(&self) -> QuantTypeKind {
        self.inner.qnt_type.into()
    }

    pub fn dfp_param(&self) -> i8 {
        self.inner.fl
    }

    pub fn affine_asymmetric_param(&self) -> f32 {
        self.inner.scale
    }
}
