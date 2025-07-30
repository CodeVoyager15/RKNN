/// Model input attributes.
use {
    crate::{
        query::QueryWithInput,
        tensor::{DataTypeKind, QuantTypeKind, TensorFormatKind},
    },
    rknpu2_sys::{
        _rknn_query_cmd::{RKNN_QUERY_INPUT_ATTR, Type},
        rknn_tensor_attr,
    },
    std::ffi::CStr,
};

/// Query a specific input's attributes.
pub struct InputAttr {
    pub(crate) inner: rknn_tensor_attr,
}

impl InputAttr {
    /// Index of the input tensor.
    pub fn index(&self) -> u32 {
        self.inner.index
    }

    /// Number of dimensions of the input tensor.
    pub fn num_dims(&self) -> u32 {
        self.inner.n_dims
    }

    /// Dimensions of the input tensor.
    pub fn dims(&self) -> &[u32] {
        &self.inner.dims[..self.inner.n_dims as usize]
    }

    /// Name of the input tensor.
    pub fn name(&self) -> String {
        let cstr = unsafe { CStr::from_ptr(self.inner.name.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    /// Number of elements in the input tensor.
    pub fn num_elements(&self) -> u32 {
        self.inner.n_elems
    }

    /// Size of the input tensor in bytes.
    pub fn size(&self) -> u32 {
        self.inner.size
    }

    /// Format (Layout) of the input tensor.
    pub fn format(&self) -> TensorFormatKind {
        self.inner.fmt.into()
    }

    /// Data type of the input tensor.
    pub fn dtype(&self) -> DataTypeKind {
        self.inner.type_.into()
    }

    /// Quantization type of the input tensor.
    pub fn qnt_type(&self) -> QuantTypeKind {
        self.inner.qnt_type.into()
    }

    /// Fixed-point parameters of the input tensor.
    pub fn dfp_param(&self) -> i8 {
        self.inner.fl
    }

    /// Affine asymmetric parameters of the input tensor.
    pub fn affine_asymmetric_param(&self) -> f32 {
        self.inner.scale
    }
}

impl QueryWithInput for InputAttr {
    const QUERY_TYPE: Type = RKNN_QUERY_INPUT_ATTR;

    type Output = rknn_tensor_attr;
    type Input = u32;

    fn prepare(input: Self::Input, output: &mut Self::Output) {
        output.index = input;
    }
}

impl From<rknn_tensor_attr> for InputAttr {
    fn from(attr: rknn_tensor_attr) -> Self {
        Self { inner: attr }
    }
}
