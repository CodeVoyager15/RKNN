use {
    crate::{
        query::QueryWithInput,
        tensor::{DataTypeKind, QuantTypeKind, TensorFormatKind},
    },
    rknpu2_sys::rknn_tensor_attr,
    std::ffi::CStr,
};

pub struct NativeNC1HWC2OutputAttr {
    pub(crate) inner: rknn_tensor_attr,
}

impl QueryWithInput for NativeNC1HWC2OutputAttr {
    const QUERY_TYPE: rknpu2_sys::_rknn_query_cmd::Type =
        rknpu2_sys::_rknn_query_cmd::RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR;

    type Input = u32;
    type Output = rknn_tensor_attr;

    fn prepare(input: Self::Input, output: &mut Self::Output) {
        output.index = input;
    }
}

impl From<rknn_tensor_attr> for NativeNC1HWC2OutputAttr {
    fn from(attr: rknn_tensor_attr) -> Self {
        Self { inner: attr }
    }
}

impl NativeNC1HWC2OutputAttr {
    /// Index of the output tensor.
    pub fn index(&self) -> u32 {
        self.inner.index
    }

    /// Number of dimensions of the output tensor.
    pub fn num_dims(&self) -> u32 {
        self.inner.n_dims
    }

    /// Dimensions of the output tensor.
    pub fn dims(&self) -> &[u32] {
        &self.inner.dims[..self.inner.n_dims as usize]
    }

    /// Name of the output tensor.
    pub fn name(&self) -> String {
        let cstr = unsafe { CStr::from_ptr(self.inner.name.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    /// Number of elements in the output tensor.
    pub fn num_elements(&self) -> u32 {
        self.inner.n_elems
    }

    /// Size of the output tensor in bytes.
    pub fn size(&self) -> u32 {
        self.inner.size
    }

    /// Format (Layout) of the output tensor.
    pub fn format(&self) -> TensorFormatKind {
        self.inner.fmt.into()
    }

    /// Data type of the output tensor.
    pub fn dtype(&self) -> DataTypeKind {
        self.inner.type_.into()
    }

    /// Quantization type of the output tensor.
    pub fn qnt_type(&self) -> QuantTypeKind {
        self.inner.qnt_type.into()
    }

    /// Fixed-point parameters of the output tensor.
    pub fn dfp_param(&self) -> i8 {
        self.inner.fl
    }

    /// Affine asymmetric parameters of the output tensor.
    pub fn affine_asymmetric_param(&self) -> f32 {
        self.inner.scale
    }

    pub fn w_stride(&self) -> u32 {
        self.inner.w_stride
    }

    pub fn size_with_stride(&self) -> u32 {
        self.inner.size_with_stride
    }
}
