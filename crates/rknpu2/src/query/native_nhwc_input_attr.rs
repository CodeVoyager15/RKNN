use {
    crate::{
        query::{Io, QueryWithInput, TensorAttrView},
        tensor::{DataTypeKind, QuantTypeKind, TensorFormatKind},
    },
    rknpu2_sys::rknn_tensor_attr,
    std::ffi::CStr,
};

pub struct NativeNHWCInputAttr {
    pub(crate) inner: rknn_tensor_attr,
}

impl QueryWithInput for NativeNHWCInputAttr {
    const QUERY_TYPE: rknpu2_sys::_rknn_query_cmd::Type =
        rknpu2_sys::_rknn_query_cmd::RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR;

    type Input = u32;
    type Output = rknn_tensor_attr;

    fn prepare(input: Self::Input, output: &mut Self::Output) {
        output.index = input;
    }
}

impl From<rknn_tensor_attr> for NativeNHWCInputAttr {
    fn from(attr: rknn_tensor_attr) -> Self {
        Self { inner: attr }
    }
}

impl TensorAttrView for NativeNHWCInputAttr {
    fn io(&self) -> Io {
        Io::Input
    }

    fn index(&self) -> u32 {
        self.inner.index
    }

    fn num_dims(&self) -> u32 {
        self.inner.n_dims
    }

    fn dims(&self) -> &[u32] {
        &self.inner.dims[..self.inner.n_dims as usize]
    }

    fn name(&self) -> String {
        let cstr = unsafe { CStr::from_ptr(self.inner.name.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    fn num_elements(&self) -> u32 {
        self.inner.n_elems
    }

    fn size(&self) -> u32 {
        self.inner.size
    }

    fn format(&self) -> TensorFormatKind {
        self.inner.fmt.into()
    }

    fn dtype(&self) -> DataTypeKind {
        self.inner.type_.into()
    }

    fn qnt_type(&self) -> QuantTypeKind {
        self.inner.qnt_type.into()
    }

    fn fl(&self) -> i8 {
        self.inner.fl
    }

    fn scale(&self) -> f32 {
        self.inner.scale
    }

    fn zero_point(&self) -> i32 {
        self.inner.zp
    }

    fn h_stride(&self) -> u32 {
        self.inner.h_stride
    }

    fn w_stride(&self) -> u32 {
        self.inner.w_stride
    }

    fn size_with_stride(&self) -> u32 {
        self.inner.size_with_stride
    }
}
