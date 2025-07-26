use {
    crate::{
        query::QueryWithInput,
        tensor::{DataTypeKind, QuantTypeKind, TensorFormatKind},
    },
    rknpu2_sys::{
        _rknn_query_cmd::{RKNN_QUERY_INPUT_ATTR, Type},
        rknn_tensor_attr,
    },
};

/// Query a specific input's attributes.
pub struct InputAttr {
    pub(crate) inner: rknn_tensor_attr,
}

impl InputAttr {
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
        String::from_utf8_lossy(&self.inner.name).into_owned()
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
