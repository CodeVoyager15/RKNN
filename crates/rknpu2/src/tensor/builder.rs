use crate::{
    Error, RKNN,
    query::{InputAttr, output_attr::OutputAttr},
    tensor::{StrideInfo, TensorType, tensor::Tensor},
};

pub struct TensorBuilder<'a> {
    model: &'a RKNN,
    index: u32,
    kind: TensorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Input,
    Output,
}

impl<'a> TensorBuilder<'a> {
    pub fn new_input(model: &'a RKNN, index: u32) -> Self {
        Self {
            model,
            index,
            kind: TensorKind::Input,
        }
    }

    pub fn new_output(model: &'a RKNN, index: u32) -> Self {
        Self {
            model,
            index,
            kind: TensorKind::Output,
        }
    }

    pub fn allocate<T: TensorType + Copy>(self) -> Result<Tensor<T>, Error> {
        let attr = match self.kind {
            TensorKind::Input => self.model.query_with_input::<InputAttr>(self.index)?.inner,
            TensorKind::Output => self.model.query_with_input::<OutputAttr>(self.index)?.inner,
        };

        if attr.type_ != T::TYPE {
            return Err(Error::TensorTypeMismatch {
                expected: attr.type_,
                actual: T::TYPE,
            });
        }

        let len = attr.n_elems as usize;
        let mut buffer = vec![T::default(); len];

        unsafe {
            Ok(Tensor::<T>::from_raw_parts(
                buffer.as_mut_ptr(),
                len,
                self.index,
                attr.fmt.into(),
                attr.fmt,
                false,
                None,
                Some(buffer.into_boxed_slice()),
            ))
        }
    }

    pub unsafe fn from_mmap<T: TensorType>(
        self,
        ptr: *mut T,
        len: usize,
    ) -> Result<Tensor<T>, Error> {
        let attr = match self.kind {
            TensorKind::Input => self.model.query_with_input::<InputAttr>(self.index)?.inner,
            TensorKind::Output => self.model.query_with_input::<OutputAttr>(self.index)?.inner,
        };

        if attr.type_ != T::TYPE {
            return Err(Error::TensorTypeMismatch {
                expected: attr.type_,
                actual: T::TYPE,
            });
        }

        unsafe {
            Ok(Tensor::<T>::from_raw_parts(
                ptr,
                len,
                self.index,
                attr.fmt.into(),
                attr.fmt,
                true,
                Some(StrideInfo {
                    w_stride: attr.w_stride,
                    h_stride: attr.h_stride,
                    size_with_stride: attr.size_with_stride,
                }),
                None,
            ))
        }
    }
}
