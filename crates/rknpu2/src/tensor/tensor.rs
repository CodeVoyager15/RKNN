use {
    crate::{
        Error,
        tensor::{StrideInfo, TensorFormatKind, TensorType},
    },
    std::{marker::PhantomData, ptr::NonNull},
};

#[derive(Debug)]
pub struct Tensor<T> {
    ptr: NonNull<T>,
    len: usize,
    _layout: TensorFormatKind,
    fmt: rknpu2_sys::rknn_tensor_format,
    index: u32,
    pass_through: bool,
    _stride_info: Option<StrideInfo>,
    _marker: PhantomData<T>,
    _buffer: Option<Box<[T]>>,
}

impl<T> Tensor<T> {
    pub(crate) unsafe fn from_raw_parts(
        ptr: *mut T,
        len: usize,
        index: u32,
        layout: TensorFormatKind,
        fmt: rknpu2_sys::rknn_tensor_format,
        pass_through: bool,
        stride_info: Option<StrideInfo>,
        buffer: Option<Box<[T]>>,
    ) -> Self {
        Self {
            ptr: NonNull::new(ptr)
                .expect("Tensor::from_raw_parts got null pointer when creating Tensor"),
            len,
            _layout: layout,
            fmt,
            index,
            pass_through,
            _stride_info: stride_info,
            _marker: PhantomData,
            _buffer: buffer,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn copy_data(&mut self, data: &[T]) -> Result<(), Error>
    where
        T: Copy,
    {
        if data.len() != self.len {
            return Err(Error::SizeMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), self.len);
        }

        Ok(())
    }

    pub fn fill_with(&mut self, value: T)
    where
        T: Copy,
    {
        self.as_mut_slice().fill(value);
    }

    pub(crate) fn as_input(&self) -> rknpu2_sys::rknn_input
    where
        T: TensorType,
    {
        rknpu2_sys::rknn_input {
            index: self.index,
            buf: self.ptr.as_ptr() as *mut _,
            size: (self.len * std::mem::size_of::<T>()) as u32,
            pass_through: self.pass_through as u8,
            type_: T::TYPE,
            fmt: self.fmt,
        }
    }

    pub(crate) fn as_output(&mut self) -> rknpu2_sys::rknn_output
    where
        T: TensorType,
    {
        rknpu2_sys::rknn_output {
            index: self.index,
            want_float: (T::TYPE == rknpu2_sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32) as u8,
            is_prealloc: 1,
            buf: self.ptr.as_ptr() as *mut _,
            size: (self.len * std::mem::size_of::<T>()) as u32,
        }
    }
}
