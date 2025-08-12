use {
    crate::{
        Error,
        tensor::{TensorFormatKind, TensorType},
    },
    std::ptr::NonNull,
};

/// An input or output tensor of type T.
#[derive(Debug)]
pub struct Tensor<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: TensorFormatKind,
    index: u32,
    pass_through: bool,
    buffer: Option<Box<[T]>>,
}

/// Clone implementation for Tensor<T>
impl<T: Clone> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        // The extra code is necessary for deep copying the tensor's buffer.
        // It ensures that the cloned tensor has its own independent buffer.

        let buffer = self.buffer.as_ref().map(|b| b.clone()); // deep copy if present

        let ptr = match &buffer {
            Some(b) => NonNull::new(b.as_ptr() as *mut T).unwrap(),
            None => self.ptr,
        };

        Tensor {
            ptr,
            len: self.len,
            layout: self.layout,
            index: self.index,
            pass_through: self.pass_through,
            buffer: buffer,
        }
    }
}

impl<T> Tensor<T> {
    pub(crate) unsafe fn from_raw_parts(
        ptr: *mut T,
        len: usize,
        index: u32,
        layout: TensorFormatKind,
        pass_through: bool,
        buffer: Option<Box<[T]>>,
    ) -> Self {
        Self {
            ptr: NonNull::new(ptr)
                .expect("Tensor::from_raw_parts got null pointer when creating Tensor"),
            len,
            layout,
            index,
            pass_through,
            buffer,
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
            fmt: self.layout.into(),
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
