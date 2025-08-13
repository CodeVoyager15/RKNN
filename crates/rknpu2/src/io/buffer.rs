use {
    crate::tensor::{DataType, DataTypeKind},
    half::{bf16, f16},
    std::ffi::c_void,
};

/// A borrowed view into user owned data.
#[derive(Debug)]
pub enum BufMutView<'a> {
    F32(&'a mut [f32]),
    I32(&'a mut [i32]),
    U8(&'a mut [u8]),
    U16(&'a mut [u16]),
    U32(&'a mut [u32]),
    I64(&'a mut [i64]),
    F16(&'a mut [f16]),
    BF16(&'a mut [bf16]),
    I8(&'a mut [i8]),
}

impl<'a> BufMutView<'a> {
    pub fn len(&self) -> usize {
        match self {
            BufMutView::F32(data) => data.len(),
            BufMutView::I32(data) => data.len(),
            BufMutView::U8(data) => data.len(),
            BufMutView::U16(data) => data.len(),
            BufMutView::U32(data) => data.len(),
            BufMutView::I64(data) => data.len(),
            BufMutView::F16(data) => data.len(),
            BufMutView::BF16(data) => data.len(),
            BufMutView::I8(data) => data.len(),
        }
    }

    pub fn num_bytes(&self) -> usize {
        match self {
            BufMutView::F32(data) => data.len() * std::mem::size_of::<f32>(),
            BufMutView::I32(data) => data.len() * std::mem::size_of::<i32>(),
            BufMutView::U8(data) => data.len() * std::mem::size_of::<u8>(),
            BufMutView::U16(data) => data.len() * std::mem::size_of::<u16>(),
            BufMutView::U32(data) => data.len() * std::mem::size_of::<u32>(),
            BufMutView::I64(data) => data.len() * std::mem::size_of::<i64>(),
            BufMutView::F16(data) => data.len() * std::mem::size_of::<f16>(),
            BufMutView::BF16(data) => data.len() * std::mem::size_of::<bf16>(),
            BufMutView::I8(data) => data.len() * std::mem::size_of::<i8>(),
        }
    }

    pub fn dtype(&self) -> DataTypeKind {
        match self {
            BufMutView::F32(_) => DataTypeKind::Float32(DataType::FLOAT32),
            BufMutView::I32(_) => DataTypeKind::Int32(DataType::INT32),
            BufMutView::U8(_) => DataTypeKind::UInt8(DataType::UINT8),
            BufMutView::U16(_) => DataTypeKind::UInt16(DataType::UINT16),
            BufMutView::U32(_) => DataTypeKind::UInt32(DataType::UINT32),
            BufMutView::I64(_) => DataTypeKind::Int64(DataType::INT64),
            BufMutView::F16(_) => DataTypeKind::Float16(DataType::FLOAT16),
            BufMutView::BF16(_) => DataTypeKind::BFloat16(DataType::BFLOAT16),
            BufMutView::I8(_) => DataTypeKind::Int8(DataType::INT8),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        match self {
            BufMutView::F32(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::I32(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::U8(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::U16(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::U32(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::I64(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::F16(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::BF16(data) => data.as_mut_ptr() as *mut c_void,
            BufMutView::I8(data) => data.as_mut_ptr() as *mut c_void,
        }
    }
}

#[derive(Debug)]
pub enum BufView<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    I64(&'a [i64]),
    F16(&'a [f16]),
    BF16(&'a [bf16]),
    I8(&'a [i8]),
}

impl<'a> BufView<'a> {
    pub fn len(&self) -> usize {
        match self {
            BufView::F32(data) => data.len(),
            BufView::I32(data) => data.len(),
            BufView::U8(data) => data.len(),
            BufView::U16(data) => data.len(),
            BufView::U32(data) => data.len(),
            BufView::I64(data) => data.len(),
            BufView::F16(data) => data.len(),
            BufView::BF16(data) => data.len(),
            BufView::I8(data) => data.len(),
        }
    }

    pub fn num_bytes(&self) -> usize {
        match self {
            BufView::F32(data) => data.len() * std::mem::size_of::<f32>(),
            BufView::I32(data) => data.len() * std::mem::size_of::<i32>(),
            BufView::U8(data) => data.len() * std::mem::size_of::<u8>(),
            BufView::U16(data) => data.len() * std::mem::size_of::<u16>(),
            BufView::U32(data) => data.len() * std::mem::size_of::<u32>(),
            BufView::I64(data) => data.len() * std::mem::size_of::<i64>(),
            BufView::F16(data) => data.len() * std::mem::size_of::<f16>(),
            BufView::BF16(data) => data.len() * std::mem::size_of::<bf16>(),
            BufView::I8(data) => data.len() * std::mem::size_of::<i8>(),
        }
    }

    pub fn dtype(&self) -> DataTypeKind {
        match self {
            BufView::F32(_) => DataTypeKind::Float32(DataType::FLOAT32),
            BufView::I32(_) => DataTypeKind::Int32(DataType::INT32),
            BufView::U8(_) => DataTypeKind::UInt8(DataType::UINT8),
            BufView::U16(_) => DataTypeKind::UInt16(DataType::UINT16),
            BufView::U32(_) => DataTypeKind::UInt32(DataType::UINT32),
            BufView::I64(_) => DataTypeKind::Int64(DataType::INT64),
            BufView::F16(_) => DataTypeKind::Float16(DataType::FLOAT16),
            BufView::BF16(_) => DataTypeKind::BFloat16(DataType::BFLOAT16),
            BufView::I8(_) => DataTypeKind::Int8(DataType::INT8),
        }
    }

    /// Returns a pointer to the buffer for the C API.
    ///
    /// # Safety
    /// This performs a **const cast** from an immutable slice pointer (`*const T`)
    /// to a mutable void pointer (`*mut c_void`).
    ///
    /// The caller **must guarantee** that the C API function this pointer
    /// is passed to will **not** modify the memory it points to.
    pub fn as_mut_ptr(&self) -> *mut c_void {
        match self {
            BufView::F32(data) => data.as_ptr() as *mut c_void,
            BufView::I32(data) => data.as_ptr() as *mut c_void,
            BufView::U8(data) => data.as_ptr() as *mut c_void,
            BufView::U16(data) => data.as_ptr() as *mut c_void,
            BufView::U32(data) => data.as_ptr() as *mut c_void,
            BufView::I64(data) => data.as_ptr() as *mut c_void,
            BufView::F16(data) => data.as_ptr() as *mut c_void,
            BufView::BF16(data) => data.as_ptr() as *mut c_void,
            BufView::I8(data) => data.as_ptr() as *mut c_void,
        }
    }
}

pub struct RknnBuffer {
    pub(crate) ptr: *mut c_void,
    pub(crate) size: usize,
}

impl RknnBuffer {
    pub fn new(ptr: *mut c_void, size: usize) -> Self {
        RknnBuffer { ptr, size }
    }
}
