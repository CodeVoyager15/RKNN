use {
    half::f16,
    rknpu2_sys::{
        _rknn_tensor_format::{
            self, RKNN_TENSOR_FORMAT_MAX, RKNN_TENSOR_NC1HWC2, RKNN_TENSOR_NCHW, RKNN_TENSOR_NHWC,
            RKNN_TENSOR_UNDEFINED,
        },
        _rknn_tensor_qnt_type,
        _rknn_tensor_type::{
            self, RKNN_TENSOR_BFLOAT16, RKNN_TENSOR_BOOL, RKNN_TENSOR_FLOAT16, RKNN_TENSOR_FLOAT32,
            RKNN_TENSOR_INT4, RKNN_TENSOR_INT8, RKNN_TENSOR_INT16, RKNN_TENSOR_INT32,
            RKNN_TENSOR_INT64, RKNN_TENSOR_TYPE_MAX, RKNN_TENSOR_UINT8, RKNN_TENSOR_UINT16,
            RKNN_TENSOR_UINT32,
        },
    },
};

pub mod builder;
pub mod tensor;

#[derive(Debug)]
pub struct TensorFormat;

impl TensorFormat {
    pub const NCHW: _rknn_tensor_format::Type = RKNN_TENSOR_NCHW;
    pub const NHWC: _rknn_tensor_format::Type = RKNN_TENSOR_NHWC;
    pub const NC1HWC2: _rknn_tensor_format::Type = RKNN_TENSOR_NC1HWC2;
    pub const UNDEFINED: _rknn_tensor_format::Type = RKNN_TENSOR_UNDEFINED;
    pub const MAX: _rknn_tensor_format::Type = RKNN_TENSOR_FORMAT_MAX;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TensorFormatKind {
    NCHW(_rknn_tensor_format::Type),
    NHWC(_rknn_tensor_format::Type),
    NC1HWC2(_rknn_tensor_format::Type),
    UNDEFINED(_rknn_tensor_format::Type),
    Max(_rknn_tensor_format::Type),
    Other(_rknn_tensor_format::Type),
}

impl From<_rknn_tensor_format::Type> for TensorFormatKind {
    fn from(format: _rknn_tensor_format::Type) -> Self {
        match format {
            RKNN_TENSOR_NCHW => TensorFormatKind::NCHW(format),
            RKNN_TENSOR_NHWC => TensorFormatKind::NHWC(format),
            RKNN_TENSOR_NC1HWC2 => TensorFormatKind::NC1HWC2(format),
            RKNN_TENSOR_UNDEFINED => TensorFormatKind::UNDEFINED(format),
            RKNN_TENSOR_FORMAT_MAX => TensorFormatKind::Max(format),
            _ => TensorFormatKind::Other(format),
        }
    }
}

impl From<TensorFormatKind> for _rknn_tensor_format::Type {
    fn from(format: TensorFormatKind) -> Self {
        match format {
            TensorFormatKind::NCHW(_) => RKNN_TENSOR_NCHW,
            TensorFormatKind::NHWC(_) => RKNN_TENSOR_NHWC,
            TensorFormatKind::NC1HWC2(_) => RKNN_TENSOR_NC1HWC2,
            TensorFormatKind::UNDEFINED(_) => RKNN_TENSOR_UNDEFINED,
            TensorFormatKind::Max(_) => RKNN_TENSOR_FORMAT_MAX,
            TensorFormatKind::Other(a) => a,
        }
    }
}

pub struct DataType;

impl DataType {
    pub const FLOAT32: _rknn_tensor_type::Type = RKNN_TENSOR_FLOAT32;
    pub const FLOAT16: _rknn_tensor_type::Type = RKNN_TENSOR_FLOAT16;
    pub const BFLOAT16: _rknn_tensor_type::Type = RKNN_TENSOR_BFLOAT16;
    pub const INT4: _rknn_tensor_type::Type = RKNN_TENSOR_INT4;
    pub const INT8: _rknn_tensor_type::Type = RKNN_TENSOR_INT8;
    pub const UINT8: _rknn_tensor_type::Type = RKNN_TENSOR_UINT8;
    pub const INT16: _rknn_tensor_type::Type = RKNN_TENSOR_INT16;
    pub const UINT16: _rknn_tensor_type::Type = RKNN_TENSOR_UINT16;
    pub const INT32: _rknn_tensor_type::Type = RKNN_TENSOR_INT32;
    pub const UINT32: _rknn_tensor_type::Type = RKNN_TENSOR_UINT32;
    pub const INT64: _rknn_tensor_type::Type = RKNN_TENSOR_INT64;
    pub const BOOL: _rknn_tensor_type::Type = RKNN_TENSOR_BOOL;
    pub const MAX: _rknn_tensor_type::Type = RKNN_TENSOR_TYPE_MAX;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DataTypeKind {
    Float32(_rknn_tensor_type::Type),
    Float16(_rknn_tensor_type::Type),
    BFloat16(_rknn_tensor_type::Type),
    Int4(_rknn_tensor_type::Type),
    Int8(_rknn_tensor_type::Type),
    UInt8(_rknn_tensor_type::Type),
    Int16(_rknn_tensor_type::Type),
    UInt16(_rknn_tensor_type::Type),
    Int32(_rknn_tensor_type::Type),
    UInt32(_rknn_tensor_type::Type),
    Int64(_rknn_tensor_type::Type),
    Bool(_rknn_tensor_type::Type),
    Max(_rknn_tensor_type::Type),
    Other(_rknn_tensor_type::Type),
}

impl From<_rknn_tensor_type::Type> for DataTypeKind {
    fn from(data_type: _rknn_tensor_type::Type) -> Self {
        match data_type {
            RKNN_TENSOR_FLOAT32 => DataTypeKind::Float32(RKNN_TENSOR_FLOAT32),
            RKNN_TENSOR_FLOAT16 => DataTypeKind::Float16(RKNN_TENSOR_FLOAT16),
            RKNN_TENSOR_BFLOAT16 => DataTypeKind::BFloat16(RKNN_TENSOR_BFLOAT16),
            RKNN_TENSOR_INT4 => DataTypeKind::Int4(RKNN_TENSOR_INT4),
            RKNN_TENSOR_INT8 => DataTypeKind::Int8(RKNN_TENSOR_INT8),
            RKNN_TENSOR_UINT8 => DataTypeKind::UInt8(RKNN_TENSOR_UINT8),
            RKNN_TENSOR_INT16 => DataTypeKind::Int16(RKNN_TENSOR_INT16),
            RKNN_TENSOR_UINT16 => DataTypeKind::UInt16(RKNN_TENSOR_UINT16),
            RKNN_TENSOR_INT32 => DataTypeKind::Int32(RKNN_TENSOR_INT32),
            RKNN_TENSOR_UINT32 => DataTypeKind::UInt32(RKNN_TENSOR_UINT32),
            RKNN_TENSOR_INT64 => DataTypeKind::Int64(RKNN_TENSOR_INT64),
            RKNN_TENSOR_BOOL => DataTypeKind::Bool(RKNN_TENSOR_BOOL),
            RKNN_TENSOR_TYPE_MAX => DataTypeKind::Max(RKNN_TENSOR_TYPE_MAX),
            _ => DataTypeKind::Other(data_type),
        }
    }
}

impl From<DataTypeKind> for _rknn_tensor_type::Type {
    fn from(data_type: DataTypeKind) -> Self {
        match data_type {
            DataTypeKind::Float32(_) => RKNN_TENSOR_FLOAT32,
            DataTypeKind::Float16(_) => RKNN_TENSOR_FLOAT16,
            DataTypeKind::BFloat16(_) => RKNN_TENSOR_BFLOAT16,
            DataTypeKind::Int4(_) => RKNN_TENSOR_INT4,
            DataTypeKind::Int8(_) => RKNN_TENSOR_INT8,
            DataTypeKind::UInt8(_) => RKNN_TENSOR_UINT8,
            DataTypeKind::Int16(_) => RKNN_TENSOR_INT16,
            DataTypeKind::UInt16(_) => RKNN_TENSOR_UINT16,
            DataTypeKind::Int32(_) => RKNN_TENSOR_INT32,
            DataTypeKind::UInt32(_) => RKNN_TENSOR_UINT32,
            DataTypeKind::Int64(_) => RKNN_TENSOR_INT64,
            DataTypeKind::Bool(_) => RKNN_TENSOR_BOOL,
            DataTypeKind::Max(_) => RKNN_TENSOR_TYPE_MAX,
            DataTypeKind::Other(a) => a,
        }
    }
}

#[derive(Debug)]
pub struct QuantType;

impl QuantType {
    pub const QNT_NONE: _rknn_tensor_qnt_type::Type = _rknn_tensor_qnt_type::RKNN_TENSOR_QNT_NONE;
    pub const QNT_DFP: _rknn_tensor_qnt_type::Type = _rknn_tensor_qnt_type::RKNN_TENSOR_QNT_DFP;
    pub const QNT_AFFINE_ASYMMETRIC: _rknn_tensor_qnt_type::Type =
        _rknn_tensor_qnt_type::RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTypeKind {
    None(_rknn_tensor_qnt_type::Type),
    Dfp(_rknn_tensor_qnt_type::Type),
    AffineAsymmetric(_rknn_tensor_qnt_type::Type),
    Other(_rknn_tensor_qnt_type::Type),
}

impl From<QuantTypeKind> for _rknn_tensor_qnt_type::Type {
    fn from(quant_type: QuantTypeKind) -> Self {
        match quant_type {
            QuantTypeKind::None(_) => QuantType::QNT_NONE,
            QuantTypeKind::Dfp(_) => QuantType::QNT_DFP,
            QuantTypeKind::AffineAsymmetric(_) => QuantType::QNT_AFFINE_ASYMMETRIC,
            QuantTypeKind::Other(a) => a,
        }
    }
}

impl From<_rknn_tensor_qnt_type::Type> for QuantTypeKind {
    fn from(quant_type: _rknn_tensor_qnt_type::Type) -> Self {
        match quant_type {
            QuantType::QNT_NONE => QuantTypeKind::None(QuantType::QNT_NONE),
            QuantType::QNT_DFP => QuantTypeKind::Dfp(QuantType::QNT_DFP),
            QuantType::QNT_AFFINE_ASYMMETRIC => {
                QuantTypeKind::AffineAsymmetric(QuantType::QNT_AFFINE_ASYMMETRIC)
            }
            _ => QuantTypeKind::Other(quant_type),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StrideInfo {
    pub w_stride: u32,
    pub h_stride: u32,
    pub size_with_stride: u32,
}

pub trait TensorType: Sized + Default {
    const TYPE: _rknn_tensor_type::Type;
}

impl TensorType for f32 {
    const TYPE: _rknn_tensor_type::Type = _rknn_tensor_type::RKNN_TENSOR_FLOAT32;
}

impl TensorType for f16 {
    const TYPE: _rknn_tensor_type::Type = _rknn_tensor_type::RKNN_TENSOR_FLOAT16;
}

impl TensorType for u8 {
    const TYPE: _rknn_tensor_type::Type = _rknn_tensor_type::RKNN_TENSOR_UINT8;
}

impl TensorType for i8 {
    const TYPE: _rknn_tensor_type::Type = _rknn_tensor_type::RKNN_TENSOR_INT8;
}

impl TensorType for i32 {
    const TYPE: _rknn_tensor_type::Type = _rknn_tensor_type::RKNN_TENSOR_INT32;
}
