/// Error type
use rknpu2_sys::_rknn_tensor_type;

#[derive(Debug)]
pub enum Error {
    /// Execution error
    Fail,
    /// Execution timeout
    Timeout,
    /// NPU Device unavailable
    DeviceUnavailable,
    /// Memory allocation failed
    MallocFailed,
    /// Parameter error
    ParamInvalid,
    /// RKNN model is invalid
    ModelInvalid,
    /// Context is invalid
    CtxInvalid,
    /// Input is invalid
    InputInvalid,
    /// Output is invalid
    OutputInvalid,
    /// Version does not match
    DeviceUnmatch,
    /// RKNN model uses an optimization level mode
    /// that is not compatible with the target platform
    IncompatibleOptimizationLevelVersion,
    /// RKNN model isn't compatible with the target platform
    TargetPlatformUnmatch,
    IncompatiblePreCompiledModel,
    TensorTypeMismatch {
        expected: _rknn_tensor_type::Type,
        actual: _rknn_tensor_type::Type,
    },
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Fail => write!(f, "Execution error"),
            Error::Timeout => write!(f, "Execution timeout"),
            Error::DeviceUnavailable => write!(f, "NPU Device unavailable"),
            Error::MallocFailed => write!(f, "Memory allocation failed"),
            Error::ParamInvalid => write!(f, "Parameter error"),
            Error::ModelInvalid => write!(f, "RKNN model is invalid"),
            Error::CtxInvalid => write!(f, "Context is invalid"),
            Error::InputInvalid => write!(f, "Input is invalid"),
            Error::OutputInvalid => write!(f, "Output is invalid"),
            Error::DeviceUnmatch => write!(f, "Version does not match"),
            Error::IncompatibleOptimizationLevelVersion => write!(
                f,
                "RKNN model uses an optimization level mode that is not compatible with the target platform"
            ),
            Error::IncompatiblePreCompiledModel => write!(
                f,
                "RKNN model uses a pre-compiled model that is not compatible with the target platform"
            ),
            Error::TargetPlatformUnmatch => {
                write!(f, "RKNN model isn't compatible with the target platform")
            }
            Error::TensorTypeMismatch { expected, actual } => {
                write!(
                    f,
                    "Tensor type mismatch: expected {:?}, actual {:?}",
                    expected, actual
                )
            }
            Error::SizeMismatch { expected, actual } => {
                write!(f, "Size mismatch: expected {}, actual {}", expected, actual)
            }
        }
    }
}

impl From<std::ffi::c_int> for Error {
    fn from(err: std::ffi::c_int) -> Self {
        match err {
            rknpu2_sys::RKNN_ERR_FAIL => Error::Fail,
            rknpu2_sys::RKNN_ERR_TIMEOUT => Error::Timeout,
            rknpu2_sys::RKNN_ERR_DEVICE_UNAVAILABLE => Error::DeviceUnavailable,
            rknpu2_sys::RKNN_ERR_MALLOC_FAIL => Error::MallocFailed,
            rknpu2_sys::RKNN_ERR_PARAM_INVALID => Error::ParamInvalid,
            rknpu2_sys::RKNN_ERR_MODEL_INVALID => Error::ModelInvalid,
            rknpu2_sys::RKNN_ERR_CTX_INVALID => Error::CtxInvalid,
            rknpu2_sys::RKNN_ERR_INPUT_INVALID => Error::InputInvalid,
            rknpu2_sys::RKNN_ERR_OUTPUT_INVALID => Error::OutputInvalid,
            rknpu2_sys::RKNN_ERR_DEVICE_UNMATCH => Error::DeviceUnmatch,
            rknpu2_sys::RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION => {
                Error::IncompatibleOptimizationLevelVersion
            }
            rknpu2_sys::RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL => {
                Error::IncompatiblePreCompiledModel
            }
            rknpu2_sys::RKNN_ERR_TARGET_PLATFORM_UNMATCH => Error::TargetPlatformUnmatch,
            _ => Error::Fail,
        }
    }
}
