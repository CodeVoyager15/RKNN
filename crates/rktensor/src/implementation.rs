use std::fmt;
use std::error::Error;

/// Tensor error type
#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch,
    InvalidType,
    UnsafeOperation,
    Other(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch => write!(f, "Shape mismatch"),
            TensorError::InvalidType => write!(f, "Invalid tensor type"),
            TensorError::UnsafeOperation => write!(f, "Unsafe operation attempted"),
            TensorError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for TensorError {}

/// Improved Tensor struct
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(TensorError::ShapeMismatch);
        }
        Ok(Tensor { data, shape })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }
}

// Usage of unsafe blocks should be minimized.
// Example: replace unsafe indexing with safe methods.
impl<T> Tensor<T> {
    pub fn get(&self, idx: usize) -> Result<&T, TensorError> {
        self.data.get(idx).ok_or(TensorError::Other("Index out of bounds".to_string()))
    }
}

// Any FFI or memory manipulation should be wrapped safely and errors propagated.