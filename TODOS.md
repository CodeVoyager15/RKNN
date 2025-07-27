## TODO

### Ergonomics

- Some of the error types have integers as part of their error message. I should have a way to match these integers to strings.

### Documentation

- Most docstrings are missing
- I should use cfg_attr(feature = "docs", doc(cfg(feature = "..."))) on all the feature gates in the high-level bindings.

### Testing

- I should use cargo-tarpaulin to ensure code coverage

### API coverage

- Missing coverage of custom_ops, matmul_api
- Re-look at the API design of Tensor and TensorBuilder for Zero-Copy tensors
- Need TensorType implementations for all supported data types
