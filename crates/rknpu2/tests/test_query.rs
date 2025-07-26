use rknpu2::{
    RKNN,
    query::{InputAttr, InputOutputNum, SdkVersion, output_attr::OutputAttr},
    tensor::{DataType, DataTypeKind, QuantType, QuantTypeKind},
};

static MODEL_DATA: &[u8] = include_bytes!("./fixtures/mobilenet_v2.rknn");

#[test]
fn test_input_output_num() {
    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new(&mut model_data, 0).unwrap();
    let io_num = rknn.query::<InputOutputNum>().unwrap();

    assert_eq!(io_num.input_num(), 1);
    assert_eq!(io_num.output_num(), 1);
}

#[test]
fn test_sdk_version() {
    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new(&mut model_data, 0).unwrap();
    let sdk_version = rknn.query::<SdkVersion>().unwrap();

    assert!(!sdk_version.api_version().is_empty());
    assert!(!sdk_version.driver_version().is_empty());
}

#[test]
fn test_input_attr() {
    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new(&mut model_data, 0).unwrap();
    let input_attr = rknn.query_with_input::<InputAttr>(0).unwrap();

    assert_eq!(input_attr.dims(), &[1, 224, 224, 3]);
    assert!(!input_attr.name().is_empty());
    assert_eq!(input_attr.dtype(), DataTypeKind::Int8(DataType::INT8));
}

#[test]
fn test_output_attr() {
    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new(&mut model_data, 0).unwrap();
    let output_attr = rknn.query_with_input::<OutputAttr>(0).unwrap();

    assert_eq!(output_attr.dims(), &[1, 1000]);
    assert!(!output_attr.name().is_empty());
    assert_eq!(output_attr.dtype(), DataTypeKind::Int8(DataType::INT8));
    assert_eq!(
        output_attr.qnt_type(),
        QuantTypeKind::AffineAsymmetric(QuantType::QNT_AFFINE_ASYMMETRIC)
    );
}
