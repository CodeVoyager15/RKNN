#[cfg(feature = "libloading")]
use rknpu2::utils;
use rknpu2::{
    RKNN,
    query::{
        InputAttr, InputOutputNum, NativeInputAttr, NativeNHWCInputAttr, NativeNHWCOutputAttr,
        NativeOutputAttr, SdkVersion, output_attr::OutputAttr,
    },
    tensor::{DataType, DataTypeKind, QuantType, QuantTypeKind},
};

static MODEL_DATA: &[u8] = include_bytes!("./fixtures/mobilenet_v2.rknn");

#[cfg(not(feature = "libloading"))]
use rknpu2::api::linked::LinkedAPI;

#[cfg(not(feature = "libloading"))]
fn get_rknn() -> RKNN<LinkedAPI> {
    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new(&mut model_data, 0).unwrap();
    rknn
}

#[cfg(feature = "libloading")]
use rknpu2::api::runtime::RuntimeAPI;

#[cfg(feature = "libloading")]
fn get_rknn() -> RKNN<RuntimeAPI> {
    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new_with_library(
        utils::find_rknn_library()
            .next()
            .expect("No RKNN library found. Please install librknnrt.so."),
        &mut model_data,
        0,
    )
    .unwrap();
    rknn
}

#[test]
fn test_input_output_num() {
    let rknn = get_rknn();

    let io_num = rknn.query::<InputOutputNum>().unwrap();

    assert_eq!(io_num.input_num(), 1);
    assert_eq!(io_num.output_num(), 1);
}

#[test]
fn test_sdk_version() {
    let rknn = get_rknn();
    let sdk_version = rknn.query::<SdkVersion>().unwrap();

    assert!(!sdk_version.api_version().is_empty());
    assert!(!sdk_version.driver_version().is_empty());
}

#[test]
fn test_input_attr() {
    let rknn = get_rknn();
    let input_attr = rknn.query_with_input::<InputAttr>(0).unwrap();

    assert_eq!(input_attr.dims(), &[1, 224, 224, 3]);
    assert!(!input_attr.name().is_empty());
    assert_eq!(input_attr.dtype(), DataTypeKind::Int8(DataType::INT8));
}

#[test]
fn test_output_attr() {
    let rknn = get_rknn();
    let output_attr = rknn.query_with_input::<OutputAttr>(0).unwrap();

    assert_eq!(output_attr.dims(), &[1, 1000]);
    assert!(!output_attr.name().is_empty());
    assert_eq!(output_attr.dtype(), DataTypeKind::Int8(DataType::INT8));
    assert_eq!(
        output_attr.qnt_type(),
        QuantTypeKind::AffineAsymmetric(QuantType::QNT_AFFINE_ASYMMETRIC)
    );
}

#[test]
fn test_native_input_attr() {
    let rknn = get_rknn();
    let native_input_attr = rknn.query_with_input::<NativeInputAttr>(0).unwrap();

    assert_eq!(native_input_attr.dims(), &[1, 224, 224, 3]);
    assert!(!native_input_attr.name().is_empty());
    assert_eq!(
        native_input_attr.dtype(),
        DataTypeKind::Int8(DataType::INT8)
    );
    assert_eq!(
        native_input_attr.qnt_type(),
        QuantTypeKind::AffineAsymmetric(QuantType::QNT_AFFINE_ASYMMETRIC)
    );
}

#[test]
fn test_native_output_attr() {
    let rknn = get_rknn();
    let native_output_attr = rknn.query_with_input::<NativeOutputAttr>(0).unwrap();

    assert_eq!(native_output_attr.dims(), &[1, 1000]);
    assert!(!native_output_attr.name().is_empty());
    assert_eq!(
        native_output_attr.dtype(),
        DataTypeKind::Int8(DataType::INT8)
    );
    assert_eq!(
        native_output_attr.qnt_type(),
        QuantTypeKind::AffineAsymmetric(QuantType::QNT_AFFINE_ASYMMETRIC)
    );
}

#[test]
fn test_native_nhwc_input_attr() {
    let rknn = get_rknn();
    let native_nhwc_input_attr = rknn.query_with_input::<NativeNHWCInputAttr>(0).unwrap();

    assert_eq!(native_nhwc_input_attr.dims(), &[1, 224, 224, 3]);
    assert!(!native_nhwc_input_attr.name().is_empty());
    assert_eq!(
        native_nhwc_input_attr.dtype(),
        DataTypeKind::Int8(DataType::INT8)
    );
    assert_eq!(
        native_nhwc_input_attr.qnt_type(),
        QuantTypeKind::AffineAsymmetric(QuantType::QNT_AFFINE_ASYMMETRIC)
    );
}

#[test]
fn test_native_nhwc_output_attr() {
    let rknn = get_rknn();
    let native_nhwc_output_attr = rknn.query_with_input::<NativeNHWCOutputAttr>(0).unwrap();

    assert_eq!(native_nhwc_output_attr.dims(), &[1, 1000]);
    assert!(!native_nhwc_output_attr.name().is_empty());
    assert_eq!(
        native_nhwc_output_attr.dtype(),
        DataTypeKind::Int8(DataType::INT8)
    );
    assert_eq!(
        native_nhwc_output_attr.qnt_type(),
        QuantTypeKind::AffineAsymmetric(QuantType::QNT_AFFINE_ASYMMETRIC)
    );
}
