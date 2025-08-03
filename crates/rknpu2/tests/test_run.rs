use rknpu2::{
    RKNN,
    tensor::{TensorT, builder::TensorBuilder, tensor::Tensor},
};

static MODEL_DATA: &[u8] = include_bytes!("./fixtures/mobilenet_v2.rknn");

#[cfg(not(feature = "libloading"))]
use rknpu2::api::linked::LinkedAPI;

#[cfg(not(feature = "libloading"))]
fn get_rknn(flag: u32) -> RKNN<LinkedAPI> {
    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new(&mut model_data, flag).unwrap();
    rknn
}

#[cfg(feature = "libloading")]
use rknpu2::api::runtime::RuntimeAPI;

#[cfg(feature = "libloading")]
fn get_rknn(flag: u32) -> RKNN<RuntimeAPI> {
    use rknpu2::utils;

    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new_with_library(
        utils::find_rknn_library()
            .next()
            .expect("No RKNN library found. Please install librknnrt.so."),
        &mut model_data,
        flag,
    )
    .unwrap();
    rknn
}

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
#[test]
fn test_run() {
    let model = get_rknn(0);

    let mut input = TensorBuilder::new_input(&model, 0)
        .allocate::<i8>()
        .unwrap();
    input.fill_with(0i8);
    model.set_inputs(&[input]).unwrap();
    model.run().unwrap();
    let mut outputs = model.get_outputs().unwrap();
    let output = <TensorT as TryInto<Tensor<i8>>>::try_into(outputs.remove(0)).unwrap();
    let output = output.as_slice();
    assert_eq!(output.len(), 1000);
}

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
#[test]
fn test_perf_detail() {
    use {
        rknpu2::query::{PerfDetail, PerfRun},
        rknpu2_sys::RKNN_FLAG_COLLECT_PERF_MASK,
    };

    let model = get_rknn(RKNN_FLAG_COLLECT_PERF_MASK);

    let mut input = TensorBuilder::new_input(&model, 0)
        .allocate::<i8>()
        .unwrap();
    input.fill_with(0i8);
    model.set_inputs(&[input]).unwrap();
    model.run().unwrap();
    let mut outputs = model.get_outputs().unwrap();
    let output = <TensorT as TryInto<Tensor<i8>>>::try_into(outputs.remove(0)).unwrap();
    let output = output.as_slice();
    assert_eq!(output.len(), 1000);

    let perf_run = model.query::<PerfRun>().unwrap();
    assert!(perf_run.run_duration() > 0);

    let perf_detail = model.query::<PerfDetail>().unwrap();
    assert!(perf_detail.details().len() > 0);
}
