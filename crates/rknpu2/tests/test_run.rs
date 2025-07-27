use rknpu2::{RKNN, tensor::builder::TensorBuilder};

static MODEL_DATA: &[u8] = include_bytes!("./fixtures/mobilenet_v2.rknn");

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
#[test]
fn test_run() {
    let mut model_data = MODEL_DATA.to_vec();
    let model = RKNN::new(&mut model_data, 0).unwrap();

    let mut input = TensorBuilder::new_input(&model, 0)
        .allocate::<i8>()
        .unwrap();
    input.fill_with(0i8);
    model.set_inputs(&[input]).unwrap();
    model.run().unwrap();
    let outputs = model.get_outputs::<i8>().unwrap();
    let output = outputs[0].as_slice();
    assert_eq!(output.len(), 1000);
}

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
#[test]
fn test_perf_detail() {
    use {
        rknpu2::query::{PerfDetail, PerfRun},
        rknpu2_sys::RKNN_FLAG_COLLECT_PERF_MASK,
    };
    let mut model_data = MODEL_DATA.to_vec();
    let model = RKNN::new(&mut model_data, RKNN_FLAG_COLLECT_PERF_MASK).unwrap();

    let mut input = TensorBuilder::new_input(&model, 0)
        .allocate::<i8>()
        .unwrap();
    input.fill_with(0i8);
    model.set_inputs(&[input]).unwrap();
    model.run().unwrap();
    let outputs = model.get_outputs::<i8>().unwrap();
    let output = outputs[0].as_slice();
    assert_eq!(output.len(), 1000);

    let perf_run = model.query::<PerfRun>().unwrap();
    assert!(perf_run.run_duration() > 0);

    let perf_detail = model.query::<PerfDetail>().unwrap();
    assert!(perf_detail.details().len() > 0);
}
