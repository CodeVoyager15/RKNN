use rknpu2::RKNN;

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
    use rknpu2::{
        io::{
            buffer::{BufMutView, BufView},
            input::Input,
            output::{Output, OutputKind},
        },
        tensor::{TensorFormat, TensorFormatKind},
    };

    let model = get_rknn(0);

    let input_buffer = vec![0i8; 1 * 224 * 224 * 3];

    let input = Input::new(
        0,
        BufView::I8(&input_buffer),
        false,
        TensorFormatKind::NHWC(TensorFormat::NHWC),
    );
    model.set_inputs(input).unwrap();
    model.run().unwrap();

    let mut logits = vec![0.0f32; 1000];

    let output = Output {
        index: 0,
        kind: OutputKind::Preallocated {
            buf: BufMutView::F32(&mut logits),
            want_float: true,
        },
    };

    model.get_outputs(&mut vec![output]).unwrap();

    assert_eq!(logits.len(), 1000);
}

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
#[test]
fn test_perf_detail() {
    use {
        rknpu2::{
            io::{
                buffer::{BufMutView, BufView},
                input::Input,
                output::{Output, OutputKind},
            },
            query::{PerfDetail, PerfRun},
            tensor::{TensorFormat, TensorFormatKind},
        },
        rknpu2_sys::RKNN_FLAG_COLLECT_PERF_MASK,
    };

    let model = get_rknn(RKNN_FLAG_COLLECT_PERF_MASK);

    let input_buffer = vec![0i8; 1 * 3 * 224 * 224];

    let input = Input::new(
        0,
        BufView::I8(&input_buffer),
        true,
        TensorFormatKind::NHWC(TensorFormat::NHWC),
    );
    model.set_inputs(input).unwrap();
    model.run().unwrap();

    let mut logits = vec![0.0f32; 1000];

    let output = Output {
        index: 0,
        kind: OutputKind::Preallocated {
            buf: BufMutView::F32(&mut logits),
            want_float: true,
        },
    };

    model.get_outputs(&mut vec![output]).unwrap();

    assert_eq!(logits.len(), 1000);

    let perf_run = model.query::<PerfRun>().unwrap();
    assert!(perf_run.run_duration() > 0);

    let perf_detail = model.query::<PerfDetail>().unwrap();
    assert!(perf_detail.details().len() > 0);
}
