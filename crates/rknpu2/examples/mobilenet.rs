use {
    image::ImageReader,
    itertools::Itertools,
    rknpu2::{
        RKNN,
        tensor::{TensorT, builder::TensorBuilder, tensor::Tensor},
    },
    std::{
        collections::BTreeMap,
        io::{BufRead, BufReader},
    },
};

static MODEL_DATA: &[u8] = include_bytes!("models/mobilenet_v2.rknn");

static MODEL_OUTPUTS: &str = include_str!("models/mobilenet-synset.txt");

const SCALE: f32 = 0.018658448;

const OUTPUT_SCALE: f32 = 0.14192297;

fn preprocess_image(img: &image::RgbImage) -> Vec<i8> {
    // ImageNet mean and std
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let mut result = Vec::with_capacity(224 * 224 * 3);
    for y in 0..224 {
        for x in 0..224 {
            let pixel = img.get_pixel(x, y);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let norm = (val - mean[c]) / std[c];
                let quant = (norm / SCALE).round();
                result.push(quant.clamp(-128.0, 127.0) as i8);
            }
        }
    }
    result
}

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
fn main() {
    let img_path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("Usage: {} <image_path>", std::env::args().nth(0).unwrap());
            std::process::exit(1);
        }
    };

    let img = ImageReader::open(&img_path)
        .unwrap()
        .decode()
        .unwrap()
        .resize_exact(224, 224, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let quantized_input = preprocess_image(&img);

    let model = get_rknn(0);

    let mut input = TensorBuilder::new_input(&model, 0)
        .allocate::<i8>()
        .unwrap();
    input.copy_data(&quantized_input).unwrap();

    model.set_inputs(&[input]).unwrap();
    model.run().unwrap();

    let mut outputs = model.get_outputs().unwrap();
    let output = <TensorT as TryInto<Tensor<i8>>>::try_into(outputs.remove(0)).unwrap();
    let output = output.as_slice();
    let logits = dequantize_output(output, OUTPUT_SCALE);
    let softmax_output = softmax(&logits);
    let top5 = softmax_output
        .into_iter()
        .enumerate()
        .sorted_by(|&(_, a), &(_, b)| b.partial_cmp(&a).unwrap())
        .take(5)
        .collect::<Vec<_>>();

    let class_reader = BufReader::with_capacity(512, MODEL_OUTPUTS.as_bytes());
    let classes: BTreeMap<usize, String> = class_reader
        .lines()
        .map(|line| line.unwrap())
        .enumerate()
        .collect();

    for (idx, prob) in top5 {
        println!("Probability: {}", prob);

        if let Some(class_name) = classes.get(&idx) {
            println!("{}: {}", class_name, idx);
        }
    }
}

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
    use rknpu2::utils::find_rknn_library;

    let mut model_data = MODEL_DATA.to_vec();
    let rknn = RKNN::new_with_library(
        find_rknn_library()
            .next()
            .expect("No RKNN library found. Please install librknnrt.so."),
        &mut model_data,
        flag,
    )
    .unwrap();
    rknn
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

fn dequantize_output(output: &[i8], scale: f32) -> Vec<f32> {
    output.iter().map(|&x| x as f32 * scale).collect()
}

#[cfg(not(any(feature = "rk3576", feature = "rk35xx")))]
fn main() {
    println!("mobilenet example requires rk3576 or rk35xx feature");
}
