use {
    crate::markers::{Compatible, DataType, Layout, Normalization, QuantParams},
    image::DynamicImage,
};

/// Converts a `DynamicImage` to a tensor of type `D` with normalization `N` and layout `L`.
pub fn to_tensor<D: DataType<QuantParams = ()>, N: Normalization, L: Layout>(
    img: &DynamicImage,
) -> Vec<D::Repr>
where
    (): Compatible<N, D>,
{
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);

    let mut out = vec![D::Repr::default(); w * h * L::CHANNELS];

    // Single-pass: read pixel -> convert to f32 (0..1 or 0..255) -> normalize -> dtype -> store by L::index
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            // base floats
            let mut f = [p[0] as f32, p[1] as f32, p[2] as f32];
            N::apply(&mut f);

            // store
            for c in 0..3 {
                let idx = L::index(w, h, x, y, c);
                out[idx] = D::from_f32(f[c], &());
            }
        }
    }
    out
}

/// Converts a `DynamicImage` to a tensor of type `D` with normalization `N`, layout `L`, and quantization parameters `quant_params`.
pub fn to_tensor_with_quant<D: DataType<QuantParams = QuantParams>, N: Normalization, L: Layout>(
    img: &DynamicImage,
    quant_params: D::QuantParams,
) -> Vec<D::Repr>
where
    (): Compatible<N, D>,
{
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);

    let mut out = vec![D::Repr::default(); w * h * L::CHANNELS];

    // Single-pass: read pixel -> convert to f32 -> normalize -> dtype -> store by L::index
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            // base floats
            let mut f = [p[0] as f32, p[1] as f32, p[2] as f32];
            N::apply(&mut f);

            // store
            for c in 0..3 {
                let idx = L::index(w, h, x, y, c);
                out[idx] = D::from_f32(f[c], &quant_params);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::markers::*,
        image::{DynamicImage, RgbImage},
    };

    // Build a tiny RGB with distinctive channels:
    // base(x,y) = 10*x + y
    // R = base
    // G = 100 + base
    // B = 200 + base
    fn make_distinct_rgb(w: u32, h: u32) -> DynamicImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let base = (10 * x + y) as u8;
                img.put_pixel(x, y, image::Rgb([base, 100 + base, 200 + base]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn nhwc_flatten_2x2_nonorm_f32() {
        let img = make_distinct_rgb(2, 2);
        let v = to_tensor::<F32, NoNorm, NHWC>(&img);

        // NHWC order: (0,0),(1,0),(0,1),(1,1) with channels R,G,B
        let mut exp = Vec::<f32>::new();
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let base = (10 * x + y) as f32;
            exp.extend_from_slice(&[base, 100.0 + base, 200.0 + base]);
        }

        assert_eq!(v, exp, "NHWC flatten mismatch");
    }

    #[test]
    fn nchw_flatten_2x2_nonorm_f32() {
        let img = make_distinct_rgb(2, 2);
        let v = to_tensor::<F32, NoNorm, NCHW>(&img);

        // NCHW: channel planes
        let mut exp = Vec::<f32>::new();

        // R plane
        for (y, x) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            exp.push((10 * x + y) as f32);
        }
        // G plane
        for (y, x) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            exp.push(100.0 + (10 * x + y) as f32);
        }
        // B plane
        for (y, x) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            exp.push(200.0 + (10 * x + y) as f32);
        }

        assert_eq!(v, exp, "NCHW flatten mismatch");
    }

    #[test]
    fn imagenet_normalization_math_1x1() {
        let img = make_distinct_rgb(1, 1); // single pixel: [0, 100, 200]
        let v = to_tensor::<F32, ImageNet, NHWC>(&img);
        assert_eq!(v.len(), 3);

        // Expected: (val/255 - mean)/std   (your ImageNet::apply does the /255.0)
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let src = [0.0f32, 100.0, 200.0];
        let exp: Vec<f32> = src
            .iter()
            .zip(MEAN)
            .zip(STD)
            .map(|((v, m), s)| ((*v / 255.0) - m) / s)
            .collect();

        for i in 0..3 {
            let d = (v[i] - exp[i]).abs();
            assert!(
                d <= 1e-6,
                "imagenet norm diff at c{}: got {}, exp {}",
                i,
                v[i],
                exp[i]
            );
        }
    }
}
