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
                let base = 10 * x + y;
                img.put_pixel(x, y, image::Rgb([base as u8, (100 + base) as u8, (200 + base) as u8]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_to_tensor() {
        let img = make_distinct_rgb(2, 2);
        let tensor: Vec<u8> = to_tensor::<U8, Identity, CHW>(&img);
        assert_eq!(tensor.len(), 2 * 2 * 3);
        // Add more test logic here as needed
    }

    #[test]
    fn test_to_tensor_with_quant() {
        let img = make_distinct_rgb(2, 2);
        let quant_params = QuantParams { scale: 0.5, zero_point: 128 };
        let tensor: Vec<u8> = to_tensor_with_quant::<U8, Identity, CHW>(&img, quant_params);
        assert_eq!(tensor.len(), 2 * 2 * 3);
        // Add more test logic here as needed
    }
}