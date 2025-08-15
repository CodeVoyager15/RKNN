pub trait Normalization {
    /// Normalize a single RGB triplet in-place
    fn apply(rgb: &mut [f32; 3]);
}

pub struct NoNorm;
impl Normalization for NoNorm {
    #[inline]
    fn apply(_rgb: &mut [f32; 3]) {}
}

pub struct ImageNet;
impl Normalization for ImageNet {
    #[inline]
    fn apply(rgb: &mut [f32; 3]) {
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        for c in 0..3 {
            rgb[c] = (rgb[c] / 255.0 - MEAN[c]) / STD[c];
        }
    }
}

// --- Data type markers ---
pub trait DataType: Copy {
    type Repr: Copy + Default;

    type QuantParams;

    fn from_f32(x: f32, q: &Self::QuantParams) -> Self::Repr;
}

#[derive(Copy, Clone, Debug)]
pub struct F32;

impl DataType for F32 {
    type Repr = f32;

    type QuantParams = ();

    #[inline]
    fn from_f32(x: f32, _q: &Self::QuantParams) -> Self::Repr {
        x
    }
}

#[derive(Copy, Clone, Debug)]
pub struct F16; // uses `half::f16` unified via workspace deps
impl DataType for F16 {
    type Repr = half::f16;

    type QuantParams = ();

    #[inline]
    fn from_f32(x: f32, _q: &Self::QuantParams) -> Self::Repr {
        half::f16::from_f32(x)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BF16;

impl DataType for BF16 {
    type Repr = half::bf16;

    type QuantParams = ();

    #[inline]
    fn from_f32(x: f32, _q: &Self::QuantParams) -> Self::Repr {
        half::bf16::from_f32(x)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct I8;

impl DataType for I8 {
    type Repr = i8;

    type QuantParams = QuantParams;

    #[inline]
    fn from_f32(x: f32, q: &Self::QuantParams) -> Self::Repr {
        let v = (x / q.scale).round() as i32 + q.zero_point;
        v.clamp(i8::MIN as i32, i8::MAX as i32) as i8
    }
}

#[derive(Copy, Clone, Debug)]
pub struct U8;

impl DataType for U8 {
    type Repr = u8;

    type QuantParams = ();

    #[inline]
    fn from_f32(x: f32, _q: &Self::QuantParams) -> Self::Repr {
        (x.round().clamp(0.0, 255.0)) as u8
    }
}

pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
}

// --- Layout markers ---
pub trait Layout {
    /// Given (x,y,c) and width/height, return flat index for the *float staging buffer*
    fn index(w: usize, h: usize, x: usize, y: usize, c: usize) -> usize;
    const CHANNELS: usize = 3;
}

pub struct NHWC;
impl Layout for NHWC {
    #[inline]
    fn index(w: usize, _h: usize, x: usize, y: usize, c: usize) -> usize {
        (y * w + x) * 3 + c
    }
}

pub struct NCHW;
impl Layout for NCHW {
    #[inline]
    fn index(w: usize, h: usize, x: usize, y: usize, c: usize) -> usize {
        c * (w * h) + y * w + x
    }
}

pub trait Compatible<N, D> {}
impl Compatible<NoNorm, U8> for () {}
impl Compatible<NoNorm, F32> for () {}
impl Compatible<ImageNet, F32> for () {}
impl Compatible<ImageNet, F16> for () {}
impl Compatible<ImageNet, BF16> for () {}
impl Compatible<ImageNet, I8> for () {}
