use half::f16;

pub fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

pub fn softmax_f16(logits: &[f16]) -> Vec<f16> {
    let max = logits
        .iter()
        .copied()
        .map(f32::from)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits
        .iter()
        .copied()
        .map(f32::from)
        .map(|x| (x - max).exp())
        .collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter()
        .map(|x| x / sum)
        .map(f16::from_f32)
        .collect()
}
