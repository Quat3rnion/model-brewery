use crate::{
    exl3_layer::half,
    gguf_conversion::inventory::{Model, Namespace, Tensor, align},
};
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub const STOCK: Namespace = Namespace::Stock;
pub const EMBEDDING: &str = "token_embd.weight";
pub const HEAD: &str = "output.weight";

pub fn region<'a>(
    file: &'a mut File,
    model: &Model,
    tensor: &Tensor,
) -> Result<io::Take<&'a mut File>> {
    file.seek(SeekFrom::Start(model.data + tensor.offset))?;
    Ok(file.take(tensor.bytes(tensor.ty, STOCK)?))
}

pub fn name(ty: u64) -> &'static str {
    STOCK.spec(ty).expect("known dtype").1
}

pub fn supported(ty: u64) -> bool {
    matches!(ty, 0 | 1 | 2 | 6 | 8 | 12 | 13 | 14 | 30)
}

fn u16_at(bytes: &[u8], n: usize) -> u16 {
    u16::from_le_bytes(bytes[n..n + 2].try_into().unwrap())
}

pub fn f32_at(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

fn bf16(bytes: &[u8], n: usize) -> f32 {
    f32::from_bits(u32::from(u16_at(bytes, n)) << 16)
}

// layouts match llama.cpp's ggml-quants.c; GGML dimension 0 is contiguous.
fn decode(ty: u64, b: &[u8], out: &mut [f32; 256]) {
    let h = |n| half(u16_at(b, n)) as f32;
    match ty {
        0 => out[0] = f32_at(b),
        1 => out[0] = h(0),
        30 => out[0] = bf16(b, 0),
        2 | 6 => {
            let d = h(0);
            let high = if ty == 6 {
                u32::from_le_bytes(b[2..6].try_into().unwrap())
            } else {
                0
            };
            let bias = if ty == 6 { 16 } else { 8 };
            let low = if ty == 6 { 6 } else { 2 };
            for (i, value) in out[..32].iter_mut().enumerate() {
                let code = (b[low + i % 16] >> (4 * (i / 16))) & 15;
                *value = d * (i32::from(code) + (((high >> i) & 1) as i32) * 16 - bias) as f32;
            }
        }
        8 => {
            let d = h(0);
            for (value, &code) in out.iter_mut().zip(&b[2..]) {
                *value = d * f32::from(code as i8);
            }
        }
        12 | 13 => {
            let (d, dm) = (h(0), h(2));
            let s = &b[4..16];
            let low = if ty == 13 { 48 } else { 16 };
            for (g, group) in out.chunks_exact_mut(32).enumerate() {
                let (scale, min) = if g < 4 {
                    (s[g] & 63, s[g + 4] & 63)
                } else {
                    (
                        (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4),
                        (s[g + 4] >> 4) | ((s[g] >> 6) << 4),
                    )
                };
                let (scale, min) = (d * f32::from(scale), dm * f32::from(min));
                for (i, value) in group.iter_mut().enumerate() {
                    let lo = (b[low + (g / 2) * 32 + i] >> (4 * (g % 2))) & 15;
                    let hi = if ty == 13 {
                        ((b[16 + i] >> g) & 1) * 16
                    } else {
                        0
                    };
                    *value = scale * f32::from(lo + hi) - min;
                }
            }
        }
        14 => {
            let d = h(208);
            for (g, group) in out.chunks_exact_mut(32).enumerate() {
                for (i, value) in group.iter_mut().enumerate() {
                    let lo = (b[g / 4 * 64 + g % 2 * 32 + i] >> (4 * ((g % 4) / 2))) & 15;
                    let hi = ((b[128 + g / 4 * 32 + i] >> (2 * (g % 4))) & 3) * 16;
                    let scale = b[192 + g * 2 + i / 16] as i8;
                    *value = d * f32::from(scale) * f32::from(lo as i8 + hi as i8 - 32);
                }
            }
        }
        _ => unreachable!("validated dtype"),
    }
}

/// Stream matching source blocks and never materialize a dequantized tensor.
pub fn delta(
    reference: impl Read,
    quantized: impl Read,
    bf_type: u64,
    quantized_type: u64,
    elements: u64,
    mut block: impl FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    let (_, _, n, qsize) = *STOCK.spec(quantized_type)?;
    let (_, _, _, fsize) = *STOCK.spec(bf_type)?;
    let (n, qsize, fsize) = (n as usize, qsize as usize, fsize as usize);
    let (mut a, mut b) = (
        BufReader::with_capacity(128 << 10, reference),
        BufReader::with_capacity(128 << 10, quantized),
    );
    let (mut f, mut q, mut values) = ([0u8; 1024], [0u8; 256], [0.; 256]);
    for _ in 0..elements / n as u64 {
        a.read_exact(&mut f[..n * fsize])?;
        b.read_exact(&mut q[..qsize])?;
        decode(quantized_type, &q[..qsize], &mut values);
        for i in 0..n {
            let reference = if bf_type == 30 {
                bf16(&f, i * 2)
            } else {
                f32_at(&f[i * 4..i * 4 + 4])
            };
            values[i] = reference - values[i];
        }
        block(&values[..n])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn extracts_k_quant_scales_mins_and_half_block_order() {
        let mut b = [0u8; 176];
        // d = .5, dmin = .25. Group 4 has scale 17 and min 33.
        b[..4].copy_from_slice(&[0, 0x38, 0, 0x34]);
        b[4] = 0x40;
        b[8] = 0x80;
        b[12] = 0x11;
        let mut out = [0.; 256];
        b[16] = 3;
        decode(12, &b[..144], &mut out);
        assert_eq!(out[0], 0.);
        b[80] = 3;
        decode(12, &b[..144], &mut out);
        assert_eq!(out[128], 17.25);
        assert_eq!(out[129], -8.25);
        b[16] = 0x10; // Q5_K high bit for group 4, lane 0
        b[112] = 3;
        decode(13, &b, &mut out);
        assert_eq!(out[128], 153.25);
        assert_eq!(out[129], -8.25);

        let mut b = [0u8; 210];
        b[208..].copy_from_slice(&[0, 0x38]);
        b[0] = 0x31;
        b[128] = 0x32;
        b[192] = 2;
        b[196] = (-2i8) as u8;
        decode(14, &b, &mut out);
        assert_eq!(out[0], 1.);
        assert_eq!(out[64], -19.);
    }

    #[test]
    fn extracts_q5_0_high_bits_from_both_halves() {
        let mut b = [0u8; 22];
        b[..2].copy_from_slice(&[0, 0x38]);
        b[2] = 1;
        b[4] = 2;
        b[6] = 0x21;
        b[7] = 0x43;
        let mut out = [0.; 256];
        decode(6, &b, &mut out);
        assert_eq!([out[0], out[1], out[16], out[17]], [0.5, -6.5, -7., 2.]);
    }
}

pub struct CopyTensor<'a> {
    pub tensor: &'a Tensor,
    pub model: &'a Model,
    pub file: usize,
    pub ty: u64,
}

/// Repacks an unchanged GGUF metadata block with a new tensor table and streamed payloads.
pub fn repack(
    path: &Path,
    header: &Model,
    header_file: usize,
    files: &mut [File; 2],
    tensors: &std::collections::BTreeMap<&str, CopyTensor<'_>>,
) -> Result<()> {
    let metadata = &mut files[header_file];
    metadata.seek(SeekFrom::Start(0))?;
    let mut out = BufWriter::with_capacity(
        1 << 20,
        File::options().create_new(true).write(true).open(path)?,
    );
    ensure!(
        io::copy(&mut metadata.take(8), &mut out)? == 8,
        "truncated GGUF header"
    );
    out.write_all(&(tensors.len() as u64).to_le_bytes())?;
    metadata.seek(SeekFrom::Start(16))?;
    ensure!(
        io::copy(&mut metadata.take(header.table - 16), &mut out)? == header.table - 16,
        "truncated GGUF metadata"
    );
    let mut offset = 0u64;
    for (&name, entry) in tensors {
        out.write_all(&(name.len() as u64).to_le_bytes())?;
        out.write_all(name.as_bytes())?;
        out.write_all(&(entry.tensor.shape.len() as u32).to_le_bytes())?;
        for &dim in &entry.tensor.shape {
            out.write_all(&dim.to_le_bytes())?;
        }
        out.write_all(&(entry.ty as u32).to_le_bytes())?;
        out.write_all(&offset.to_le_bytes())?;
        offset = align(
            offset
                .checked_add(entry.tensor.bytes(entry.ty, STOCK)?)
                .context("GGUF payload size overflow")?,
            header.align,
        )?;
    }
    let pad = |w: &mut BufWriter<File>, pos| -> Result<()> {
        w.write_all(&[0; 4096][..(align(pos, header.align)? - pos) as usize])?;
        Ok(())
    };
    let data = out.stream_position()?;
    pad(&mut out, data)?;
    for entry in tensors.values() {
        let mut bytes = region(&mut files[entry.file], entry.model, entry.tensor)?;
        let size = bytes.limit();
        if entry.ty == entry.tensor.ty {
            ensure!(
                io::copy(&mut bytes, &mut out)? == size,
                "truncated GGUF tensor"
            );
        } else {
            ensure!(entry.ty == 0 && entry.tensor.ty == 30, "invalid GGUF cast");
            let (mut src, mut dst) = ([0u8; 64 << 10], [0u8; 128 << 10]);
            while bytes.limit() != 0 {
                let n = bytes.limit().min(src.len() as u64) as usize;
                bytes.read_exact(&mut src[..n])?;
                for (i, word) in dst[..n * 2].chunks_exact_mut(4).enumerate() {
                    word.copy_from_slice(&bf16(&src, i * 2).to_le_bytes());
                }
                out.write_all(&dst[..n * 2])?;
            }
        }
        // The tensor payload starts aligned.
        pad(&mut out, entry.tensor.bytes(entry.ty, STOCK)?)?;
    }
    out.flush()?;
    Ok(())
}
