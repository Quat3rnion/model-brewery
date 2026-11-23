use super::native_tensor_name;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Deserializer, de};
use std::{collections::BTreeMap, io::Read};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

pub struct Header {
    pub payload_start: u64,
    pub entries: BTreeMap<String, Tensor>,
}

impl Header {
    pub fn read(mut reader: impl Read, file_size: u64, limit: u64) -> Result<Self> {
        let mut size = [0u8; 8];
        reader.read_exact(&mut size)?;
        let size = u64::from_le_bytes(size);
        ensure!(
            (2..=limit.min(file_size.saturating_sub(8))).contains(&size),
            "SafeTensors header exceeds bounded extent"
        );
        let mut bytes = vec![0; usize::try_from(size)?];
        reader.read_exact(&mut bytes)?;
        ensure!(
            bytes.first() == Some(&b'{'),
            "invalid SafeTensors header start"
        );
        let mut json = serde_json::Deserializer::from_slice(&bytes);
        let entries = json.deserialize_map(Entries)?;
        json.end()?;
        ensure!(!entries.is_empty(), "missing SafeTensors tensors");
        let payload_start = 8 + size;
        let capacity = file_size - payload_start;
        let mut ranges = Vec::with_capacity(entries.len());
        for tensor in entries.values() {
            let width = match tensor.dtype.as_str() {
                "BOOL" | "I8" | "U8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
                "I16" | "U16" | "F16" | "BF16" => 2,
                "I32" | "U32" | "F32" => 4,
                "I64" | "U64" | "F64" | "C64" => 8,
                _ => bail!("unsupported SafeTensors dtype"),
            };
            let [start, end] = tensor.data_offsets;
            ensure!(
                tensor.shape.len() <= 8 && start <= end && end <= capacity,
                "invalid SafeTensors shape or offset"
            );
            let bytes = if tensor.shape.contains(&0) {
                0
            } else {
                tensor
                    .shape
                    .iter()
                    .try_fold(width, |n: u64, axis| n.checked_mul(*axis))
                    .context("SafeTensors shape overflow")?
            };
            ensure!(end - start == bytes, "SafeTensors tensor extent mismatch");
            ranges.push((start, end));
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (start, next) in ranges {
            ensure!(start == end, "SafeTensors payload has a gap or overlap");
            end = next;
        }
        ensure!(end == capacity, "unindexed SafeTensors payload bytes");
        Ok(Self {
            payload_start,
            entries,
        })
    }
}

// Reject repeated tensor names and repeated fields in a tensor record.
struct Entries;
impl<'de> de::Visitor<'de> for Entries {
    type Value = BTreeMap<String, Tensor>;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a SafeTensors header object")
    }
    fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entries = BTreeMap::new();
        let mut metadata = false;
        while let Some(name) = map.next_key::<String>()? {
            if name == "__metadata__" && !metadata {
                map.next_value::<BTreeMap<String, String>>()?;
                metadata = true;
            } else if native_tensor_name(&name) && !entries.contains_key(&name) {
                entries.insert(name, map.next_value::<Tensor>()?);
                if entries.len() > 100_000 {
                    return Err(de::Error::custom("too many SafeTensors tensors"));
                }
            } else {
                return Err(de::Error::custom(
                    "invalid or duplicate SafeTensors tensor name",
                ));
            }
        }
        Ok(entries)
    }
}
