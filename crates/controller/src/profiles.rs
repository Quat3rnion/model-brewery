use crate::gguf::{Format, Selection};
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::LazyLock};

pub const CATALOG_JSON: &str = include_str!("profile_catalog.json");
static CATALOG: LazyLock<Result<Catalog, serde_json::Error>> =
    LazyLock::new(|| serde_json::from_str(CATALOG_JSON));

#[derive(Deserialize)]
struct Catalog {
    architecture: String,
    catalog_sha256: String,
    base_bytes: Bases,
    tensors: Vec<Tensor>,
}

#[derive(Deserialize)]
struct Bases {
    v2: u64,
    ht43: u64,
}

#[derive(Deserialize)]
struct Tensor {
    name: String,
    editable: bool,
    delta_bytes: i64,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Base {
    V2,
    Ht43,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NewHgnProfile {
    #[serde(default)]
    format: Format,
    name: String,
    pub base: Base,
    catalog_sha256: String,
    #[serde(deserialize_with = "unique_overrides")]
    overrides: BTreeMap<String, Base>,
}

#[derive(Deserialize, Serialize)]
struct Size {
    base_bytes: u64,
    predicted_bytes: u64,
    delta_bytes: i64,
}

#[derive(Deserialize, Serialize)]
pub struct HgnProfile {
    pub revision: String,
    pub name: String,
    pub base: Base,
    catalog_sha256: String,
    overrides: BTreeMap<String, Base>,
    size: Size,
    created_at: u64,
    // Save architecture with the content so a future embedded catalog update
    // cannot change a previously saved or queued recipe.
    architecture: String,
}

fn unique_overrides<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<BTreeMap<String, T>, D::Error> {
    struct Unique<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> de::Visitor<'de> for Unique<T> {
        type Value = BTreeMap<String, T>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a map of distinct tensor names to alternate profiles")
        }
        fn visit_map<M: de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
            let mut out = BTreeMap::new();
            while let Some((tensor, base)) = map.next_entry()? {
                if out.insert(tensor, base).is_some() {
                    return Err(de::Error::custom("duplicate tensor override"));
                }
            }
            Ok(out)
        }
    }
    d.deserialize_map(Unique(std::marker::PhantomData))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewGgufProfile {
    format: Format,
    name: String,
    preset: String,
    #[serde(default, deserialize_with = "unique_overrides")]
    overrides: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum NewProfile {
    Hgn(NewHgnProfile),
    Gguf(NewGgufProfile),
}

#[derive(Deserialize, Serialize)]
pub struct GgufProfile {
    format: Format,
    name: String,
    revision: String,
    created_at: u64,
    #[serde(flatten)]
    selection: Selection,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub enum Profile {
    Hgn(HgnProfile),
    Gguf(GgufProfile),
}

impl NewProfile {
    pub fn save(self) -> Result<Profile, String> {
        match self {
            Self::Hgn(hgn) => hgn.save().map(Profile::Hgn),
            Self::Gguf(gguf) => {
                if gguf.format != Format::Gguf {
                    return Err("expected format gguf".into());
                }
                check_name(&gguf.name)?;
                let selection = Selection::new(Some(&gguf.preset), gguf.overrides)?;
                let mut identity = json!({"schema":1,"format":"gguf","name":gguf.name,"preset":selection.preset,
                    "overrides":selection.overrides,"encoder_revision":selection.encoder_revision});
                if let Some(family) = selection.encoder_family {
                    identity["encoder_family"] = json!(family);
                }
                identity.sort_all_objects();
                let revision = format!(
                    "{:x}",
                    Sha256::digest(serde_json::to_vec(&identity).map_err(|_| "invalid recipe")?)
                );
                Ok(Profile::Gguf(GgufProfile {
                    format: gguf.format,
                    name: gguf.name,
                    revision,
                    created_at: crate::domain::now(),
                    selection,
                }))
            }
        }
    }
}

impl Profile {
    pub fn name(&self) -> &str {
        match self {
            Self::Hgn(row) => &row.name,
            Self::Gguf(row) => &row.name,
        }
    }
    pub fn revision(&self) -> &str {
        match self {
            Self::Hgn(row) => &row.revision,
            Self::Gguf(row) => &row.revision,
        }
    }
    pub fn format(&self) -> Format {
        match self {
            Self::Hgn(_) => Format::Hgn,
            Self::Gguf(_) => Format::Gguf,
        }
    }
    pub fn recipe(&self) -> Option<Value> {
        match self {
            Self::Hgn(row) => Some(row.recipe()),
            _ => None,
        }
    }
    pub fn gguf(&self) -> Option<Selection> {
        match self {
            Self::Gguf(row) => Some(row.selection.clone()),
            _ => None,
        }
    }
}

fn check_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
        Err("profile name must be 1–256 bytes without controls".into())
    } else {
        Ok(())
    }
}

impl NewHgnProfile {
    fn save(self) -> Result<HgnProfile, String> {
        if !self.format.is_hgn() {
            return Err("expected format hgn".into());
        }
        let catalog = CATALOG
            .as_ref()
            .map_err(|_| "profile catalog unavailable")?;
        if self.catalog_sha256 != catalog.catalog_sha256 {
            return Err("profile catalog digest does not match".into());
        }
        check_name(&self.name)?;
        if !(1..=4096).contains(&self.overrides.len()) {
            return Err("profile requires between 1 and 4096 tensor overrides".into());
        }
        let editable: BTreeMap<_, _> = catalog
            .tensors
            .iter()
            .filter(|t| t.editable)
            .map(|t| (t.name.as_str(), t.delta_bytes))
            .collect();
        let mut delta = 0_i128;
        for (name, choice) in &self.overrides {
            if name.len() > 512 || *choice == self.base {
                return Err(
                    "each override must select the alternate profile for an editable tensor".into(),
                );
            }
            let signed = i128::from(
                *editable
                    .get(name.as_str())
                    .ok_or("unknown or non-editable tensor override")?,
            );
            delta += if self.base == Base::V2 {
                signed
            } else {
                -signed
            };
        }
        let base = match self.base {
            Base::V2 => catalog.base_bytes.v2,
            Base::Ht43 => catalog.base_bytes.ht43,
        };
        let mut profile = HgnProfile {
            revision: String::new(),
            architecture: catalog.architecture.clone(),
            name: self.name,
            base: self.base,
            catalog_sha256: self.catalog_sha256,
            overrides: self.overrides,
            created_at: crate::domain::now(),
            size: Size {
                base_bytes: base,
                predicted_bytes: (i128::from(base) + delta)
                    .try_into()
                    .map_err(|_| "invalid predicted size")?,
                delta_bytes: delta
                    .try_into()
                    .map_err(|_| "invalid predicted size delta")?,
            },
        };
        let mut canonical = profile.body();
        canonical.sort_all_objects();
        profile.revision = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&canonical).map_err(|_| "invalid recipe")?)
        );
        Ok(profile)
    }
}

impl HgnProfile {
    fn body(&self) -> Value {
        json!({"schema":1, "name":self.name, "architecture":self.architecture,
            "base":self.base, "catalog_sha256":self.catalog_sha256, "overrides":self.overrides})
    }

    pub fn recipe(&self) -> Value {
        let mut body = self.body();
        body["revision"] = json!(self.revision);
        body
    }
}

pub fn valid_revision(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
