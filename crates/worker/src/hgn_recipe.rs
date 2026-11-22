use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub const CAPABILITY: &str = "hgn_layer_recipe";
pub const FORMAT: &str = "hgn-layer-recipe";
pub const ARCHIVE: &str = "hgn-archive";
pub const OUTPUT: &str = "converted.hgn-bundle";
pub const HELPER: &str = "BREWERY_LAYER_HGN_HELPER";
pub const HELPER_SHA: &str = "BREWERY_LAYER_HGN_HELPER_SHA256";

pub fn policy(value: &Value) -> Result<(), String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Recipe {
        schema: String,
        source_sha256: String,
        default_type: String,
        layers: BTreeMap<String, String>,
    }
    let recipe: Recipe =
        serde_json::from_value(value.clone()).map_err(|_| "invalid native HGN layer recipe")?;
    let baseline = |ty: &str| matches!(ty, "F32" | "F16" | "BF16" | "Q8G64");
    let trellis = |ty: &str| matches!(ty, "HT3" | "HT4");
    if recipe.schema != "brewery.layer-quant.v1"
        || !crate::is_digest(&recipe.source_sha256)
        || !baseline(&recipe.default_type)
        || recipe.layers.len() > 256
        || !recipe.layers.values().any(|ty| trellis(ty))
        || recipe.layers.iter().any(|(id, ty)| {
            !id.parse::<u32>()
                .is_ok_and(|n| n < 256 && n.to_string() == *id)
                || !(baseline(ty) || trellis(ty))
        })
    {
        return Err("managed native HGN recipes require an HT3 or HT4 layer override and a supported default type".into());
    }
    Ok(())
}

pub fn layout<'a>(verify: bool, inputs: impl IntoIterator<Item = (&'a str, &'a str)>) -> bool {
    let mut inputs = inputs.into_iter();
    let Some((source, host)) = inputs.next() else {
        return false;
    };
    matches!(source, "hf" | "safetensors")
        && (!verify || inputs.next() == Some((ARCHIVE, host)))
        && (source != "safetensors" || inputs.next() == Some(("assets", host)))
        && inputs.next().is_none()
}

pub fn options(kind: &str, options: &Value) -> bool {
    matches!(kind, "inspect" | "plan" | "convert" | "verify")
        && options["output_format"] == FORMAT
        && options
            .as_object()
            .is_some_and(|map| map.len() == if kind == "inspect" { 2 } else { 3 })
        && policy(&options["layer_recipe"]).is_ok()
        && (kind == "inspect"
            || options["layer_plan_sha256"]
                .as_str()
                .is_some_and(crate::is_digest))
}
