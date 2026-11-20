use serde_json::{Value, json};

pub const OPTION: &str = "hy4_overrides";
pub const CAPABILITY: &str = "hy4_plan";
pub const ENCODE: &str = "hy4_encode";
pub const FILE: &str = "hy4-gate-q8.gguf";
pub const RECEIPT: &str = "hy4-gate-q8.json";

pub fn capability(kind: &str, options: &Value) -> Option<&'static str> {
    match kind {
        "plan" if *options == json!({(OPTION): {"blk.1.ffn_down_exps.weight": "Q8_0"}}) => {
            Some(CAPABILITY)
        }
        "convert" if *options == json!({(OPTION): {"blk.1.attn_gate.weight": "Q8_0"}}) => {
            Some(ENCODE)
        }
        _ => None,
    }
}
