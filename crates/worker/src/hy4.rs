use serde_json::{Value, json};

pub const OPTION: &str = "hy4_overrides";
pub const CAPABILITY: &str = "hy4_plan";

pub fn valid(kind: &str, options: &Value) -> bool {
    kind == "plan" && *options == json!({(OPTION): {"blk.1.ffn_down_exps.weight": "Q8_0"}})
}
