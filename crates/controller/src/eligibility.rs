use crate::{
    conversion::{Conversion, NewConversion},
    domain::{
        Host, Model, NewModel, Reason, SIMPLE_WORKFLOWS, manual_registration_blocker,
        standard_job_host_blocker,
    },
    gguf::Format,
    profiles::Profile,
};
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Serialize)]
pub struct Entry {
    eligible: bool,
    reasons: Vec<Reason>,
}

impl Entry {
    pub fn new(reasons: Vec<Reason>) -> Self {
        Self {
            eligible: reasons.is_empty(),
            reasons,
        }
    }

    fn any<'a>(entries: impl Iterator<Item = &'a Entry>) -> Self {
        let mut reasons = Vec::new();
        for entry in entries {
            if entry.eligible {
                return Self::new(Vec::new());
            }
            for reason in &entry.reasons {
                if !reasons.contains(reason) {
                    reasons.push(reason.clone());
                }
            }
        }
        if reasons.is_empty() {
            reasons.push(Reason::link(
                "Configure a worker in Settings > Hosts.",
                "#/settings/infrastructure",
            ));
        }
        Self::new(reasons)
    }
}

#[derive(Serialize)]
pub struct Snapshot<'a> {
    global: BTreeMap<&'static str, Entry>,
    hosts: BTreeMap<&'a str, HostWorkflows>,
}

#[derive(Serialize)]
struct HostWorkflows {
    #[serde(flatten)]
    entries: BTreeMap<&'static str, Entry>,
    gguf_output_by_kind: BTreeMap<&'static str, Entry>,
    gguf_rocmfpx_by_kind: BTreeMap<&'static str, Entry>,
}

const CONVERSION_WORKFLOWS: &[&str] = &[
    "managed_conversion",
    "component_conversion",
    "custom_profile_conversion",
    "pool_conversion",
    "pool_component_conversion",
    "pool_custom_profile_conversion",
    "gguf_output",
    "register_manual",
];

pub fn snapshot(hosts: &[Host]) -> Snapshot<'_> {
    let by_host: BTreeMap<_, _> = hosts
        .iter()
        .map(|host| {
            let mut entries: BTreeMap<_, _> = SIMPLE_WORKFLOWS
                .iter()
                .map(|&kind| {
                    (
                        kind,
                        Entry::new(standard_job_host_blocker(host, kind).into_iter().collect()),
                    )
                })
                .collect();
            entries.insert(
                "register_manual",
                Entry::new(
                    manual_registration_blocker(&host.id, "gguf")
                        .into_iter()
                        .collect(),
                ),
            );
            for (name, pooled, components, custom) in [
                ("managed_conversion", false, false, false),
                ("component_conversion", false, true, false),
                ("custom_profile_conversion", false, false, true),
                ("pool_conversion", true, false, false),
                ("pool_component_conversion", true, true, false),
                ("pool_custom_profile_conversion", true, false, true),
            ] {
                let profile = if custom {
                    Profile::eligibility_example()
                } else {
                    None
                };
                if custom && profile.is_none() {
                    entries.insert(
                        name,
                        Entry::new(vec![Reason::link(
                            "Check the custom profile catalog.",
                            "#/profiles",
                        )]),
                    );
                    continue;
                }
                entries.insert(
                    name,
                    conversion(
                        host,
                        hosts,
                        Format::Hgn,
                        "hf",
                        components,
                        pooled,
                        profile,
                        None,
                    ),
                );
            }
            let gguf = |preset| {
                ["gguf", "hf", "safetensors"]
                    .into_iter()
                    .map(|kind| {
                        (
                            kind,
                            conversion(
                                host,
                                hosts,
                                Format::Gguf,
                                kind,
                                kind == "safetensors",
                                false,
                                None,
                                preset,
                            ),
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            };
            let gguf_output_by_kind = gguf(None);
            let gguf_rocmfpx_by_kind = gguf(Some("Q8_0_ROCMFPX"));
            entries.insert("gguf_output", Entry::any(gguf_output_by_kind.values()));
            (
                host.id.as_str(),
                HostWorkflows {
                    entries,
                    gguf_output_by_kind,
                    gguf_rocmfpx_by_kind,
                },
            )
        })
        .collect();
    let global = SIMPLE_WORKFLOWS
        .iter()
        .chain(CONVERSION_WORKFLOWS)
        .map(|&kind| {
            (
                kind,
                Entry::any(by_host.values().map(|entry| &entry.entries[kind])),
            )
        })
        .collect();
    Snapshot {
        global,
        hosts: by_host,
    }
}

// Run the actual admission rules for representative, well-formed inputs.
// Eligibility is independent of saved model inventory and does not queue work.
// It does not replace validation of the model IDs or options on submission.
#[allow(clippy::too_many_arguments)]
fn conversion(
    host: &Host,
    hosts: &[Host],
    format: Format,
    kind: &str,
    components: bool,
    pooled: bool,
    profile: Option<&Profile>,
    preset: Option<&str>,
) -> Entry {
    let model = |kind: &str| -> Model {
        NewModel {
            name: Some(kind.into()),
            host: host.id.clone(),
            path: format!("/eligibility/{kind}"),
            kind: kind.into(),
        }
        .validate()
        .expect("eligibility example model")
    };
    let mut inputs = vec![model(kind)];
    let mut options = Map::new();
    if components {
        inputs.push(model("assets"));
        if format.is_hgn() {
            options.insert("assets".into(), Value::from(1));
        }
    }
    if let Some(preset) = preset {
        options.insert("preset".into(), preset.into());
    }
    let request = NewConversion {
        format,
        host: Some(if pooled {
            "auto".into()
        } else {
            host.id.clone()
        }),
        input_ids: inputs.iter().map(|input| input.id.clone()).collect(),
        options,
        decode_backend: None,
        profile_revision: profile.map(|p| p.revision().into()),
    };
    match Conversion::prepare(request, inputs, profile) {
        Ok((_, job, envelope, _)) => Entry::new(Conversion::host_blockers(
            host, &job, &envelope, None, hosts,
        )),
        Err(message) => Entry::new(vec![Reason::link(&message, "#/convert")]),
    }
}
