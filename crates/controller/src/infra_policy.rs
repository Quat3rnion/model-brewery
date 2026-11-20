use crate::{
    Error,
    domain::{Host, KINDS},
};
use brewery_worker::resources::Limits;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Default, Deserialize, Serialize)]
pub(super) struct Policy {
    pub revision: u64,
    pub paused: bool,
    pub disabled_jobs: BTreeSet<String>,
    #[serde(default)]
    pub resources: Limits,
    #[serde(default)]
    pub weight_editor: Option<bool>,
    #[serde(default)]
    pub remote_editor: Option<crate::remote_edit::Receipt>,
    #[serde(default)]
    pub model_directories: Vec<brewery_worker::source_grants::SourceGrant>,
}

impl Policy {
    pub(super) fn load(db: &Connection, worker: &str) -> Result<Self, Error> {
        db.query_row(
            "SELECT body FROM infra_host_policy WHERE worker=?",
            [worker],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .map(|body| serde_json::from_str(&body).map_err(Into::into))
        .transpose()
        .map(Option::unwrap_or_default)
    }

    pub(super) fn save(&self, tx: &Transaction<'_>, worker: &str) -> Result<(), Error> {
        tx.execute(
            "INSERT INTO infra_host_policy(worker,body) VALUES(?,?)
            ON CONFLICT(worker) DO UPDATE SET body=excluded.body",
            params![worker, serde_json::to_string(self)?],
        )?;
        Ok(())
    }

    pub(super) fn restrict(&self, host: &mut Host) -> Result<(), Error> {
        host.scheduling_paused |= self.paused;
        for kind in &self.disabled_jobs {
            if let Some(enabled @ true) = host.capabilities.get_mut(kind) {
                *enabled = false;
                host.reasons
                    .insert(kind.clone(), "Disabled by infrastructure job policy".into());
            }
        }
        if !self.resources.is_empty() {
            host.resource_bounds
                .unwrap_or_default()
                .validate(&self.resources)?;
            host.resources = self.resources;
        }
        Ok(())
    }
}

pub(super) fn job_kinds(host: &Host) -> Vec<&'static str> {
    KINDS
        .iter()
        .copied()
        .filter(|kind| host.capabilities.get(*kind) == Some(&true))
        .collect()
}

pub(super) fn validate_jobs(host: &Host, jobs: &[String]) -> Option<BTreeSet<String>> {
    let installed = job_kinds(host);
    let mut disabled = BTreeSet::new();
    for kind in jobs {
        if !installed.contains(&kind.as_str()) || !disabled.insert(kind.clone()) {
            return None;
        }
    }
    Some(disabled)
}
