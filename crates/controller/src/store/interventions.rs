use super::{Result, Store, encode, read_list, read_one};
use crate::{
    idempotency::{self, Submission},
    interventions::{Artifact, Summary},
};
use rusqlite::{Connection, OptionalExtension as _, params};

pub(super) fn install(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS intervention_versions(
            id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL,
            PRIMARY KEY(id, version)
         );",
    )
}

impl Store {
    pub fn interventions(&self) -> Result<Vec<Summary>> {
        read_list(
            &self.conn,
            "SELECT json_object(
               'id', v.id, 'name', json_extract(v.body, '$.name'),
               'model_id', json_extract(v.body, '$.model_id'),
               'model_sha256', json_extract(v.body, '$.model_sha256'),
               'latest_version', (SELECT max(version) FROM intervention_versions WHERE id=v.id)
             ) FROM intervention_versions v WHERE v.version=1 ORDER BY v.rowid",
            [],
        )
    }

    pub fn latest_intervention(&self, id: &str) -> Result<Option<Artifact>> {
        read_one(
            &self.conn,
            "SELECT body FROM intervention_versions WHERE id=? ORDER BY version DESC LIMIT 1",
            id,
        )
    }

    pub fn intervention(&self, id: &str, version: u32) -> Result<Option<Artifact>> {
        let text: Option<String> = self
            .conn
            .query_row(
                "SELECT body FROM intervention_versions WHERE id=? AND version=?",
                params![id, version],
                |row| row.get(0),
            )
            .optional()?;
        Ok(text.map(|text| serde_json::from_str(&text)).transpose()?)
    }

    pub fn insert_intervention(
        &mut self,
        version: &Artifact,
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO intervention_versions(id, version, body) VALUES(?, ?, ?)",
            params![version.binding.id, version.version, encode(version)?],
        )?;
        idempotency::record(
            &tx,
            submission,
            &format!("{}/{}", version.binding.id, version.version),
        )?;
        tx.commit()?;
        Ok(())
    }
}
