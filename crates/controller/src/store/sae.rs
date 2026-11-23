use super::{Result, Store, encode, read_list, read_one};
use crate::{
    idempotency::{self, Submission},
    sae::Artifact,
};
use rusqlite::{Connection, OptionalExtension as _, blob::Blob, params};

pub(super) fn install(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sae_versions(
           id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL, weights BLOB NOT NULL,
           PRIMARY KEY(id, version)
         );",
    )
}

impl Store {
    pub fn saes(&self) -> Result<Vec<Artifact>> {
        read_list(
            &self.conn,
            "SELECT body FROM sae_versions AS v WHERE version = (
               SELECT max(version) FROM sae_versions WHERE id = v.id
             ) ORDER BY v.rowid",
            [],
        )
    }

    pub fn latest_sae(&self, id: &str) -> Result<Option<Artifact>> {
        read_one(
            &self.conn,
            "SELECT body FROM sae_versions WHERE id=? ORDER BY version DESC LIMIT 1",
            id,
        )
    }

    pub fn sae(&self, id: &str, version: u32) -> Result<Option<Artifact>> {
        let text: Option<String> = self
            .conn
            .query_row(
                "SELECT body FROM sae_versions WHERE id=? AND version=?",
                params![id, version],
                |row| row.get(0),
            )
            .optional()?;
        Ok(text.map(|text| serde_json::from_str(&text)).transpose()?)
    }

    pub fn sae_blob(&self, id: &str, version: u32) -> Result<Blob<'_>> {
        let rowid = self.conn.query_row(
            "SELECT rowid FROM sae_versions WHERE id=? AND version=?",
            params![id, version],
            |row| row.get(0),
        )?;
        Ok(self
            .conn
            .blob_open("main", "sae_versions", "weights", rowid, true)?)
    }

    pub fn insert_sae(
        &mut self,
        a: &Artifact,
        weights: &[u8],
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO sae_versions(id, version, body, weights) VALUES(?, ?, ?, ?)",
            params![a.id, a.version, encode(a)?, weights],
        )?;
        idempotency::record(&tx, submission, &format!("{}/{}", a.id, a.version))?;
        tx.commit()?;
        Ok(())
    }
}
