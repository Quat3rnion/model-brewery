use super::{Result, Store, encode, read_one};
use rusqlite::{Connection, params};
use serde_json::Value;

pub(super) fn install(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS crosscoder_artifacts(
            id TEXT PRIMARY KEY NOT NULL, body TEXT NOT NULL, weights BLOB NOT NULL
        );",
    )
}

impl Store {
    pub fn crosscoder(&self, id: &str) -> Result<Option<Value>> {
        read_one(
            &self.conn,
            "SELECT body FROM crosscoder_artifacts WHERE id=?",
            id,
        )
    }

    pub fn save_crosscoder(&self, id: &str, report: &Value, weights: &[u8]) -> Result<()> {
        self.conn.execute(
            "INSERT INTO crosscoder_artifacts(id, body, weights) VALUES(?, ?, ?)",
            params![id, encode(report)?, weights],
        )?;
        Ok(())
    }

    pub fn crosscoder_weights(&self, id: &str) -> Result<Vec<u8>> {
        Ok(self.conn.query_row(
            "SELECT weights FROM crosscoder_artifacts WHERE id=?",
            [id],
            |row| row.get(0),
        )?)
    }
}
