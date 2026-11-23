use super::{Result, Store, read_list, read_one};
use crate::{
    findings::{Finding, Summary},
    idempotency::{self, Submission},
};
use rusqlite::{Connection, params};

pub fn install(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS findings(
            id TEXT PRIMARY KEY NOT NULL,
            body TEXT NOT NULL
         );
         UPDATE findings SET body=json_set(body, '$.state', 'failed',
             '$.error', 'The controller stopped before the replay completed.')
         WHERE json_extract(body, '$.state')='running';",
    )?;
    Ok(())
}

impl Store {
    pub fn findings(&self) -> Result<Vec<Summary>> {
        read_list(
            &self.conn,
            "SELECT json_remove(body, '$.experiments', '$.comparison', '$.provenance') \
                              FROM findings ORDER BY rowid DESC LIMIT 100",
            [],
        )
    }

    pub fn finding(&self, id: &str) -> Result<Option<Finding>> {
        read_one(&self.conn, "SELECT body FROM findings WHERE id=?", id)
    }

    pub fn add_finding(&mut self, row: &Finding, submission: &Submission<'_>) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO findings(id,body) VALUES (?1,?2)",
            params![row.summary.id, serde_json::to_string(row)?],
        )?;
        idempotency::record(&tx, Some(submission), &row.summary.id)?;
        tx.commit()?;
        Ok(())
    }

    pub fn finish_finding(&self, row: &Finding) -> Result<()> {
        self.conn.execute(
            "UPDATE findings SET body=?2 WHERE id=?1 AND json_extract(body, '$.state')='running'",
            params![row.summary.id, serde_json::to_string(row)?],
        )?;
        Ok(())
    }
}
