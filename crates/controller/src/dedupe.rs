use crate::{domain::now, idempotency};
use rusqlite::{Connection, OptionalExtension as _, Transaction, params};
use serde::Serialize;

// Terminal resources stop matching 10 seconds after acceptance. Active
// resources continue to match; cancellation keeps them active until terminal.
const WINDOW_MS: u64 = 10_000;

pub fn install(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS duplicate_submissions(
           scope TEXT NOT NULL, fingerprint BLOB NOT NULL, resource_id TEXT NOT NULL,
           accepted_at INTEGER NOT NULL, PRIMARY KEY(scope, fingerprint, resource_id)
         );
         CREATE INDEX IF NOT EXISTS duplicate_submissions_expiry ON duplicate_submissions(accepted_at);",
    )
}

pub struct Request {
    scope: &'static str,
    fingerprint: [u8; 32],
}

impl Request {
    pub fn new(scope: &'static str, request: &impl Serialize) -> serde_json::Result<Self> {
        Ok(Self {
            scope,
            fingerprint: idempotency::fingerprint(request)?,
        })
    }

    pub fn find(&self, tx: &Transaction<'_>) -> rusqlite::Result<Option<String>> {
        tx.query_row(
            "SELECT d.resource_id FROM duplicate_submissions d
             LEFT JOIN jobs j ON d.scope='jobs' AND d.resource_id=j.id
             LEFT JOIN conversions c ON d.scope='conversions' AND d.resource_id=c.id
             WHERE d.scope=? AND d.fingerprint=?
               AND (d.accepted_at > ? OR coalesce(j.state,c.state) IN ('queued','running','cancelling'))
             ORDER BY d.accepted_at, d.resource_id LIMIT 1",
            params![self.scope, &self.fingerprint[..], now().saturating_sub(WINDOW_MS)],
            |row| row.get(0),
        ).optional()
    }

    pub fn record(&self, tx: &Transaction<'_>, id: &str) -> rusqlite::Result<()> {
        let accepted_at = now();
        tx.execute(
            "DELETE FROM duplicate_submissions AS d WHERE accepted_at <= ?
             AND NOT EXISTS (SELECT 1 FROM jobs j WHERE d.scope='jobs' AND d.resource_id=j.id
                             AND j.state IN ('queued','running','cancelling'))
             AND NOT EXISTS (SELECT 1 FROM conversions c WHERE d.scope='conversions' AND d.resource_id=c.id
                             AND c.state IN ('queued','running','cancelling'))",
            [accepted_at.saturating_sub(WINDOW_MS)],
        )?;
        tx.execute(
            "INSERT INTO duplicate_submissions(scope,fingerprint,resource_id,accepted_at) VALUES (?,?,?,?)",
            params![self.scope, &self.fingerprint[..], id, accepted_at],
        )?;
        Ok(())
    }
}
