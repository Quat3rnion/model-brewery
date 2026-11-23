use super::{Result, Store, read_list, read_one};
use crate::{
    agent::{Provider, Session, Summary},
    idempotency::{self, Submission},
};
use rusqlite::{Connection, params};

pub fn install(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS agent_provider(id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS agent_sessions(id TEXT PRIMARY KEY NOT NULL, body TEXT NOT NULL);",
    )?;
    // The controller holds its data lock before migrations. Never re-run a turn
    // or re-submit a tool after the process restarts.
    conn.execute(
        "UPDATE agent_sessions SET body=json_set(body, '$.status', 'failed', '$.message', ?1)
         WHERE json_extract(body,'$.status')='running'",
        ["The controller stopped before the session completed."],
    )?;
    Ok(())
}

impl Store {
    pub fn agent_provider(&self) -> Result<Option<Provider>> {
        read_one(
            &self.conn,
            "SELECT body FROM agent_provider WHERE id=?",
            "1",
        )
    }

    pub fn put_agent_provider(&self, provider: &Provider) -> Result<()> {
        self.conn.execute(
            "INSERT INTO agent_provider(id,body) VALUES (1,?1)
             ON CONFLICT(id) DO UPDATE SET body=excluded.body",
            [serde_json::to_string(provider)?],
        )?;
        Ok(())
    }

    pub fn agent_sessions(&self) -> Result<Vec<Summary>> {
        read_list(
            &self.conn,
            "SELECT json_remove(body,'$.entries') FROM agent_sessions ORDER BY rowid DESC LIMIT 100",
            [],
        )
    }

    pub fn agent_session(&self, id: &str) -> Result<Option<Session>> {
        read_one(&self.conn, "SELECT body FROM agent_sessions WHERE id=?", id)
    }

    pub fn add_agent_session(
        &mut self,
        row: &Session,
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO agent_sessions(id,body) VALUES (?1,?2)",
            params![row.info.id, serde_json::to_string(row)?],
        )?;
        idempotency::record(&tx, submission, &row.info.id)?;
        tx.commit()?;
        Ok(())
    }

    pub fn put_agent_session(&self, row: &Session) -> Result<()> {
        self.conn.execute(
            "UPDATE agent_sessions SET body=?2 WHERE id=?1",
            params![row.info.id, serde_json::to_string(row)?],
        )?;
        Ok(())
    }

    pub fn stop_agent_session(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE agent_sessions SET body=json_set(body, '$.status', 'cancelled', '$.message', ?2)
             WHERE id=?1 AND json_extract(body,'$.status')='running'",
            params![id, "The operator cancelled the session."],
        )?;
        Ok(())
    }
}
