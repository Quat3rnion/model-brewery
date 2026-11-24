use super::{Connection, Result, Store, encode, params, read_list, read_one};
use crate::registered_evaluation::{Record, Summary};

pub(super) fn install(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS evaluation_sets(name TEXT PRIMARY KEY, body TEXT NOT NULL);",
    )?;
    Ok(())
}

impl Store {
    pub(crate) fn insert_evaluation_set(&self, row: &Record) -> Result<bool> {
        Ok(self.conn.execute(
            "INSERT INTO evaluation_sets(name, body) VALUES(?1, ?2) ON CONFLICT(name) DO NOTHING",
            params![row.spec.name, encode(row)?],
        )? == 1)
    }

    pub(crate) fn evaluation_set(&self, name: &str) -> Result<Option<Record>> {
        read_one(
            &self.conn,
            "SELECT body FROM evaluation_sets WHERE name=?1",
            name,
        )
    }

    pub(crate) fn evaluation_sets(&self) -> Result<Vec<Summary>> {
        let rows: Vec<Record> = read_list(
            &self.conn,
            "SELECT body FROM evaluation_sets ORDER BY name",
            [],
        )?;
        Ok(rows.into_iter().map(Record::summary).collect())
    }
}
