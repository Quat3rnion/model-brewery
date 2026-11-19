use crate::App;
use axum::response::sse::Event;
use futures_util::{Stream, stream};
use rusqlite::{Connection, params};
use serde::Serialize;
use std::{io, sync::Arc, time::Duration};
use tokio::{sync::OwnedSemaphorePermit, time::sleep};

pub const SUBSCRIBERS: usize = 16;
pub const HEARTBEAT: Duration = Duration::from_secs(15);
const POLL: Duration = Duration::from_millis(250);
const RETAIN: i64 = 1024;
const BATCH: i64 = 128;

pub fn install(conn: &Connection) -> rusqlite::Result<()> {
    let tx = conn.unchecked_transaction()?;
    // This is an invalidation journal, separate from full immutable job logs.
    // Only IDs, change kinds, and current job states are persisted here.
    tx.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS state_feed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            state TEXT
        );
        CREATE TRIGGER IF NOT EXISTS state_feed_retain AFTER INSERT ON state_feed BEGIN
            DELETE FROM state_feed WHERE id <= NEW.id - {RETAIN};
        END;
        CREATE TRIGGER IF NOT EXISTS state_feed_model_insert AFTER INSERT ON models BEGIN
            INSERT INTO state_feed(kind, entity_id) VALUES ('model.created', NEW.id);
        END;
        CREATE TRIGGER IF NOT EXISTS state_feed_model_delete AFTER DELETE ON models BEGIN
            INSERT INTO state_feed(kind, entity_id) VALUES ('model.deleted', OLD.id);
        END;
        CREATE TRIGGER IF NOT EXISTS state_feed_prompt_insert AFTER INSERT ON prompts BEGIN
            INSERT INTO state_feed(kind, entity_id) VALUES ('prompt.created', NEW.id);
        END;
        CREATE TRIGGER IF NOT EXISTS state_feed_job_insert AFTER INSERT ON jobs BEGIN
            INSERT INTO state_feed(kind, entity_id, state) VALUES ('job.created', NEW.id, NEW.state);
        END;
        CREATE TRIGGER IF NOT EXISTS state_feed_job_update AFTER UPDATE ON jobs
        WHEN OLD.state IS NOT NEW.state OR OLD.body IS NOT NEW.body OR OLD.result IS NOT NEW.result
        BEGIN
            INSERT INTO state_feed(kind, entity_id, state) VALUES ('job.updated', NEW.id, NEW.state);
        END;"
    ))?;
    tx.commit()
}

pub fn cursor(conn: &Connection) -> rusqlite::Result<i64> {
    conn.query_row("SELECT COALESCE(MAX(id), 0) FROM state_feed", [], |row| {
        row.get(0)
    })
}

#[derive(Serialize)]
pub struct Change {
    id: i64,
    kind: String,
    entity_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<String>,
}

pub enum Batch {
    Changes(Vec<Change>),
    Resync {
        reason: &'static str,
        event_cursor: i64,
    },
}

pub fn batch(conn: &Connection, after: i64) -> rusqlite::Result<Batch> {
    // One SQLite read snapshot for the retention boundary and corresponding rows.
    let tx = conn.unchecked_transaction()?;
    let (oldest, tip): (i64, i64) = tx.query_row(
        "SELECT COALESCE((SELECT MIN(id) FROM state_feed), 1),
                COALESCE((SELECT MAX(id) FROM state_feed), 0)",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if after > tip {
        return Ok(Batch::Resync {
            reason: "ahead",
            event_cursor: tip,
        });
    }
    if after < oldest - 1 {
        return Ok(Batch::Resync {
            reason: "stale",
            event_cursor: tip,
        });
    }
    let mut stmt = tx.prepare(
        "SELECT id, kind, entity_id, state FROM state_feed WHERE id > ? ORDER BY id LIMIT ?",
    )?;
    let mut previous = after;
    let mut changes = Vec::new();
    let mut rows = stmt.query(params![after, BATCH])?;
    while let Some(row) = rows.next()? {
        let id: i64 = row.get(0)?;
        if Some(id) != previous.checked_add(1) {
            return Ok(Batch::Resync {
                reason: "stale",
                event_cursor: tip,
            });
        }
        previous = id;
        changes.push(Change {
            id,
            kind: row.get(1)?,
            entity_id: row.get(2)?,
            state: row.get(3)?,
        });
    }
    Ok(Batch::Changes(changes))
}

struct Subscription {
    app: Arc<App>,
    _permit: OwnedSemaphorePermit,
    after: i64,
    pending: std::vec::IntoIter<Change>,
    closed: bool,
}

pub fn subscribe(
    app: Arc<App>,
    permit: OwnedSemaphorePermit,
    after: i64,
) -> impl Stream<Item = io::Result<Event>> + Send + 'static {
    let state = Subscription {
        app,
        _permit: permit,
        after,
        pending: Vec::new().into_iter(),
        closed: false,
    };
    stream::unfold(state, |mut state| async move {
        if state.closed {
            return None;
        }
        let result = loop {
            if let Some(change) = state.pending.next() {
                state.after = change.id;
                break json_event(
                    Event::default().event("change").id(change.id.to_string()),
                    change,
                );
            }
            let batch = match state.app.store.lock() {
                Ok(store) => store.feed_batch(state.after).map_err(io::Error::other),
                Err(_) => Err(io::Error::other("controller store lock poisoned")),
            };
            match batch {
                Ok(Batch::Changes(changes)) if changes.is_empty() => sleep(POLL).await,
                Ok(Batch::Changes(changes)) => state.pending = changes.into_iter(),
                Ok(Batch::Resync {
                    reason,
                    event_cursor,
                }) => {
                    state.closed = true;
                    break json_event(
                        Event::default().event("resync"),
                        serde_json::json!({
                            "reason": reason, "event_cursor": event_cursor,
                        }),
                    );
                }
                Err(error) => {
                    state.closed = true;
                    eprintln!("controller feed read error: {error}");
                    break Err(io::Error::other("state feed unavailable"));
                }
            }
        };
        Some((result, state))
    })
}

fn json_event(event: Event, value: impl Serialize) -> io::Result<Event> {
    event.json_data(value).map_err(io::Error::other)
}
