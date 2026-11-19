use crate::conversion::{self, Conversion};
use crate::domain::{Envelope, Event, Job, Model, Prompt, State, now};
use crate::idempotency::{self, Replay, Submission};
use rusqlite::{Connection, OptionalExtension as _, Transaction, params};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    fs::{File, OpenOptions},
    os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _, PermissionsExt as _},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct Store {
    conn: Connection,
    _lock: File,
}

impl Store {
    pub fn open(directory: &Path) -> Result<Self> {
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(directory)?;
        if directory.metadata()?.permissions().mode() & 0o077 != 0 {
            return Err("BREWERY_DATA must be private (chmod 700)".into());
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(directory.join("controller.lock"))?;
        fs2::FileExt::try_lock_exclusive(&lock)
            .map_err(|e| format!("cannot lock the controller data directory: {e}"))?;
        let conn = Connection::open(directory.join("controller.db"))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, body TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS prompts(id TEXT PRIMARY KEY, body TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS jobs(
               seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
               body TEXT NOT NULL, envelope TEXT NOT NULL, result TEXT
             );
             CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, seq);
             DROP INDEX IF EXISTS one_active_job;
             CREATE UNIQUE INDEX IF NOT EXISTS one_job_per_active_host
               ON jobs(json_extract(body, '$.host')) WHERE state = 'running';
             CREATE TABLE IF NOT EXISTS events(
               seq INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), body TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS job_events ON events(job_id, seq);",
        )?;
        crate::feed::install(&conn)?;
        idempotency::install(&conn)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS profile_revisions(revision TEXT PRIMARY KEY, body TEXT NOT NULL);",
        )?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversions(
                id TEXT PRIMARY KEY, state TEXT NOT NULL,
                current_job TEXT NOT NULL REFERENCES jobs(id), body TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS conversions_active ON conversions(state);",
        )?;
        Ok(Self { conn, _lock: lock })
    }

    pub fn feed_cursor(&self) -> rusqlite::Result<i64> {
        crate::feed::cursor(&self.conn)
    }

    pub fn feed_batch(&self, after: i64) -> rusqlite::Result<crate::feed::Batch> {
        crate::feed::batch(&self.conn, after)
    }

    // The caller holds the same Store lock from lookup through commit, and
    // open() excludes a second controller process from writing this database.
    pub fn replay(&self, submission: &Submission<'_>) -> Result<Replay> {
        Ok(submission.lookup(&self.conn)?)
    }

    pub fn add_model(&mut self, model: &Model, submission: Option<&Submission<'_>>) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO models(id, body) VALUES (?, ?)",
            params![model.id, encode(model)?],
        )?;
        idempotency::record(&tx, submission, &model.id)?;
        tx.commit()?;
        Ok(())
    }
    pub fn model(&self, id: &str) -> Result<Option<Model>> {
        read_one(&self.conn, "SELECT body FROM models WHERE id = ?", id)
    }
    pub fn models(&self) -> Result<Vec<Model>> {
        read_list(&self.conn, "SELECT body FROM models ORDER BY rowid", [])
    }
    pub fn delete_model(&self, id: &str) -> Result<bool> {
        Ok(self.conn.execute("DELETE FROM models WHERE id = ?", [id])? == 1)
    }
    pub fn add_prompt(
        &mut self,
        prompt: &Prompt,
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO prompts(id, body) VALUES (?, ?)",
            params![prompt.id, encode(prompt)?],
        )?;
        idempotency::record(&tx, submission, &prompt.id)?;
        tx.commit()?;
        Ok(())
    }
    pub fn prompt(&self, id: &str) -> Result<Option<Prompt>> {
        read_one(&self.conn, "SELECT body FROM prompts WHERE id = ?", id)
    }
    pub fn prompts(&self) -> Result<Vec<Prompt>> {
        read_list(&self.conn, "SELECT body FROM prompts ORDER BY rowid", [])
    }
    pub fn enqueue(
        &mut self,
        job: &Job,
        envelope: &Envelope,
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        insert_job(&tx, job, envelope)?;
        idempotency::record(&tx, submission, &job.id)?;
        tx.commit()?;
        Ok(())
    }
    pub fn conversions(&self) -> Result<Vec<Conversion>> {
        read_list(
            &self.conn,
            "SELECT body FROM conversions ORDER BY rowid",
            [],
        )
    }
    pub fn profiles(&self) -> Result<Vec<crate::profiles::Profile>> {
        read_list(
            &self.conn,
            "SELECT body FROM profile_revisions ORDER BY rowid",
            [],
        )
    }
    pub fn profile(&self, revision: &str) -> Result<Option<crate::profiles::Profile>> {
        read_one(
            &self.conn,
            "SELECT body FROM profile_revisions WHERE revision=?",
            revision,
        )
    }
    pub fn save_profile(&self, profile: &crate::profiles::Profile) -> Result<bool> {
        Ok(self.conn.execute(
            "INSERT INTO profile_revisions(revision,body) VALUES(?,?) ON CONFLICT(revision) DO NOTHING",
            params![profile.revision(), encode(profile)?],
        )? == 1)
    }
    pub fn conversion(&self, id: &str) -> Result<Option<Conversion>> {
        read_one(&self.conn, "SELECT body FROM conversions WHERE id=?", id)
    }
    pub fn create_conversion(
        &mut self,
        row: &Conversion,
        job: &Job,
        envelope: &Envelope,
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        insert_job(&tx, job, envelope)?;
        tx.execute(
            "INSERT INTO conversions(id,state,current_job,body) VALUES (?,?,?,?)",
            params![row.id, row.state.key(), job.id, encode(row)?],
        )?;
        idempotency::record(&tx, submission, &row.id)?;
        tx.commit()?;
        Ok(())
    }
    pub fn cancel_conversion(&mut self, id: &str) -> Result<Option<Conversion>> {
        let Some(mut row) = self.conversion(id)? else {
            return Ok(None);
        };
        if row.state.terminal() || row.cancel_requested {
            return Ok(Some(row));
        }
        let mut job = self
            .job(row.current_job())?
            .ok_or("conversion job is missing")?;
        let tx = self.conn.transaction()?;
        row.cancel_requested = true;
        row.updated_at = now();
        if request_cancel(&mut job) {
            update_job(&tx, &mut job, None)?;
        }
        if job.state.terminal() {
            row.finish(State::Cancelled, None);
        }
        save_conversion(&tx, &row)?;
        tx.commit()?;
        Ok(Some(row))
    }
    pub fn advance_conversions(&mut self) -> Result<()> {
        let pending: Vec<Conversion> = read_list(&self.conn,
            "SELECT c.body FROM conversions c JOIN jobs j ON j.id=c.current_job
             WHERE c.state='running' AND j.state IN ('succeeded','failed','cancelled') ORDER BY c.rowid", [])?;
        for mut row in pending {
            let job = self
                .job(row.current_job())?
                .ok_or("conversion job is missing")?;
            let child_cancelled = job.cancel_requested || job.state == State::Cancelled;
            let next = if row.cancel_requested || child_cancelled {
                row.cancel_requested = true;
                row.finish(State::Cancelled, None);
                None
            } else if job.state == State::Failed {
                row.finish(
                    State::Failed,
                    job.error
                        .or_else(|| Some(format!("{} job failed", row.stage))),
                );
                None
            } else {
                match self.next_conversion(&row, &job)? {
                    Ok(Some(next)) => Some(next),
                    Ok(None) => {
                        row.finish(State::Succeeded, None);
                        None
                    }
                    Err(error) => {
                        row.finish(State::Failed, Some(error));
                        None
                    }
                }
            };
            let verified_output =
                if row.format == crate::gguf::Format::Gguf && row.state == State::Succeeded {
                    Some(
                        self.envelope(&job.id)?
                            .inputs
                            .into_iter()
                            .nth(1)
                            .ok_or("verified GGUF model missing")?,
                    )
                } else {
                    None
                };
            let tx = self.conn.transaction()?;
            if let Some(model) = verified_output {
                tx.execute(
                    "INSERT INTO models(id,body) VALUES (?,?)",
                    params![model.id, encode(&model)?],
                )?;
                row.artifact_model_id = Some(model.id);
            }
            if let Some((next, envelope, artifact)) = next {
                if let Some(model) = artifact {
                    tx.execute(
                        "INSERT INTO models(id,body) VALUES (?,?)",
                        params![model.id, encode(&model)?],
                    )?;
                    row.artifact_model_id = Some(model.id);
                }
                insert_job(&tx, &next, &envelope)?;
                row.next(&next);
            }
            save_conversion(&tx, &row)?;
            tx.commit()?;
        }
        Ok(())
    }
    fn next_conversion(
        &self,
        row: &Conversion,
        job: &Job,
    ) -> Result<std::result::Result<Option<NextConversion>, String>> {
        let result = self.job_result(&job.id)?.unwrap_or(Value::Null);
        if row.format == crate::gguf::Format::Gguf {
            return self.next_gguf(row, job, &result);
        }
        if let Some(revision) = &job.profile_revision
            && (result["custom_profile"]["revision"].as_str() != Some(revision)
                || result["custom_profile"]["release_qualified"].as_bool() != Some(false))
        {
            return Ok(Err(
                "worker did not attest the saved custom profile revision".into(),
            ));
        }
        if job.kind == "verify" {
            let output = self
                .job_result(&row.job_ids["convert"])?
                .unwrap_or(Value::Null);
            return Ok(conversion::verification(&result, &output, &row.host).map(|_| None));
        }
        let inspect = self.envelope(&row.job_ids["inspect"])?;
        let inputs = inspect.inputs;
        let inspected = if inputs.len() > 1 && inputs[0].kind == "hf" {
            self.job_result(&row.job_ids["inspect"])?
        } else {
            None
        };
        Ok((|| {
            if let Some(inspected) = inspected.as_ref() {
                conversion::same_component_source(inspected, &result)?;
            }
            let (next, artifact) = match job.kind.as_str() {
                "inspect" => {
                    conversion::preflight(&result, "inspect")?;
                    ("plan", None)
                }
                "plan" => {
                    conversion::preflight(&result, "plan")?;
                    ("convert", None)
                }
                "convert" => ("verify", Some(conversion::artifact(&result, job)?)),
                _ => return Err("unexpected workflow stage".into()),
            };
            let (next, envelope) = row.build(
                next,
                inputs,
                artifact.clone(),
                inspect.options.get("profile_recipe"),
            )?;
            Ok(Some((next, envelope, artifact)))
        })())
    }
    fn next_gguf(
        &self,
        row: &Conversion,
        job: &Job,
        result: &Value,
    ) -> Result<std::result::Result<Option<NextConversion>, String>> {
        let inspect_id = &row.job_ids["inspect"];
        let inspect = self.envelope(inspect_id)?;
        let inspected = if job.id == *inspect_id {
            None
        } else {
            self.job_result(inspect_id)?
        };
        let published = if job.kind == "verify" {
            self.job_result(&row.job_ids["convert"])?
        } else {
            None
        };
        let profile = job
            .profile_revision
            .as_deref()
            .map(|id| self.profile(id))
            .transpose()?
            .flatten();
        Ok((|| {
            if job.profile_revision.is_some() && profile.is_none() {
                return Err("saved GGUF profile missing".into());
            }
            let selection = row.gguf_selection(&inspect, profile.as_ref())?;
            selection.receipt(job, result)?;
            if job.id != *inspect_id {
                conversion::same_component_source(
                    inspected.as_ref().unwrap_or(&Value::Null),
                    result,
                )?;
            }
            let (stage, artifact) = match job.kind.as_str() {
                "inspect" => {
                    conversion::preflight(result, "inspect")?;
                    ("plan", None)
                }
                "plan" => {
                    conversion::preflight(result, "plan")?;
                    ("convert", None)
                }
                "convert" => ("verify", Some(crate::gguf::artifact(job, result)?)),
                "verify" => {
                    crate::gguf::verification(
                        result,
                        published.as_ref().ok_or("prior GGUF artifact missing")?,
                        &row.host,
                    )?;
                    return Ok(None);
                }
                _ => return Err("unexpected GGUF stage".into()),
            };
            let recipe = serde_json::to_value(selection).map_err(|_| "invalid GGUF policy")?;
            let (next, envelope) = row.build(stage, inspect.inputs, artifact, Some(&recipe))?;
            // The private dispatcher can use the successful producer's output
            // before it is registered; only independent verification publishes it.
            Ok(Some((next, envelope, None)))
        })())
    }
    pub fn job_result(&self, id: &str) -> Result<Option<Value>> {
        read_one(
            &self.conn,
            "SELECT result FROM jobs WHERE id=? AND result IS NOT NULL",
            id,
        )
    }
    pub fn job(&self, id: &str) -> Result<Option<Job>> {
        read_one(&self.conn, "SELECT body FROM jobs WHERE id = ?", id)
    }
    pub fn jobs(&self) -> Result<Vec<Job>> {
        read_list(&self.conn, "SELECT body FROM jobs ORDER BY seq", [])
    }
    pub fn detail(&self, id: &str) -> Result<Option<Value>> {
        let Some(mut data): Option<Value> =
            read_one(&self.conn, "SELECT body FROM jobs WHERE id = ?", id)?
        else {
            return Ok(None);
        };
        if let Some(result) = read_one::<Value>(
            &self.conn,
            "SELECT result FROM jobs WHERE id = ? AND result IS NOT NULL",
            id,
        )? {
            data["result"] = result;
        }
        data["events"] = json!(read_list::<Event>(
            &self.conn,
            "SELECT body FROM events WHERE job_id = ? ORDER BY seq",
            [id]
        )?);
        Ok(Some(data))
    }
    pub fn envelope(&self, id: &str) -> Result<Envelope> {
        Ok(
            read_one(&self.conn, "SELECT envelope FROM jobs WHERE id = ?", id)?
                .ok_or("job envelope missing")?,
        )
    }
    pub fn by_state(&self, state: State) -> Result<Vec<Job>> {
        read_list(
            &self.conn,
            "SELECT body FROM jobs WHERE state = ? ORDER BY seq",
            [state.key()],
        )
    }
    pub fn host_has_jobs(&self, host: &str) -> rusqlite::Result<bool> {
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM jobs WHERE state IN ('queued','running')
             AND (json_extract(body,'$.host') = ? OR json_extract(body,'$.host') = 'auto'))",
            [host],
            |row| row.get(0),
        )
    }
    pub fn cancel(&mut self, id: &str) -> Result<Option<Job>> {
        let Some(mut job) = self.job(id)? else {
            return Ok(None);
        };
        if request_cancel(&mut job) {
            self.update(&mut job, None)?;
        }
        Ok(Some(job))
    }
    pub fn sample_progress(
        &self,
        id: &str,
        progress: Option<&str>,
        detail: Option<crate::domain::ProgressDetail>,
    ) -> Result<()> {
        let Some(mut job) = self.job(id)? else {
            return Ok(());
        };
        if job.state != State::Running || job.cancel_requested {
            return Ok(());
        }
        let detail_changed = match (&job.progress_detail, &detail) {
            (Some(old), Some(new)) => !old.same_measurement(new),
            (None, None) => false,
            _ => true,
        };
        if !detail_changed
            && progress.is_none_or(|text| text == job.progress && job.error.is_none())
        {
            return Ok(());
        }
        if let Some(text) = progress {
            job.progress = text.into();
            job.error = None;
        }
        if detail_changed {
            job.progress_detail = detail;
        }
        job.updated_at = now();
        // Keep only the latest sample; append-only history is for lifecycle
        // events, not every byte counter update during a long transfer.
        self.conn.execute(
            "UPDATE jobs SET body=? WHERE id=? AND state='running'",
            params![encode(&job)?, id],
        )?;
        Ok(())
    }
    pub fn update(&mut self, job: &mut Job, mut result: Option<Value>) -> Result<()> {
        // One transaction: no reader can see a terminal success without its immutable result.
        if job.state == State::Succeeded
            && let Err(error) =
                crate::weights::check_source_receipt(job, result.as_ref().unwrap_or(&Value::Null))
        {
            job.state = State::Failed;
            job.progress = "Weight source fingerprint check failed".into();
            job.error = Some(error.into());
            job.has_result = false;
            result = None;
        }
        if job.state == State::Succeeded
            && job.kind != "weight_apply"
            && crate::weights::is_kind(&job.kind)
        {
            let receipt = result.as_ref().unwrap_or(&Value::Null);
            let envelope = self.envelope(&job.id)?;
            if let Err(error) = crate::weights::check_packed_receipt(job, &envelope, receipt)
                .and_then(|()| crate::weights::check_safetensors_receipt(job, &envelope, receipt))
            {
                job.state = State::Failed;
                job.progress = "Weight receipt check failed".into();
                job.error = Some(error.into());
                job.has_result = false;
                result = None;
            }
        }
        let artifact = if job.kind == "hf_download" && job.state == State::Succeeded {
            let checked = if job.cancel_requested {
                job.state = State::Cancelled;
                job.progress =
                    "Hugging Face download cancelled; artifact was not registered".into();
                None
            } else {
                match crate::hf_download::verified_artifact(
                    job,
                    result.as_ref().unwrap_or(&Value::Null),
                ) {
                    Ok(model) => {
                        result.as_mut().expect("validated receipt")["library_model_id"] =
                            json!(model.id);
                        Some(model)
                    }
                    Err(error) => {
                        job.state = State::Failed;
                        job.progress = "Hugging Face download artifact was not registered".into();
                        job.error = Some(error.into());
                        None
                    }
                }
            };
            if checked.is_none() {
                result = None;
                job.has_result = false;
            }
            checked
        } else if job.kind == "weight_apply" && job.state == State::Succeeded {
            let envelope = self.envelope(&job.id)?;
            match crate::weights::verified_artifact(
                job,
                &envelope,
                result.as_ref().unwrap_or(&Value::Null),
            ) {
                Ok(model) if !job.cancel_requested => {
                    result.as_mut().expect("validated receipt")["library_model_id"] =
                        json!(model.id);
                    Some(model)
                }
                outcome => {
                    job.state = if job.cancel_requested {
                        State::Cancelled
                    } else {
                        State::Failed
                    };
                    job.progress = "Weight edit output was not registered".into();
                    job.error = outcome.err();
                    job.has_result = false;
                    result = None;
                    None
                }
            }
        } else {
            None
        };
        let tx = self.conn.transaction()?;
        if let Some(model) = artifact {
            tx.execute(
                "INSERT INTO models(id,body) VALUES (?,?)",
                params![model.id, encode(&model)?],
            )?;
        }
        update_job(&tx, job, result)?;
        tx.commit()?;
        Ok(())
    }
}

type NextConversion = (Job, Envelope, Option<Model>);

fn save_conversion(tx: &Transaction<'_>, row: &Conversion) -> Result<()> {
    tx.execute(
        "UPDATE conversions SET state=?,current_job=?,body=? WHERE id=?",
        params![row.state.key(), row.current_job(), encode(row)?, row.id],
    )?;
    Ok(())
}

fn insert_job(tx: &Transaction<'_>, job: &Job, envelope: &Envelope) -> Result<()> {
    tx.execute(
        "INSERT INTO jobs(id,state,body,envelope) VALUES (?,?,?,?)",
        params![job.id, job.state.key(), encode(job)?, encode(envelope)?],
    )?;
    event(tx, job)
}

fn request_cancel(job: &mut Job) -> bool {
    if job.state.terminal() || job.cancel_requested {
        return false;
    }
    job.cancel_requested = true;
    job.progress_detail = None;
    job.error = None;
    if job.state == State::Queued {
        job.state = State::Cancelled;
        job.progress = "Cancelled before dispatch".into();
    } else {
        job.progress = "Cancellation requested; waiting for the worker".into();
    }
    true
}

fn update_job(tx: &Transaction<'_>, job: &mut Job, result: Option<Value>) -> Result<()> {
    job.updated_at = now();
    let count = tx.execute(
        "UPDATE jobs SET state=?, body=?, result=? WHERE id=? AND state IN ('queued', 'running')",
        params![
            job.state.key(),
            encode(job)?,
            result.map(|r| encode(&r)).transpose()?,
            job.id
        ],
    )?;
    if count != 1 {
        return Err("cannot modify a terminal or missing job".into());
    }
    event(tx, job)
}

fn encode(row: &impl Serialize) -> Result<String> {
    Ok(serde_json::to_string(row)?)
}

fn event(tx: &Transaction<'_>, job: &Job) -> Result<()> {
    tx.execute(
        "INSERT INTO events(job_id, body) VALUES (?, ?)",
        params![job.id, encode(&Event::from(job))?],
    )?;
    Ok(())
}

fn read_one<T: DeserializeOwned>(conn: &Connection, sql: &str, key: &str) -> Result<Option<T>> {
    let text: Option<String> = conn.query_row(sql, [key], |r| r.get(0)).optional()?;
    Ok(text.map(|s| serde_json::from_str(&s)).transpose()?)
}

fn read_list<T: DeserializeOwned>(
    conn: &Connection,
    sql: &str,
    args: impl rusqlite::Params,
) -> Result<Vec<T>> {
    let mut statement = conn.prepare(sql)?;
    let rows = statement.query_map(args, |row| row.get::<_, String>(0))?;
    rows.map(|row| Ok(serde_json::from_str(&row?)?)).collect()
}
