use crate::conversion::{self, Conversion};
use crate::domain::{Envelope, Event, Job, Model, NewModel, Prompt, State, now};
use crate::idempotency::{self, Replay, Submission};
use crate::scratch::{Revision, RevisionState};
use brewery_worker::hgn_recipe;
use brewery_worker::operational::Settings;
use rusqlite::{Connection, OptionalExtension as _, Transaction, params};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    fs::{File, OpenOptions},
    os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _, PermissionsExt as _},
    path::{Path, PathBuf},
};

mod interventions;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct Store {
    conn: Connection,
    _lock: File,
    scratch: PathBuf,
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
              DROP INDEX IF EXISTS one_job_per_active_host;
              CREATE UNIQUE INDEX IF NOT EXISTS one_job_per_active_host_slot
                ON jobs(json_extract(body, '$.host'),
                        coalesce(json_extract(body, '$.host_slot'), 0))
                WHERE state = 'running';
             CREATE TABLE IF NOT EXISTS events(
               seq INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), body TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS job_events ON events(job_id, seq);",
        )?;
        crate::feed::install(&conn)?;
        idempotency::install(&conn)?;
        interventions::install(&conn)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS profile_revisions(revision TEXT PRIMARY KEY, body TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS host_evaluation_profiles(
                 binding TEXT PRIMARY KEY, body TEXT NOT NULL
             );",
        )?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversions(
                id TEXT PRIMARY KEY, state TEXT NOT NULL,
                current_job TEXT NOT NULL REFERENCES jobs(id), body TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS conversions_active ON conversions(state);",
        )?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS scratch_revisions(
                id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, state TEXT NOT NULL, body TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS scratch_parent ON scratch_revisions(parent_id, state);",
        )?;
        Ok(Self {
            conn,
            _lock: lock,
            scratch: directory.canonicalize()?.join("scratch"),
        })
    }

    pub fn scratch_root(&self) -> &Path {
        &self.scratch
    }

    pub fn is_scratch_path(&self, path: &Path) -> bool {
        path.starts_with(&self.scratch)
            || path
                .canonicalize()
                .is_ok_and(|path| path.starts_with(&self.scratch))
    }

    pub fn is_private_editor_path(&self, path: &Path) -> Result<bool> {
        let lanes = self.scratch.with_file_name("local-edit");
        let candidate = |path: &Path| -> Result<bool> {
            // Do not admit an enclosing directory as an HF or bundle source.
            if lanes.starts_with(path) {
                return Ok(true);
            }
            let Ok(relative) = path.strip_prefix(&lanes) else {
                return Ok(false);
            };
            let mut parts = relative.iter();
            parts.next(); // worker host
            if parts.next().and_then(|p| p.to_str()) != Some("worker")
                || parts.next().and_then(|p| p.to_str()) != Some("jobs")
            {
                return Ok(false);
            }
            let Some(id) = parts.next().and_then(|p| p.to_str()) else {
                return Ok(false);
            };
            Ok(self.job(id)?.is_some_and(|job| job.scratch_job))
        };
        Ok(candidate(path)? || match path.canonicalize() {
            Ok(path) => candidate(&path)?,
            Err(_) => false,
        })
    }

    pub fn scratch_has_jobs(&self, id: &str, kind: Option<&str>) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM jobs j WHERE j.state IN ('queued','running')
             AND json_extract(j.body,'$.scratch_job') = 1
             AND (?2 IS NULL OR json_extract(j.body,'$.kind') = ?2)
             AND EXISTS(SELECT 1 FROM json_each(j.body,'$.input_ids') WHERE value = ?1))",
            params![id, kind],
            |row| row.get(0),
        )?)
    }

    pub fn scratch(&self, id: &str) -> Result<Option<Revision>> {
        read_one(
            &self.conn,
            "SELECT body FROM scratch_revisions WHERE id=?",
            id,
        )
    }

    pub fn scratches(&self) -> Result<Vec<Revision>> {
        read_list(
            &self.conn,
            "SELECT body FROM scratch_revisions WHERE state <> 'discarded' ORDER BY rowid",
            [],
        )
    }

    pub fn scratch_count(&self) -> Result<u32> {
        Ok(self.conn.query_row(
            "SELECT count(*) FROM scratch_revisions WHERE state <> 'discarded'",
            [],
            |row| row.get(0),
        )?)
    }

    pub fn has_scratch_children(&self, id: &str) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM scratch_revisions WHERE parent_id=? AND state <> 'discarded')",
            [id],
            |row| row.get(0),
        )?)
    }

    pub fn insert_scratch(
        &mut self,
        row: &Revision,
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO scratch_revisions(id,parent_id,state,body) VALUES(?,?,?,?)",
            params![
                row.id,
                row.provenance.parent_id,
                row.state.key(),
                encode(row)?
            ],
        )?;
        idempotency::record(&tx, submission, &row.id)?;
        tx.commit()?;
        Ok(())
    }

    pub fn update_scratch(
        &mut self,
        row: &Revision,
        old: RevisionState,
        library: Option<&Model>,
        submission: Option<&Submission<'_>>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        if tx.execute(
            "UPDATE scratch_revisions SET state=?, body=? WHERE id=? AND state=?",
            params![row.state.key(), encode(row)?, row.id, old.key()],
        )? != 1
        {
            return Err("scratch state changed before commit".into());
        }
        if let Some(model) = library {
            tx.execute(
                "INSERT INTO models(id, body) VALUES (?, ?)",
                params![model.id, encode(model)?],
            )?;
        }
        idempotency::record(&tx, submission, &row.id)?;
        tx.commit()?;
        Ok(())
    }

    pub fn feed_cursor(&self) -> rusqlite::Result<i64> {
        crate::feed::cursor(&self.conn)
    }

    pub fn evaluation_profile(&self, binding: &str) -> Result<Option<Settings>> {
        read_one(
            &self.conn,
            "SELECT body FROM host_evaluation_profiles WHERE binding = ?",
            binding,
        )
    }

    // The caller holds App's Store lock across revision comparison and write.
    pub fn save_evaluation_profile(&self, binding: &str, settings: &Settings) -> Result<()> {
        self.conn.execute(
            "INSERT INTO host_evaluation_profiles(binding, body) VALUES (?, ?)
             ON CONFLICT(binding) DO UPDATE SET body=excluded.body",
            params![binding, encode(settings)?],
        )?;
        Ok(())
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
    pub fn backend_publication(&self, model: &Model) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM jobs
             WHERE state='succeeded' AND result IS NOT NULL
               AND json_extract(body,'$.host')=?1
               AND json_extract(body,'$.kind') IN ('hf_download','weight_apply')
               AND json_extract(body,'$.cancel_requested')=0
               AND json_extract(result,'$.verified')=1
               AND json_extract(result,'$.library_model_id')=?2
               AND json_extract(result,'$.artifact.path')=?3
               AND json_extract(result,'$.artifact.kind')=?4)",
            params![crate::backend_hf::ID, model.id, model.path, model.kind],
            |row| row.get(0),
        )?)
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
            let verified_output = if (row.format == crate::gguf::Format::Gguf
                || row.native_layer_recipe())
                && row.state == State::Succeeded
            {
                Some(
                    self.envelope(&job.id)?
                        .inputs
                        .into_iter()
                        .nth(1)
                        .ok_or("verified conversion output missing")?,
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
        if row.native_layer_recipe() {
            return self.next_hgn_recipe(row, job, &result);
        }
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

    fn next_hgn_recipe(
        &self,
        row: &Conversion,
        job: &Job,
        result: &Value,
    ) -> Result<std::result::Result<Option<NextConversion>, String>> {
        use sha2::{Digest, Sha256};
        let inspect_id = &row.job_ids["inspect"];
        let inspect = self.envelope(inspect_id)?;
        let inspected = self.job_result(inspect_id)?.unwrap_or(Value::Null);
        let convert = row
            .job_ids
            .get("convert")
            .map(|id| self.job_result(id))
            .transpose()?
            .flatten();
        let current = self.envelope(&job.id)?;
        Ok((|| {
            let plan = &result["layer_recipe_plan"];
            let receipt = &result["layer_recipe_receipt"];
            let expected_sha = format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(plan).map_err(|_| "invalid native HGN plan")?)
            );
            let policy = &inspect.options["layer_recipe"];
            let inputs = result["inputs"]
                .as_array()
                .ok_or("missing native HGN input receipts")?;
            if result["kind"] != job.kind
                || result["host"] != job.host
                || result["output_format"] != "hgn"
                || result["report"]["passed"] != true
                || result["runtime_load_verified"] != false
                || result["release_qualified"] != false
                || inputs.len() != current.inputs.len()
                || inputs
                    .iter()
                    .zip(&current.inputs)
                    .any(|(actual, expected)| {
                        actual["id"] != expected.id
                            || actual["path"] != expected.path
                            || actual["host"] != expected.host
                            || actual["kind"] != expected.kind
                            || actual["source_host"] != expected.host
                    })
                || inputs[0]["sha256"] != policy["source_sha256"]
                || plan["recipe"] != *policy
                || plan["source_bundle_sha256"] != result["source_sha256"]
                || result["layer_plan_sha256"] != expected_sha
                || ![
                    "helper_sha256",
                    "exporter_sha256",
                    "python_sha256",
                    "safetensors_reader_sha256",
                    "source_bundle_sha256",
                ]
                .iter()
                .all(|key| plan[key].as_str().is_some_and(brewery_worker::is_digest))
                || !plan["compiled"]["tensors"]
                    .as_array()
                    .is_some_and(|t| !t.is_empty())
                || plan["calibration"] != "none"
            {
                return Err("worker did not attest the source-bound native HGN layer plan".into());
            }
            if job.id != *inspect_id
                && (inspected["layer_recipe_plan"] != *plan
                    || inspected["layer_plan_sha256"] != expected_sha)
            {
                return Err("native HGN layer plan changed after inspection".into());
            }
            if matches!(job.kind.as_str(), "convert" | "verify")
                && (result["structural_reader_verified"] != true
                    || receipt["structural_reader_verified"] != true
                    || receipt["runtime_load_verified"] != false
                    || receipt["format"] != "native_hgn_v2"
                    || receipt["source_sha256"] != policy["source_sha256"]
                    || receipt["trellis"] != plan["compiled"]["trellis"])
            {
                return Err("native HGN structural reader attestation missing".into());
            }
            let (stage, output) = match job.kind.as_str() {
                "inspect" | "plan" => {
                    if result["report"]["disk"]["fits"] != true {
                        return Err("native HGN layer recipe does not fit on disk".into());
                    }
                    (
                        if job.kind == "inspect" {
                            "plan"
                        } else {
                            "convert"
                        },
                        None,
                    )
                }
                "convert" => {
                    let a = &result["artifact"];
                    let path = a["path"].as_str().unwrap_or_default();
                    if a["kind"] != hgn_recipe::ARCHIVE
                        || !path.starts_with('/')
                        || !path.ends_with(&format!("/jobs/{}/work/{}", job.id, hgn_recipe::OUTPUT))
                        || path.contains('\\')
                        || path[1..].split('/').any(|s| matches!(s, "" | "." | ".."))
                        || !a["sha256"].as_str().is_some_and(brewery_worker::is_digest)
                        || !a["bytes"].as_u64().is_some_and(|n| n > 0)
                    {
                        return Err("worker did not return a valid native HGN archive".into());
                    }
                    (
                        "verify",
                        Some(
                            NewModel {
                                name: Some(format!("Structural HGN archive ({})", job.id)),
                                host: job.host.clone(),
                                kind: hgn_recipe::ARCHIVE.into(),
                                path: path.into(),
                            }
                            .validate()?,
                        ),
                    )
                }
                "verify" => {
                    let published = convert
                        .as_ref()
                        .ok_or("missing native HGN producer result")?;
                    let artifact = &published["artifact"];
                    let verdict = &result["file_verification"];
                    if verdict["status"] != "passed"
                        || verdict["scope"] != "structure_only"
                        || verdict["archive_sha256"] != artifact["sha256"]
                        || verdict["model_sha256"] != receipt["model"]["sha256"]
                        || receipt != &published["layer_recipe_receipt"]
                        || ["kind", "path", "sha256", "bytes", "files"]
                            .iter()
                            .any(|key| inputs[1][key] != artifact[key])
                    {
                        return Err("independent native HGN verification did not attest the producer archive".into());
                    }
                    return Ok(None);
                }
                _ => return Err("unexpected native HGN recipe stage".into()),
            };
            let (next, envelope) = row.build(
                stage,
                inspect.inputs,
                output,
                Some(&result["layer_plan_sha256"]),
            )?;
            Ok(Some((next, envelope, None)))
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
        if let Some(policy) = inspect.options.get("layer_recipe") {
            return Ok((|| {
                crate::gguf::layer_receipt(job, result, policy)?;
                if let Some(inspected) = &inspected {
                    conversion::same_component_source(inspected, result)?;
                    if inspected["layer_plan_sha256"] != result["layer_plan_sha256"]
                        || inspected["layer_recipe_plan"] != result["layer_recipe_plan"]
                    {
                        return Err("GGUF layer recipe plan changed after inspection".into());
                    }
                } else if job.id != *inspect_id {
                    return Err("prior GGUF layer recipe inspection missing".into());
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
                    _ => return Err("unexpected GGUF layer recipe stage".into()),
                };
                let (next, envelope) = row.build(
                    stage,
                    inspect.inputs.clone(),
                    artifact,
                    Some(&result["layer_plan_sha256"]),
                )?;
                Ok(Some((next, envelope, None)))
            })());
        }
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
                let inspected = inspected.as_ref().unwrap_or(&Value::Null);
                conversion::same_component_source(inspected, result)?;
                if selection.encoder_family.is_some()
                    && inspected["backend_receipt"]["executable_sha256"]
                        != result["backend_receipt"]["executable_sha256"]
                {
                    return Err("ROCmFPX executable changed after inspection".into());
                }
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
    pub fn ssh_host_in_use(&self, host: &str) -> rusqlite::Result<bool> {
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM jobs j WHERE
              (j.state IN ('queued','running') OR EXISTS(
                 SELECT 1 FROM conversions c WHERE c.current_job=j.id AND c.state IN ('queued','running')
              )) AND (json_extract(j.body,'$.host') IN (?1,'auto')
                OR EXISTS(SELECT 1 FROM json_each(j.envelope,'$.inputs')
                   WHERE json_extract(value,'$.host')=?1)))",
            [host],
            |row| row.get(0),
        )
    }

    pub fn host_has_history(&self, host: &str) -> rusqlite::Result<bool> {
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM models WHERE lower(json_extract(body,'$.host'))=lower(?1))
                 OR EXISTS(SELECT 1 FROM jobs WHERE lower(json_extract(body,'$.host'))=lower(?1)
                     OR EXISTS(SELECT 1 FROM json_each(jobs.envelope,'$.inputs')
                         WHERE lower(json_extract(value,'$.host'))=lower(?1)))",
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
        let mut scratch = None;
        if job.scratch_job && job.state == State::Succeeded {
            let verified = if job.cancel_requested {
                job.state = State::Cancelled;
                job.progress = "Scratch job cancelled".into();
                Err("cancelled".into())
            } else {
                crate::scratch::jobs::finish(self, job, result.as_ref().unwrap_or(&Value::Null))
            };
            match verified {
                Ok(Some((revision, safe_result))) => {
                    scratch = Some(revision);
                    result = Some(safe_result);
                }
                Ok(None) => {}
                Err(error) => {
                    if job.state != State::Cancelled {
                        job.state = State::Failed;
                        job.progress = "Scratch job result was not accepted".into();
                        job.error = Some(error.to_string());
                    }
                    job.has_result = false;
                    result = None;
                }
            }
        }
        if job.scratch_job && matches!(job.state, State::Failed | State::Cancelled) {
            crate::scratch::jobs::clean_failed_conversion(self, job);
        }
        if job.state == State::Succeeded && job.kind == brewery_worker::exl3_layer::JOB {
            let receipt = result.as_ref().unwrap_or(&Value::Null);
            let envelope = self.envelope(&job.id)?;
            let observed = &receipt["inputs"][0];
            let check = brewery_worker::exl3_layer::Spec::parse(&Value::Object(envelope.options))
                .and_then(|spec| spec.check_receipt(&job.host, receipt));
            // cancel_requested is intent, not a worker acknowledgement. If
            // the worker committed success first, keep its verified receipt.
            if check.is_err()
                || !receipt["inputs"]
                    .as_array()
                    .is_some_and(|rows| rows.len() == 1)
                || !matches!(envelope.inputs.as_slice(), [source] if
                    observed["id"] == source.id
                    && observed["path"] == source.path
                    && observed["host"] == source.host
                    && observed["source_host"] == source.host
                    && observed["kind"] == source.kind
                    && receipt["artifact"]["path"] != source.path)
                || observed["sha256"] != receipt["source_sha256"]
            {
                job.state = State::Failed;
                job.progress = "EXL3 layer artifact was not accepted".into();
                job.error = Some("Invalid EXL3 layer receipt".into());
                job.has_result = false;
                result = None;
            }
        }
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
        if let Some(row) = scratch {
            tx.execute(
                "UPDATE scratch_revisions SET body=? WHERE id=? AND state='scratch'",
                params![encode(&row)?, row.id],
            )?;
        }
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
