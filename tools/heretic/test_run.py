import hashlib
from pathlib import Path
import tempfile
import unittest

import run


class PolicyTest(unittest.TestCase):
    def test_never_uses_injected_action_or_plugin(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaisesRegex(ValueError, "plugin"):
                run.prepare(
                    {"scorers": [{"plugin": "/tmp/plugin.py:Bad"}]}, root, root, 9, 0
                )
            effective = run.prepare({"model_action": "upload"}, root, root, 9, 0)
            self.assertEqual(effective["model_action"], "save")
            self.assertEqual(effective["seed"], 9)
            self.assertEqual(effective["export_strategy"], "merge")

    def test_requires_locally_pinned_datasets(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "prompts").mkdir()
            with self.assertRaisesRegex(ValueError, "offline prompt"):
                run.local_datasets({"dataset": "example/remote"}, root)

    def test_checks_the_prompt_bytes_and_uses_a_private_copy(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "prompts").mkdir()
            work = root / "work"
            work.mkdir()
            data = b"One reviewed prompt.\n"
            sha = hashlib.sha256(data).hexdigest()
            path = root / "prompts" / f"{sha}.txt"
            path.write_bytes(data)
            pins = {}
            spec = {"dataset": "example/remote", "split": "train[:1]", "brewery_sha256": sha}
            effective = run.local_datasets(spec, root, work, pins)
            self.assertEqual(effective, {"dataset": str(work / path.name), "split": "[:1]"})
            self.assertEqual(pins, {"example/remote": sha})
            path.write_bytes(b"Changed")
            with self.assertRaisesRegex(ValueError, "offline prompt SHA-256 mismatch"):
                run.local_datasets(spec, root)


    def test_study_summary_reports_trials_without_display_payloads(self):
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "study.jsonl"
            store = JournalStorage(JournalFileBackend(str(checkpoint)))
            study = optuna.create_study(study_name="heretic", storage=store)
            trial = study.ask()
            trial.set_user_attr("index", 1)
            trial.set_user_attr("scores", [{
                "name": "KL divergence", "score": {"value": 0.125, "rich_display": "ignored"},
                "baseline": {"value": 0.0, "md_display": "ignored"},
            }])
            study.tell(trial, 0.125)
            self.assertEqual(run.study_trials(Path(root), 1, 1), [{
                "index": 1, "scores": [{"name": "KL divergence", "value": 0.125, "baseline": 0.0}],
            }])


if __name__ == "__main__":
    unittest.main()
