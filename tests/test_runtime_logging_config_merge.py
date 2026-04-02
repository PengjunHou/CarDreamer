import unittest

from train_config_merge import diff_explicit_config_overrides


class RuntimeLoggingConfigMergeTest(unittest.TestCase):
    def test_no_cli_override_produces_empty_diff(self):
        base = {
            "level": "INFO",
            "file_level": "INFO",
            "step_debug_interval": 100,
        }
        parsed = {
            "level": "INFO",
            "file_level": "INFO",
            "step_debug_interval": 100,
        }
        self.assertEqual(diff_explicit_config_overrides(parsed, base), {})

    def test_cli_override_keeps_only_changed_runtime_logging_fields(self):
        base = {
            "level": "INFO",
            "file_level": "INFO",
            "step_debug_interval": 100,
            "console_key_only": True,
        }
        parsed = {
            "level": "DEBUG",
            "file_level": "DEBUG",
            "step_debug_interval": 1,
            "console_key_only": True,
        }
        self.assertEqual(
            diff_explicit_config_overrides(parsed, base),
            {
                "level": "DEBUG",
                "file_level": "DEBUG",
                "step_debug_interval": 1,
            },
        )

    def test_nested_override_keeps_only_changed_fields(self):
        base = {
            "run": {
                "steps": 100,
                "save_every": 10,
            },
            "wandb": {
                "enable": True,
                "project": "CardDreamer",
            },
        }
        parsed = {
            "run": {
                "steps": 200,
                "save_every": 10,
            },
            "wandb": {
                "enable": True,
                "project": "CardDreamer",
            },
        }
        self.assertEqual(
            diff_explicit_config_overrides(parsed, base),
            {"run": {"steps": 200}},
        )


if __name__ == "__main__":
    unittest.main()
