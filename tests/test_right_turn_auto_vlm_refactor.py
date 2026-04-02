import ast
import py_compile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_DIR = REPO_ROOT / "car_dreamer" / "toolkit" / "vlm"


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class_def(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class '{class_name}' not found in module.")


class RightTurnAutoVLMRefactorSmokeTest(unittest.TestCase):
    def test_key_modules_compile(self):
        paths = [
            REPO_ROOT / "car_dreamer" / "toolkit" / "vlm" / "right_turn_auto_prompts.py",
            REPO_ROOT / "car_dreamer" / "toolkit" / "vlm" / "right_turn_auto_context.py",
            REPO_ROOT / "car_dreamer" / "toolkit" / "vlm" / "right_turn_auto_scoring.py",
            REPO_ROOT / "car_dreamer" / "toolkit" / "vlm" / "right_turn_auto_mixin.py",
            REPO_ROOT / "car_dreamer" / "right_turn_auto_runtime.py",
            REPO_ROOT / "car_dreamer" / "carla_group_right_turn_auto_env.py",
        ]
        for path in paths:
            py_compile.compile(str(path), doraise=True)

    def test_scoring_mixin_inherits_context_mixin(self):
        module = _parse_module(VLM_DIR / "right_turn_auto_scoring.py")
        class_def = _class_def(module, "RightTurnAutoVLMScoringMixin")
        bases = [base.id for base in class_def.bases if isinstance(base, ast.Name)]
        self.assertIn("RightTurnAutoVLMContextMixin", bases)

    def test_top_level_mixin_has_single_scoring_base(self):
        module = _parse_module(VLM_DIR / "right_turn_auto_mixin.py")
        class_def = _class_def(module, "RightTurnAutoVLMMixin")
        bases = [base.id for base in class_def.bases if isinstance(base, ast.Name)]
        self.assertEqual(bases, ["RightTurnAutoVLMScoringMixin"])

    def test_context_mixin_provides_shared_image_helpers(self):
        module = _parse_module(VLM_DIR / "right_turn_auto_context.py")
        class_def = _class_def(module, "RightTurnAutoVLMContextMixin")
        method_names = {
            node.name for node in class_def.body if isinstance(node, ast.FunctionDef)
        }
        required = {
            "_normalize_pose_dict",
            "_normalize_velocity_dict",
            "_get_received_messages_in_window",
            "_get_raw_shared_images_info",
            "_get_received_shared_images_info",
            "_get_ego_and_shared_images_info",
        }
        self.assertTrue(required.issubset(method_names))

    def test_scoring_module_has_no_print_or_traceback_print_exc(self):
        module = _parse_module(VLM_DIR / "right_turn_auto_scoring.py")
        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "print")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                is_traceback = (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "traceback"
                    and node.func.attr == "print_exc"
                )
                self.assertFalse(is_traceback)


if __name__ == "__main__":
    unittest.main()
