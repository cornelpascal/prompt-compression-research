"""Dependency-free checks for the optional TPU boundary in both packages."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    ROOT / "research/prompt_compression/promptcomp/devices.py",
    ROOT / "research/controlled_leakage/leaklab/devices.py",
)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.parent.name + "_devices_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeviceBoundaryTest(unittest.TestCase):
    def test_xla_is_opt_in_and_steps_are_flushed(self):
        events = []
        torch = types.ModuleType("torch")
        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return "cpu tensor"

        torch.Tensor = FakeTensor
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.device = lambda name: types.SimpleNamespace(type=name)
        xla = types.ModuleType("torch_xla")
        xla.__path__ = []
        xla.device = lambda: types.SimpleNamespace(type="xla")
        core = types.ModuleType("torch_xla.core")
        core.__path__ = []
        xm = types.ModuleType("torch_xla.core.xla_model")
        xm.mark_step = lambda: events.append("mark")
        xm.wait_device_ops = lambda: events.append("wait")
        modules = {"torch": torch, "torch_xla": xla, "torch_xla.core": core,
                   "torch_xla.core.xla_model": xm}
        for path in MODULES:
            with self.subTest(package=path.parent.name), patch.dict(sys.modules, modules), patch.dict(
                "os.environ", {"PROMPTCOMP_DEVICE": "xla", "LEAKLAB_DEVICE": "xla"}
            ):
                device_module = load_module(path)
                device = device_module.resolve_device()
                self.assertEqual(device.type, "xla")
                optimizer = types.SimpleNamespace(step=lambda: events.append("step"))
                device_module.step(optimizer, device)
                self.assertEqual(events[-2:], ["step", "mark"])
                if hasattr(device_module, "synchronize"):
                    device_module.synchronize(device)
                else:
                    device_module.release_cache(device)
                self.assertEqual(events[-2:], ["mark", "wait"])
                self.assertEqual(device_module.cpu_tree({"state": [FakeTensor()]}),
                                 {"state": ["cpu tensor"]})
                self.assertEqual(device_module.resolve_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
