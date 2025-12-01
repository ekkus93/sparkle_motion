import importlib
from pathlib import Path


def test_root_agent_local_validation_passes():
    mod = importlib.import_module("scripts.register_root_agent_local")
    cfg_path = Path("configs/root_agent.yaml")
    assert cfg_path.exists()
    data = mod.load_yaml(cfg_path)
    assert mod.validate_root_agent(data, config_path=cfg_path)
