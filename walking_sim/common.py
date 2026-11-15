from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

import gymnasium as gym


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "humanoid-v5-sac-expert.zip"


@contextmanager
def make_environment(render_mode=None):
    # MuJoCo on Windows may not open XML files under non-ASCII paths.
    asset = Path(gym.__file__).parent / "envs" / "mujoco" / "assets" / "humanoid.xml"
    with tempfile.TemporaryDirectory(prefix="mujoco_humanoid_") as temporary:
        xml = Path(temporary) / "humanoid.xml"
        shutil.copyfile(asset, xml)
        env = gym.make("Humanoid-v5", xml_file=str(xml), render_mode=render_mode)
        try:
            yield env
        finally:
            env.close()
