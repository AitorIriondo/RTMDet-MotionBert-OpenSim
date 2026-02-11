# Linux Migration Guide

This document describes everything needed to make the RTMDet-MotionBert-OpenSim pipeline work on Linux. The codebase was developed on Windows and has hardcoded Windows paths that need to be made cross-platform.

---

## 1. Hardcoded Paths to Fix

### Pose2Sim Python executable (5 occurrences)

The pipeline calls OpenSim IK as a **subprocess** using a separate conda environment's Python. The path is hardcoded to the Windows user's anaconda installation.

**`run_hybrid_pipeline.py` line 928:**
```python
POSE2SIM_PYTHON = r"C:\Users\iria\AppData\Local\anaconda3\envs\Pose2Sim\python.exe"
```

**`run_export.py` lines 243, 437, 553:**
```python
POSE2SIM_PYTHON = r"C:\Users\iria\AppData\Local\anaconda3\envs\Pose2Sim\python.exe"
```

**Fix:** Replace all occurrences with a cross-platform discovery function. Create a helper in `utils/io_utils.py`:
```python
def find_pose2sim_python() -> str:
    """Find the Pose2Sim conda environment's Python executable."""
    import shutil
    # 1. Check environment variable
    env_path = os.environ.get("POSE2SIM_PYTHON")
    if env_path and Path(env_path).exists():
        return env_path
    # 2. Try common conda locations
    conda_base = os.environ.get("CONDA_PREFIX", "").replace("/envs/mmpose", "").replace("/envs/Pose2Sim", "")
    if not conda_base:
        conda_base = str(Path.home() / "miniconda3")
    candidates = [
        Path(conda_base) / "envs" / "Pose2Sim" / "bin" / "python",        # Linux
        Path(conda_base) / "envs" / "Pose2Sim" / "python.exe",            # Windows
        Path.home() / "miniconda3" / "envs" / "Pose2Sim" / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / "Pose2Sim" / "bin" / "python",
        Path.home() / "AppData" / "Local" / "anaconda3" / "envs" / "Pose2Sim" / "python.exe",  # Windows user
        Path("C:/ProgramData/anaconda3/envs/Pose2Sim/python.exe"),         # Windows system
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # 3. Fallback: current interpreter (works if opensim is in current env)
    try:
        import opensim
        return sys.executable
    except ImportError:
        pass
    raise FileNotFoundError("Cannot find Pose2Sim conda environment. Set POSE2SIM_PYTHON env var.")
```

Then replace every `POSE2SIM_PYTHON = r"C:\Users\iria\..."` with:
```python
from utils.io_utils import find_pose2sim_python
POSE2SIM_PYTHON = find_pose2sim_python()
```

### Pose2Sim OpenSim_Setup path (`run_export.py` line 558)

```python
pose2sim_setup = Path(r"C:\Users\iria\AppData\Local\anaconda3\envs\Pose2Sim\Lib\site-packages\Pose2Sim\OpenSim_Setup")
```

**Fix:** Derive from the Pose2Sim Python path:
```python
# Linux: .../envs/Pose2Sim/lib/python3.12/site-packages/Pose2Sim/OpenSim_Setup
# Windows: .../envs/Pose2Sim/Lib/site-packages/Pose2Sim/OpenSim_Setup
import subprocess, json
result = subprocess.run(
    [POSE2SIM_PYTHON, "-c", "import Pose2Sim, pathlib; print(pathlib.Path(Pose2Sim.__file__).parent / 'OpenSim_Setup')"],
    capture_output=True, text=True
)
pose2sim_setup = Path(result.stdout.strip())
```

### Blender path (`run_export.py` line 641)

```python
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
```

**Fix:** Auto-detect:
```python
import shutil
BLENDER_PATH = shutil.which("blender")
if BLENDER_PATH is None:
    # Try common locations
    candidates = [
        "/usr/bin/blender",
        "/snap/bin/blender",
        Path.home() / "blender" / "blender",
        Path("C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"),
    ]
    for c in candidates:
        if Path(c).exists():
            BLENDER_PATH = str(c)
            break
```

### ExifTool path (`utils/video_utils.py` line 16)

```python
_EXIFTOOL_PATH = Path(__file__).parent.parent / "tools" / "exiftool" / "exiftool-13.50_64" / "exiftool.exe"
```

**Already has fallback** (line 118-120): if the bundled `.exe` doesn't exist, it tries system `exiftool`. On Linux, just install ExifTool via the system package manager and the fallback works:
```bash
sudo apt install libimage-exiftool-perl   # Debian/Ubuntu
# or
sudo dnf install perl-Image-ExifTool       # Fedora/RHEL
```

No code change needed — the fallback at line 120 already handles this.

### `fix_pose2sim.py` lines 43-44

```python
Path.home() / "AppData/Local/anaconda3/envs/Pose2Sim",
Path("C:/ProgramData/anaconda3/envs/Pose2Sim"),
```

**Fix:** Add Linux conda paths:
```python
Path.home() / "miniconda3" / "envs" / "Pose2Sim",
Path.home() / "anaconda3" / "envs" / "Pose2Sim",
Path.home() / "AppData/Local/anaconda3/envs/Pose2Sim",  # Windows
Path("C:/ProgramData/anaconda3/envs/Pose2Sim"),          # Windows
```

---

## 2. Linux Environment Setup

### Prerequisites
```bash
# NVIDIA driver (check with nvidia-smi)
# CUDA 12.1+ toolkit
sudo apt install libimage-exiftool-perl   # ExifTool
sudo snap install blender --classic        # Blender (or download from blender.org)
```

### mmpose environment (main pipeline)
```bash
conda create -n mmpose python=3.10 -y
conda activate mmpose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mmengine mmcv mmdet mmpose
pip install rtmpose3d
pip install --no-binary xtcocotools --no-build-isolation xtcocotools
pip install -r requirements.txt
```

### Pose2Sim environment (OpenSim IK subprocess)
```bash
conda create -n Pose2Sim python=3.12 -y
conda activate Pose2Sim
conda install -c opensim-org opensim=4.5.2
pip install pose2sim

# Fix Pose2Sim marker bug
python fix_pose2sim.py

conda activate mmpose  # Switch back
```

### MotionBERT checkpoint
```bash
# Download FT_MB_lite_MB_ft_h36m_global_lite from:
# https://github.com/Walter0807/MotionBERT/releases
# Place at:
mkdir -p models/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/
# Copy best_epoch.bin there
```

---

## 3. Test Files (low priority)

These files have hardcoded Windows paths but are development/debug scripts, not part of the main pipeline:

- `test_imports.py` — lines 5, 39, 62, 70
- `test_exif.py` — line 82
- `test_focal.py` — line 26
- `test_lean.py` — line 4
- `test_lean2.py` — lines 4, 15, 74
- `compare_inference.py` — line 36
- `compare_outputs.py` — lines 36, 38
- `server.py` — line 16 (hardcoded to a different Windows project path)

These can be updated to use `PROJECT_ROOT = Path(__file__).parent` or simply ignored if not needed on Linux.

---

## 4. Files to Modify (Summary)

| File | What to change |
|------|----------------|
| `utils/io_utils.py` | Add `find_pose2sim_python()` and `find_blender()` helpers |
| `run_hybrid_pipeline.py` | Line 928: use `find_pose2sim_python()` |
| `run_export.py` | Lines 243, 437, 553: use `find_pose2sim_python()` |
| `run_export.py` | Line 558: derive Pose2Sim OpenSim_Setup path dynamically |
| `run_export.py` | Line 641: use `find_blender()` |
| `fix_pose2sim.py` | Lines 43-44: add Linux conda paths |
| `utils/video_utils.py` | No change needed (ExifTool fallback already works) |

---

## 5. Running on Linux

```bash
conda activate mmpose
python run_inference.py --input videos/aitor_garden_walk.mp4
python run_hybrid_pipeline.py --input videos/aitor_garden_walk.mp4 --height 1.69 --correct-lean
```

Everything else (Path objects, OpenCV, numpy, scipy, MotionBERT, Blender headless mode) is already cross-platform.
