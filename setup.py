"""
Hunyuan3D 2 Mini — extension setup script.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe  — path to Modly's embedded Python (used to create the venv)
    ext_dir     — absolute path to this extension directory
    gpu_sm      — GPU compute capability as integer (e.g. 61 for Pascal, 86 for Ampere)

Example (manual test):
    python setup.py '{"python_exe":"C:/…/python.exe","ext_dir":"C:/…/hunyuan3d-2-mini","gpu_sm":86}'
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


ARM64_CU124_WHEELS = {
    "cp310": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp310-cp310-linux_aarch64.whl#sha256=d468d0eddc188aa3c1e417ec24ce615c48c0c3f592b0354d9d3b99837ef5faa6",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp310-cp310-linux_aarch64.whl#sha256=38765e53653f93e529e329755992ddbea81091aacedb61ed053f6a14efb289e5",
    },
    "cp311": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp311-cp311-linux_aarch64.whl#sha256=e080353c245b752cd84122e4656261eee6d4323a37cfb7d13e0fffd847bae1a3",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp311-cp311-linux_aarch64.whl#sha256=2c5350a08abe005a16c316ae961207a409d0e35df86240db5f77ec41345c82f3",
    },
    "cp312": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp312-cp312-linux_aarch64.whl#sha256=302041d457ee169fd925b53da283c13365c6de75c6bb3e84130774b10e2fbb39",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp312-cp312-linux_aarch64.whl#sha256=3e3289e53d0cb5d1b7f55b3f5912f46a08293c6791585ba2fc32c12cded9f9af",
    },
    "cp39": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp39-cp39-linux_aarch64.whl#sha256=012887a6190e562cb266d2210052c5deb5113f520a46dc2beaa57d76144a0e9b",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp39-cp39-linux_aarch64.whl#sha256=e25b4ac3c9eec3f789f1c5491331dfe236b5f06a1f406ea82fa59fed4fc6f71e",
    },
}


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def python_tag(venv: Path) -> str:
    is_win = platform.system() == "Windows"
    python_exe = venv / ("Scripts/python.exe" if is_win else "bin/python")
    return subprocess.check_output(
        [str(python_exe), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()


def setup(python_exe: str, ext_dir: Path, gpu_sm: int) -> None:
    venv = ext_dir / "venv"
    machine = platform.machine().lower()
    is_linux_arm64 = platform.system() == "Linux" and machine in {"aarch64", "arm64"}

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch — choose version based on GPU architecture
    # ------------------------------------------------------------------ #
    if is_linux_arm64 and gpu_sm >= 70:
        # Use direct wheel URLs on Linux ARM64 to avoid bad cache/CDN responses
        # while keeping the upstream-published sha256 integrity checks.
        py_tag = python_tag(venv)
        wheel_urls = ARM64_CU124_WHEELS.get(py_tag)
        if wheel_urls is None:
            raise RuntimeError(f"Unsupported Python version for Linux ARM64 PyTorch wheels: {py_tag}")
        print(f"[setup] GPU SM {gpu_sm}, Linux ARM64 -> PyTorch 2.5 + CUDA 12.4")
        print(f"[setup] Installing pinned ARM64 wheels for {py_tag} …")
        pip(
            venv,
            "install",
            "--no-cache-dir",
            wheel_urls["torch"],
            wheel_urls["torchvision"],
        )
    elif gpu_sm >= 70:
        # Volta and newer — PyTorch 2.6 + CUDA 12.4
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
        print("[setup] Installing PyTorch …")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    else:
        # Pascal (SM 6.x) — PyTorch 2.5 + CUDA 11.8 (last version with SM 6.1)
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")
        print("[setup] Installing PyTorch …")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies …")
    pip(venv, "install",
        "Pillow",
        "numpy",
        "trimesh",
        "pymeshlab",
        "opencv-python-headless",
        "huggingface_hub",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate",
        "einops",
        "scipy",
        "scikit-image",
    )

    # ------------------------------------------------------------------ #
    # rembg (background removal)
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg …")
    if is_linux_arm64:
        # rembg[gpu] pulls in onnxruntime-gpu, which has no linux_aarch64 wheel.
        pip(venv, "install", "rembg")
        pip(venv, "install", "onnxruntime")
    elif gpu_sm >= 70:
        pip(venv, "install", "rembg[gpu]")
    else:
        # onnxruntime-gpu has cuDNN FE issues on Pascal — use CPU provider
        pip(venv, "install", "rembg")
        pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Texture generation dependencies (optional — heavy)
    # Skipped here; will be installed on first texture request if needed.
    # Requires custom C++ extensions (custom_rasterizer, differentiable_renderer)
    # built via separate wheel distribution.
    # ------------------------------------------------------------------ #

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    # Accepts either JSON (from Electron) or positional args (for manual testing)
    # Positional: python setup.py <python_exe> <ext_dir> <gpu_sm>
    # JSON:       python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86}'
    if len(sys.argv) >= 4:
        setup(
            python_exe = sys.argv[1],
            ext_dir    = Path(sys.argv[2]),
            gpu_sm     = int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe = args["python_exe"],
            ext_dir    = Path(args["ext_dir"]),
            gpu_sm     = int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)
