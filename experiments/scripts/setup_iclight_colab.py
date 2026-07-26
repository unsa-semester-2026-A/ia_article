#!/usr/bin/env python3
"""Create an isolated, pinned IC-Light environment in Google Colab.

Run this script from the orchestration notebook. It intentionally uses the
preinstalled CUDA Torch from Colab through ``--system-site-packages`` rather than
downloading a second Torch wheel with uv.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ICLIGHT_REPOSITORY = "https://github.com/lllyasviel/IC-Light.git"
COMPATIBILITY_PINS = [
    "huggingface_hub<0.26",
    "peft<0.11",
    "fastapi==0.104.1",
    "starlette==0.27.0",
    "pydantic==2.4.2",
    "jinja2==3.1.2",
    "gradio_client",
]
RELIGHT_CLICK = "relight_button.click(fn=process_relight, inputs=ips, outputs=[result_gallery])"
BLOCK_WITH_QUEUE = "block = gr.Blocks().queue()"
BLOCK_WITHOUT_QUEUE = "block = gr.Blocks()"
BLOCK_LAUNCH = "block.launch(server_name='0.0.0.0')"
BLOCK_LAUNCH_WITH_ERRORS = "block.launch(server_name='0.0.0.0', show_error=True)"
RELIGHT_CLICK_WITH_API = (
    "relight_button.click(fn=_logged_process_relight, inputs=ips, outputs=[result_gallery])\n"
    "    _api_relight_button = gr.Button(visible=False)\n"
    "    _api_relight_button.click(\n"
    "        fn=_logged_process_relight_image, inputs=ips,\n"
    '        outputs=[dummy_image_for_outputs], api_name="process_relight"\n'
    "    )"
)
RELIGHT_TRACE_WRAPPER = (
    "def _logged_process_relight(*args, **kwargs):\n"
    "        try:\n"
    "            print('[ICLight] callback entered:', [\n"
    "                (type(item).__name__, getattr(item, 'shape', None), getattr(item, 'dtype', None))\n"
    "                for item in args\n"
    "            ], flush=True)\n"
    "            output = process_relight(*args, **kwargs)\n"
    "            print('[ICLight] callback completed:', [\n"
    "                (type(item).__name__, getattr(item, 'shape', None), getattr(item, 'dtype', None))\n"
    "                for item in output\n"
    "            ], flush=True)\n"
    "            return output\n"
    "        except Exception:\n"
    "            import traceback\n"
    "            traceback.print_exc()\n"
    "            raise\n"
    "\n"
    "    def _logged_process_relight_image(*args, **kwargs):\n"
    "        image = _logged_process_relight(*args, **kwargs)[0]\n"
    "        print('[ICLight] API image selected:', image.shape, image.dtype, flush=True)\n"
    "        return image\n"
    "\n"
    "    "
)


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run one checked command and stream failures to the notebook."""
    subprocess.run(command, cwd=cwd, check=True)


def expose_relight_api(root: Path) -> None:
    """Expose relighting and retain callback tracebacks in the server log."""
    demo = root / "gradio_demo_bg.py"
    source = demo.read_text()
    if BLOCK_WITH_QUEUE in source:
        # Kaggle's forced share tunnel can drop Gradio 3.41 queue WebSockets
        # after a successful callback. This pipeline is intentionally
        # sequential, so the synchronous HTTP endpoint is more reliable.
        source = source.replace(BLOCK_WITH_QUEUE, BLOCK_WITHOUT_QUEUE, 1)
    if BLOCK_LAUNCH in source:
        source = source.replace(BLOCK_LAUNCH, BLOCK_LAUNCH_WITH_ERRORS, 1)
    if RELIGHT_CLICK_WITH_API in source:
        demo.write_text(source)
        return
    if RELIGHT_CLICK not in source:
        raise RuntimeError("IC-Light demo layout changed; relight callback was not found")
    replacement = RELIGHT_TRACE_WRAPPER + RELIGHT_CLICK_WITH_API
    demo.write_text(source.replace(RELIGHT_CLICK, replacement, 1))


def main() -> None:
    """Clone IC-Light once and install its pinned environment with uv pip."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/content/IC-Light")
    args = parser.parse_args()
    root = Path(args.root)
    # Kaggle exposes TensorFlow through system site packages.  The legacy
    # IC-Light/Transformers pins use a different protobuf version, so asking
    # Transformers to import its optional TensorFlow backend crashes before
    # the PyTorch-only IC-Light server can start.
    os.environ["USE_TF"] = "0"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    if shutil.which("uv") is None:
        installer = subprocess.run(
            ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(["sh", "-c", installer.stdout], check=True)
        os.environ["PATH"] = f"{Path.home() / '.local/bin'}:{os.environ['PATH']}"
    if not root.exists():
        run(["git", "clone", "--depth", "1", ICLIGHT_REPOSITORY, str(root)])
    expose_relight_api(root)
    python = root / ".venv" / "bin" / "python"
    if not python.exists():
        run(["uv", "venv", "--system-site-packages", str(root / ".venv")])
    run(
        ["uv", "pip", "install", "--python", str(python), "-r", "requirements.txt"],
        root,
    )
    run(["uv", "pip", "install", "--python", str(python), *COMPATIBILITY_PINS], root)
    run(
        [
            str(python),
            "-c",
            "import diffusers, huggingface_hub, peft, gradio_client; print('IC-Light environment OK')",
        ],
        root,
    )


if __name__ == "__main__":
    main()
