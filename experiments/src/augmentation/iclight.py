"""Small, version-tolerant client for a local IC-Light Gradio server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ICLightRequest:
    """Parameters sent to the IC-Light background-conditioned demo."""

    foreground: Path
    background: Path
    seed: int
    prompt: str = "aerial view of a parked car on an asphalt road, daylight, realistic photograph"
    negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality, artifacts"
    steps: int = 20
    cfg: float = 7.0
    image_size: int = 512
    image_height: int | None = None
    highres_scale: float = 1.5
    highres_denoise: float = 0.5


class ICLightClient:
    """Call a local Gradio IC-Light server without hard-coding an endpoint name."""

    def __init__(self, url: str, api_name: str | None = None) -> None:
        """Create a client for a running local server.

        Args:
            url: Server URL, normally ``http://127.0.0.1:7860``.
            api_name: Optional explicit Gradio API name. If omitted, an endpoint
                containing ``relight`` is discovered from the server config.
        """
        try:
            from gradio_client import Client
        except ImportError as exc:  # pragma: no cover - Colab-only dependency
            raise RuntimeError(
                "gradio_client is required. Install it in the isolated IC-Light "
                "environment before rendering."
            ) from exc
        self._client = Client(url)
        self.api_name = api_name
        self.fn_index: int | None = None
        if self.api_name is None:
            try:
                self.api_name = self._discover_api_name()
            except RuntimeError:
                # Gradio 3 demos may omit API names. The example loader is
                # callback 1; IC-Light's 14-argument relight button is 2.
                self.fn_index = 2

    def _discover_api_name(self) -> str:
        """Return the exposed relighting endpoint or raise an actionable error."""
        endpoints = getattr(self._client, "endpoints", [])
        names = [
            str(getattr(endpoint, "api_name", ""))
            for endpoint in endpoints
            if getattr(endpoint, "api_name", None)
        ]
        matching = [name for name in names if "relight" in name.lower()]
        if matching:
            return matching[0]
        raise RuntimeError(
            "IC-Light did not expose a relighting API endpoint. Available endpoints: "
            f"{names}. Run `python -m src.augmentation.iclight --describe --url ...` "
            "and pass --iclight-api-name explicitly."
        )

    def _endpoint_parameter_names(self) -> set[str]:
        """Return parameter names advertised by the selected Gradio endpoint."""
        for endpoint in getattr(self._client, "endpoints", []):
            if getattr(endpoint, "api_name", None) == self.api_name:
                return {
                    str(getattr(parameter, "parameter_name", ""))
                    for parameter in getattr(endpoint, "parameters", [])
                }
        return set()

    def relight(self, request: ICLightRequest) -> Path:
        """Submit one background-conditioned relighting request.

        Args:
            request: Foreground and background full-frame images plus diffusion
                parameters.

        Returns:
            Path of the generated image returned by Gradio.
        """
        values: dict[str, Any] = {
            "input_fg": str(request.foreground),
            "input_bg": str(request.background),
            "prompt": request.prompt,
            "image_width": request.image_size,
            "image_height": request.image_height or request.image_size,
            "num_samples": 1,
            "seed": request.seed,
            "steps": request.steps,
            "a_prompt": "best quality",
            "n_prompt": request.negative_prompt,
            "cfg": request.cfg,
            "highres_scale": request.highres_scale,
            "highres_denoise": request.highres_denoise,
            "lowres_denoise": 0.5,
            "bg_source": "Use Background Image",
        }
        expected = self._endpoint_parameter_names()
        if expected:
            values = {name: value for name, value in values.items() if name in expected}
        if self.fn_index is not None:
            values["input_fg"] = str(request.foreground)
            values["input_bg"] = str(request.background)
            api_kwargs: dict[str, Any] = {"fn_index": self.fn_index}
        else:
            api_kwargs = {"api_name": self.api_name}
        ordered = [
            values[name]
            for name in (
                "input_fg", "input_bg", "prompt", "image_width", "image_height",
                "num_samples", "seed", "steps", "a_prompt", "n_prompt", "cfg",
                "highres_scale", "highres_denoise", "bg_source",
            )
        ]
        result: Any = self._client.predict(*ordered, **api_kwargs)
        payload = result[0] if isinstance(result, (tuple, list)) else result
        if isinstance(payload, dict):
            image = payload.get("image") or payload.get("name")
        else:
            image = payload
        if not image:
            raise RuntimeError(f"IC-Light returned no image payload: {payload!r}")
        return Path(str(image))


def _main() -> None:
    """Print the API names exposed by a local IC-Light server."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect local IC-Light Gradio APIs")
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    client = ICLightClient(args.url) if not args.describe else None
    if client:
        print(client.api_name)
        return
    from gradio_client import Client

    raw_client = Client(args.url)
    print([getattr(endpoint, "api_name", None) for endpoint in raw_client.endpoints])


if __name__ == "__main__":  # pragma: no cover - command-line wrapper
    _main()
