"""Export a trained MaskablePPO policy checkpoint to ONNX logits model.

Example:
    python export_ppo_policy_onnx.py 
        --model-path tmp/rwa_ppo/rwa_model_1200000_steps.zip 
        --output-path tmp/rwa_ppo/rwa_policy_logits.onnx 
        --verify

Notes:
- This exporter writes an ONNX model that outputs action logits.
- Action masking is NOT baked into ONNX; keep masking in the runtime loop.
- Supports Box and Dict observation spaces.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO


class _PolicyLogitsWrapper(torch.nn.Module):
    """Minimal wrapper exposing policy logits for ONNX export."""

    def __init__(self, policy: torch.nn.Module, obs_keys: tuple[str, ...] | None = None):
        super().__init__()
        self.policy = policy
        self.obs_keys = obs_keys

    def forward(self, *obs_inputs: torch.Tensor) -> torch.Tensor:
        # Same actor forward path used by SB3 policies.
        if self.obs_keys is None:
            if len(obs_inputs) != 1:
                raise ValueError(f"Expected 1 tensor input, got {len(obs_inputs)}")
            obs: torch.Tensor | dict[str, torch.Tensor] = obs_inputs[0]
        else:
            if len(obs_inputs) != len(self.obs_keys):
                raise ValueError(
                    f"Expected {len(self.obs_keys)} tensor inputs, got {len(obs_inputs)}"
                )
            obs = {k: v for k, v in zip(self.obs_keys, obs_inputs)}

        features = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        return self.policy.action_net(latent_pi)


def _build_dummy_obs(
    obs_space: spaces.Space,
) -> tuple[tuple[torch.Tensor, ...], dict[str, np.ndarray], list[str], dict[str, dict[int, str]], tuple[str, ...] | None]:
    if isinstance(obs_space, spaces.Box):
        if obs_space.shape is None:
            raise ValueError("Observation space shape is None; cannot export ONNX input shape.")
        dummy_obs_np = np.zeros((1, *obs_space.shape), dtype=np.float32)
        return (
            (torch.from_numpy(dummy_obs_np),),
            {"obs": dummy_obs_np},
            ["obs"],
            {"obs": {0: "batch"}},
            None,
        )

    if isinstance(obs_space, spaces.Dict):
        obs_keys = tuple(obs_space.spaces.keys())
        if not obs_keys:
            raise ValueError("Observation Dict space is empty; cannot export ONNX input shape.")

        torch_inputs: list[torch.Tensor] = []
        ort_inputs: dict[str, np.ndarray] = {}
        input_names: list[str] = []
        dynamic_axes: dict[str, dict[int, str]] = {}

        for key in obs_keys:
            subspace = obs_space.spaces[key]

            if isinstance(subspace, spaces.Box):
                if subspace.shape is None:
                    raise ValueError(f"Observation subspace '{key}' has shape None.")
                arr = np.zeros((1, *subspace.shape), dtype=np.float32)
            elif isinstance(subspace, spaces.MultiBinary):
                if subspace.shape is None:
                    raise ValueError(f"Observation subspace '{key}' has shape None.")
                arr = np.zeros((1, *subspace.shape), dtype=np.int64)
            elif isinstance(subspace, spaces.MultiDiscrete):
                arr = np.zeros((1, len(subspace.nvec)), dtype=np.int64)
            elif isinstance(subspace, spaces.Discrete):
                # SB3 preprocess expects shape (batch,) for Discrete observations.
                arr = np.zeros((1,), dtype=np.int64)
            else:
                raise TypeError(
                    f"Unsupported subspace type in Dict observation for key '{key}': {type(subspace)!r}. "
                    "Supported Dict subspaces: Box, MultiBinary, MultiDiscrete, Discrete."
                )

            name = f"obs_{key}"
            torch_inputs.append(torch.from_numpy(arr))
            ort_inputs[name] = arr
            input_names.append(name)
            dynamic_axes[name] = {0: "batch"}

        return (tuple(torch_inputs), ort_inputs, input_names, dynamic_axes, obs_keys)

    raise TypeError(
        f"Unsupported observation space for ONNX export: {type(obs_space)!r}. "
        "This script currently supports gymnasium.spaces.Box and gymnasium.spaces.Dict."
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MaskablePPO checkpoint to ONNX (logits output)")
    parser.add_argument("--model-path", required=True, help="Path to SB3 MaskablePPO checkpoint (.zip)")
    parser.add_argument("--output-path", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--verify", action="store_true", help="Run a post-export ONNXRuntime shape check")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = MaskablePPO.load(str(model_path), device="cpu")
    model.policy.eval()

    export_inputs, ort_inputs, input_names, dynamic_axes, obs_keys = _build_dummy_obs(
        model.observation_space
    )

    wrapper = _PolicyLogitsWrapper(model.policy, obs_keys=obs_keys).cpu().eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            export_inputs,
            str(output_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={**dynamic_axes, "logits": {0: "batch"}},
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    print(f"Exported ONNX model: {output_path}")
    if obs_keys is None:
        print(f"Input shape: {tuple(next(iter(ort_inputs.values())).shape)}")
    else:
        shape_map = {name: tuple(arr.shape) for name, arr in ort_inputs.items()}
        print(f"Input shapes: {shape_map}")

    if args.verify:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - optional dependency
            print("[WARN] onnxruntime is not installed, skipping verification.")
            print(f"       Reason: {exc}")
            return 0

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_out = session.run(None, ort_inputs)
        if not ort_out:
            raise RuntimeError("ONNXRuntime returned no outputs.")
        print(f"Verified ONNX output shape: {tuple(ort_out[0].shape)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
