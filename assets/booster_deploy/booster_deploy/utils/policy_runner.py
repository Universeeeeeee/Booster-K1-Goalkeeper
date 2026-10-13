"""Small policy inference adapters used by deployment policies.

The deployment code keeps observations and actions as torch tensors.  This
module adapts both TorchScript and CPU ONNX Runtime models to that same
interface, selected automatically from the checkpoint suffix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class TorchScriptRunner:
    """Run a TorchScript policy and return a tensor on the requested device."""

    def __init__(
        self,
        path: str,
        device: torch.device,
        use_actor_module: bool = False,
    ) -> None:
        model = torch.jit.load(path, map_location=device)
        model.to(device).eval()

        if use_actor_module:
            if not hasattr(model, "actor"):
                raise ValueError("Configured policy must expose an actor module")
            model = model.actor

        self.model = model

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            output = self.model(observation)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("TorchScript policy output must be a torch.Tensor")
        return output


class CpuOnnxRunner:
    """Run a single-input/single-output ONNX policy on the CPU.

    ONNX Runtime's CPU execution provider is deliberately used even if the
    deployment process was otherwise configured with a CUDA device.  The
    result is copied back to the policy device so the surrounding tensor
    post-processing remains unchanged.
    """

    def __init__(self, path: str, device: torch.device) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "ONNX policies require the 'onnxruntime' package"
            ) from exc

        self.device = device
        self.session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "Policy ONNX model must have exactly one input and one output"
            )
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.output_name = outputs[0].name
        self._input_array: np.ndarray | None = None
        output_shape = outputs[0].shape
        self._output_array: np.ndarray | None = None
        self._output_cpu_tensor: torch.Tensor | None = None
        self._output_tensor: torch.Tensor | None = None
        self._io_binding = None
        if all(isinstance(dim, int) for dim in output_shape):
            self._output_array = np.empty(tuple(output_shape), dtype=np.float32)
            self._output_cpu_tensor = torch.from_numpy(self._output_array)
            self._io_binding = self.session.io_binding()
            if device.type != "cpu":
                self._output_tensor = torch.empty(tuple(output_shape), dtype=torch.float32, device=device)

    def _prepare_input(self, source: torch.Tensor) -> torch.Tensor:
        """Match a flattened policy observation to an optional batch input."""
        expected_rank = len(self.input_shape)
        if source.ndim + 1 == expected_rank:
            # Most exported actors use [batch, observation] while the
            # locomotion policy historically passes a flat observation.
            source = source.unsqueeze(0)
        elif source.ndim == expected_rank + 1 and source.shape[0] == 1:
            source = source.squeeze(0)
        return source

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        source = observation.detach()
        if source.device.type != "cpu" or source.dtype != torch.float32:
            source = source.to(device="cpu", dtype=torch.float32)
        source = self._prepare_input(source).contiguous()
        if self._input_array is None or self._input_array.shape != tuple(source.shape):
            self._input_array = np.empty(tuple(source.shape), dtype=np.float32)
        np.copyto(self._input_array, source.numpy())
        if self._output_array is not None:
            io_binding = self._io_binding
            io_binding.bind_cpu_input(self.input_name, self._input_array)
            io_binding.bind_output(
                self.output_name,
                "cpu",
                0,
                np.float32,
                self._output_array.shape,
                self._output_array.ctypes.data,
            )
            self.session.run_with_iobinding(io_binding)
            output_array = self._output_array
        else:
            output = self.session.run(
                [self.output_name],
                {self.input_name: self._input_array},
            )[0]
            output_array = np.asarray(output, dtype=np.float32)
        if self._output_array is None and (
            self._output_cpu_tensor is None
            or tuple(self._output_cpu_tensor.shape) != tuple(output_array.shape)
        ):
            # Dynamic output dimensions cannot use a pre-bound output buffer.
            self._output_cpu_tensor = torch.from_numpy(output_array)
        if self.device.type == "cpu":
            return self._output_cpu_tensor
        if self._output_tensor is None or tuple(self._output_tensor.shape) != tuple(self._output_cpu_tensor.shape):
            self._output_tensor = torch.empty(
                tuple(self._output_cpu_tensor.shape),
                dtype=torch.float32,
                device=self.device,
            )
        self._output_tensor.copy_(self._output_cpu_tensor)
        return self._output_tensor


def create_policy_runner(
    path: str,
    device: torch.device,
    use_actor_module: bool = False,
) -> Any:
    """Create a policy runner based on ``path``'s extension.

    ``.pt``/``.jit``/``.torchscript`` files retain the existing TorchScript
    behavior.  ``.onnx`` files use the CPU ONNX Runtime execution provider.
    """

    suffix = Path(path).suffix.lower()
    if suffix == ".onnx":
        # ONNX exports contain the executable graph directly, so there is no
        # separate ``actor`` attribute to select.  Keep the flag as a no-op
        # for compatibility with locomotion configurations that use it for
        # TorchScript checkpoints.
        return CpuOnnxRunner(path, device)
    if suffix in {".pt", ".jit", ".torchscript"}:
        return TorchScriptRunner(path, device, use_actor_module)
    raise ValueError(
        f"Unsupported policy format '{suffix}'. "
        "Use a .pt, .jit, .torchscript, or .onnx checkpoint."
    )
