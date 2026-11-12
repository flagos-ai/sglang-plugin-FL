"""Run installed native SGLang NPU wheel kernels against CPU references."""

import torch
import torch_npu  # noqa: F401 - registers the NPU device with torch.
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu
from sgl_kernel_npu.norm.l1_norm import l1_norm


def check_l1_norm(device: str) -> None:
    inputs = torch.arange(1, 257, dtype=torch.float32).reshape(4, 64)
    expected = inputs / inputs.sum(dim=-1, keepdim=True)
    result = l1_norm(inputs.to(device))
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), expected, atol=2e-5, rtol=2e-5)
    print("native l1_norm passed", flush=True)


def check_solve_tril(device: str) -> None:
    for size in (16, 32, 64):
        indices = torch.arange(size * size, dtype=torch.float32).reshape(size, size)
        matrix = torch.tril((indices.remainder(17) - 8) * 0.001, diagonal=-1)
        identity = torch.eye(size)
        expected = torch.linalg.solve_triangular(identity + matrix, identity, upper=False)
        result = solve_tril_npu(matrix.reshape(1, size, 1, size).to(device))
        torch.npu.synchronize()
        torch.testing.assert_close(
            result.reshape(size, size).cpu(), expected, atol=2e-4, rtol=2e-4
        )
        print(f"native solve_tril size={size} passed", flush=True)


def main() -> None:
    device = "npu:0"
    torch.npu.set_device(device)
    print(f"device={torch.npu.get_device_name()}", flush=True)
    check_l1_norm(device)
    check_solve_tril(device)


if __name__ == "__main__":
    main()
