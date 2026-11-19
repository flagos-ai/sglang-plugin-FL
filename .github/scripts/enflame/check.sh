#!/bin/bash
set -euo pipefail
python3 - <<'EOF'
import torch
import torch_gcu
if not torch.gcu.is_available():
    raise RuntimeError("GCU is unavailable")
count = torch.gcu.device_count()
print("Visible GCU count:", count)
if count < 4:
    raise RuntimeError("TP4 requires at least four GCUs")
x = torch.ones(4, device="gcu:0")
torch.testing.assert_close((x + x).cpu(), torch.full((4,), 2.0))
torch.gcu.synchronize()
EOF
