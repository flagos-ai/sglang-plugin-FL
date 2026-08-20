# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Backend abstract base class.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    """
    Abstract base class for operator backends.

    Each backend provides implementations for a set of operators.
    Backends should implement is_available() to indicate whether
    the backend can be used in the current environment.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name string."""
        pass

    @property
    def vendor(self) -> Optional[str]:
        """Vendor name (for VENDOR type backends)."""
        return None
