# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class APIFuture(Generic[T]):
    """A synchronous future shim used to mirror Tinker-style client calls.

    The prototype executes work immediately. Keeping the future-shaped return
    value makes it easier to replace the implementation with a queued worker
    later without changing user training loops.
    """

    def __init__(self, value: T):
        self._value = value

    def result(self) -> T:
        """Return the completed operation result."""
        return self._value

    async def result_async(self) -> T:
        """Return the completed operation result from async code."""
        return self._value
