// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <musa_runtime_api.h>

#include <cerrno>
#include <cstdint>
#include <sys/eventfd.h>

namespace {

void notify_eventfd(void* user_data) {
  const int event_fd = static_cast<int>(reinterpret_cast<intptr_t>(user_data));
  while (eventfd_write(event_fd, 1) != 0 && errno == EINTR) {
  }
}

}  // namespace

extern "C" int enqueue_musa_eventfd_completion(uintptr_t stream_ptr,
                                                int event_fd) {
  return static_cast<int>(musaLaunchHostFunc(
      reinterpret_cast<musaStream_t>(stream_ptr), notify_eventfd,
      reinterpret_cast<void*>(static_cast<intptr_t>(event_fd))));
}
