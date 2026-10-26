// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CoreX image ABI: advanced outputs are unavailable on older host streamMgr. */
typedef int (*capture_info_v3)(void *, int *, unsigned long long *, void *,
                             void *, void *, void *);
typedef int (*capture_info_v2)(void *, int *, unsigned long long *, void *,
                             void *, void *);

int ixStreamGetCaptureInfo(void *stream, int *status, unsigned long long *id,
                          void *graph, void *dependencies, void *edge_data,
                          void *num_dependencies) {
    capture_info_v3 original =
        (capture_info_v3)dlsym(RTLD_NEXT, "ixStreamGetCaptureInfo");
    if (original == NULL) {
        static atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!atomic_flag_test_and_set(&warned)) {
            fputs("Iluvatar capture-info compatibility: original symbol missing; "
                  "returning native not-supported status 801\n", stderr);
        }
        return 801;
    }
    int result = original(stream, status, id, graph, dependencies, edge_data,
                          num_dependencies);
    if (result != 801 || status == NULL || id == NULL || graph != NULL ||
        dependencies != NULL || edge_data != NULL || num_dependencies != NULL) {
        return result;
    }
    const char *vendor = getenv("GEMS_VENDOR");
    const char *enabled = getenv("SGLANG_ILUVATAR_CAPTURE_INFO_COMPAT");
    if (vendor == NULL || strcmp(vendor, "iluvatar") != 0 || enabled == NULL ||
        strcmp(enabled, "1") != 0) {
        return result;
    }
    capture_info_v2 compatible =
        (capture_info_v2)dlsym(RTLD_NEXT, "ixStreamGetCaptureInfo_v2");
    if (compatible == NULL) {
        static atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!atomic_flag_test_and_set(&warned)) {
            fputs("Iluvatar capture-info compatibility: native v2 symbol missing; "
                  "returning native not-supported status 801\n", stderr);
        }
        return result;
    }
    static atomic_flag announced = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&announced)) {
        fputs("Iluvatar capture-info status/id query uses native v2 compatibility\n",
              stderr);
    }
    return compatible(stream, status, id, NULL, NULL, NULL);
}
