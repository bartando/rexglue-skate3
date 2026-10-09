#pragma once

#include <cstdint>
#include <string_view>

// Per-thread CPU accounting for the PERF diagnostics. Only implemented on PS5,
// where whole-process CPU percentages hide a single saturated thread.
namespace rex::perf::thread_cpu {

void RegisterCurrentThread();
void UnregisterCurrentThread();
// `thread` is the native thread handle (pthread_t) as an integer.
void SetThreadName(uintptr_t thread, std::string_view name);
// Logs the busiest threads since the previous call; zero window only primes.
void Log(double window_seconds);

}  // namespace rex::perf::thread_cpu
