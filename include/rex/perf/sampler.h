#pragma once

#include <cstdint>
#include <string_view>

// Statistical CPU profiler for one thread (PS5 only; no-ops elsewhere). A
// sampler thread interrupts the target with SIGPROF at a fixed rate; the
// handler records the interrupted instruction pointer and the first few stack
// words that look like return addresses. Samples are appended to a file as
// little-endian uint64 records: rip, then kSampleStackWords candidates (0 when
// absent). Symbolize them offline against the linked ELF.
namespace rex::perf::sampler {

inline constexpr int kSampleStackWords = 8;

// Starts sampling the calling thread. Ignored if already sampling.
void StartForCurrentThread(uint32_t hz, std::string_view path);
// Appends samples taken since the last flush to the file. Call from any
// thread except from the signal handler.
void Flush();

}  // namespace rex::perf::sampler
