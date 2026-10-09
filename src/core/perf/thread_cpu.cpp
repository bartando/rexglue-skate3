#include <rex/perf/thread_cpu.h>

#include <rex/platform.h>

#if REX_PLATFORM_PS5
#include <pthread.h>
#include <pthread_np.h>
#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <rex/logging.h>
#endif

namespace rex::perf::thread_cpu {

#if REX_PLATFORM_PS5
namespace {

struct Entry {
  uintptr_t thread;
  long tid;
  clockid_t clock;
  std::string name;
  int64_t last_ns;
};

std::mutex mutex;
std::vector<Entry> entries;
int64_t last_process_ns = -1;

int64_t ClockNs(clockid_t clock) {
  timespec time{};
  if (clock_gettime(clock, &time) != 0) return -1;
  return int64_t(time.tv_sec) * 1000000000 + time.tv_nsec;
}

int64_t ProcessCpuNs() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
  return (int64_t(usage.ru_utime.tv_sec) + usage.ru_stime.tv_sec) * 1000000000 +
         (int64_t(usage.ru_utime.tv_usec) + usage.ru_stime.tv_usec) * 1000;
}

}  // namespace

void RegisterCurrentThread() {
  const long tid = pthread_getthreadid_np();
  // libkernel's pthread_getcpuclockid hands back a clock that always reads the
  // calling thread, so build FreeBSD's per-thread CPU clock id (CPUCLOCK_BIT |
  // tid) directly; the kernel resolves it to that thread from any caller.
  const clockid_t clock = clockid_t(0x80000000u | uint32_t(tid));
  static std::atomic<bool> reported = false;
  if (ClockNs(clock) < 0) {
    if (!reported.exchange(true)) {
      REXLOG_WARN("PERF threads: thread CPU clock {:#x} unavailable", uint32_t(clock));
    }
    return;
  }
  const uintptr_t thread = uintptr_t(pthread_self());
  std::lock_guard lock(mutex);
  for (auto& entry : entries) {
    if (entry.thread == thread) {
      entry = {thread, tid, clock, std::move(entry.name), ClockNs(clock)};
      return;
    }
  }
  entries.push_back({thread, tid, clock, {}, ClockNs(clock)});
}

void UnregisterCurrentThread() {
  const uintptr_t thread = uintptr_t(pthread_self());
  std::lock_guard lock(mutex);
  std::erase_if(entries, [thread](const Entry& entry) { return entry.thread == thread; });
}

void SetThreadName(uintptr_t thread, std::string_view name) {
  std::lock_guard lock(mutex);
  for (auto& entry : entries) {
    if (entry.thread == thread) {
      entry.name = name;
      return;
    }
  }
}

void Log(double window_seconds) {
  struct Usage {
    double percent;
    const Entry* entry;
  };
  std::vector<Usage> usage;
  double registered = 0.0;
  std::lock_guard lock(mutex);
  for (auto& entry : entries) {
    const int64_t now = ClockNs(entry.clock);
    if (now < 0) continue;
    if (window_seconds > 0.0 && entry.last_ns >= 0) {
      const double percent = 100.0 * double(now - entry.last_ns) / (window_seconds * 1e9);
      registered += percent;
      usage.push_back({percent, &entry});
    }
    entry.last_ns = now;
  }
  const int64_t process_now = ProcessCpuNs();
  const int64_t process_before = last_process_ns;
  last_process_ns = process_now;
  if (window_seconds <= 0.0) return;

  std::sort(usage.begin(), usage.end(),
            [](const Usage& a, const Usage& b) { return a.percent > b.percent; });
  std::string top;
  for (size_t i = 0; i < usage.size() && i < 10 && usage[i].percent >= 1.0; ++i) {
    const Entry& entry = *usage[i].entry;
    fmt::format_to(std::back_inserter(top), "{}{}[{}]={:.0f}%", top.empty() ? "" : " ",
                   entry.name.empty() ? "unnamed" : entry.name, entry.tid, usage[i].percent);
  }
  const double process = process_now >= 0 && process_before >= 0
                             ? 100.0 * double(process_now - process_before) / (window_seconds * 1e9)
                             : -1.0;
  // 100% = one full core. process minus registered is CPU spent on threads
  // this registry never saw (process main thread, driver threads).
  REXLOG_INFO("PERF threads: process_pct={:.0f} registered_pct={:.0f} threads={} top: {}",
              process, registered, usage.size(), top);
}

#else

void RegisterCurrentThread() {}
void UnregisterCurrentThread() {}
void SetThreadName(uintptr_t, std::string_view) {}
void Log(double) {}

#endif

}  // namespace rex::perf::thread_cpu
