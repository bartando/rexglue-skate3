#include <rex/perf/hitch_diagnostics.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <numeric>
#include <thread>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>

#if REX_PLATFORM_MAC
#include <sys/resource.h>
#endif

REXCVAR_DEFINE_BOOL(frame_hitch_diagnostics, false, "Diagnostics",
                    "Log bounded guest-frame hitches and timing summaries without draw captures")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex::perf::hitch {
namespace {

struct OperationCounters {
  std::atomic<uint64_t> count{0}, microseconds{0}, maximum_us{0};
};
std::array<OperationCounters, 3> operations;
std::atomic<uint64_t> stale_recoveries{0};

struct Sample {
  uint64_t count = 0, microseconds = 0, maximum_us = 0;
};

Sample TakeSample(OperationCounters& counters) {
  // Concurrent work is attributed when it completes, so these are interval
  // observations, not proof that an operation caused the hitch. Worker times
  // can overlap each other and exceed the wall-clock frame interval.
  return {counters.count.exchange(0, std::memory_order_relaxed),
          counters.microseconds.exchange(0, std::memory_order_relaxed),
          counters.maximum_us.exchange(0, std::memory_order_relaxed)};
}

void LogHostContext(double elapsed_seconds) {
#if REX_PLATFORM_MAC
  rusage usage{};
  static double previous_cpu_seconds = 0.0;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    const double cpu_seconds = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
                               usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
    if (elapsed_seconds == 0.0) {
      previous_cpu_seconds = cpu_seconds;
      return;
    }
    double load[3]{};
    const bool load_available = getloadavg(load, 3) == 3;
    REXLOG_INFO("PERF host: game_cpu_pct={:.1f} logical_cpus={} load_available={} "
                "load_1m={:.2f} load_5m={:.2f} peak_rss_mb={:.1f}",
                100.0 * (cpu_seconds - previous_cpu_seconds) / elapsed_seconds,
                std::thread::hardware_concurrency(), load_available, load[0], load[1],
                double(usage.ru_maxrss) / (1024.0 * 1024.0));
    previous_cpu_seconds = cpu_seconds;
  }
#endif
}

}  // namespace

bool Enabled() { return REXCVAR_GET(frame_hitch_diagnostics); }

void RecordOperation(Operation operation, uint64_t microseconds) {
  auto& counters = operations[size_t(operation)];
  counters.count.fetch_add(1, std::memory_order_relaxed);
  counters.microseconds.fetch_add(microseconds, std::memory_order_relaxed);
  uint64_t maximum = counters.maximum_us.load(std::memory_order_relaxed);
  while (maximum < microseconds &&
         !counters.maximum_us.compare_exchange_weak(maximum, microseconds,
                                                    std::memory_order_relaxed)) {}
}

void RecordStaleProtectionRecovery() {
  if (Enabled()) stale_recoveries.fetch_add(1, std::memory_order_relaxed);
}

void RecordSwap() {
  if (!Enabled()) return;
  using Clock = std::chrono::steady_clock;
  static const auto session_start = Clock::now();
  static auto previous = session_start;
  static auto window_start = session_start;
  static std::array<double, 300> frame_times{};
  static std::array<Sample, 3> totals{};
  static uint64_t frame = 0, window_faults = 0;
  static size_t count = 0, hitches = 0, severe_hitches = 0;
  static size_t logged_hitches = 0, logged_severe = 0;
  const auto now = Clock::now();
  if (frame++ == 0) {
    // Separate work before the first swap from the first measured interval.
    for (size_t i = 0; i < operations.size(); ++i) {
      const Sample startup = TakeSample(operations[i]);
      REXLOG_INFO("PERF startup: operation={} count={} total_ms={:.3f} max_ms={:.3f}",
                  i == 0 ? "pipeline" : i == 1 ? "gpu_wait" : "file_read",
                  startup.count, startup.microseconds / 1000.0, startup.maximum_us / 1000.0);
    }
    stale_recoveries.exchange(0, std::memory_order_relaxed);
    LogHostContext(0.0);
    previous = window_start = now;
    REXLOG_INFO("PERF session: guest swap intervals; hitch >25ms, severe >100ms; "
                "summary every 300 intervals; max 8 hitch + 4 severe entries/window. "
                "Operations attributed on completion; concurrent times may overlap. "
                "Host CPU uses 100% per core.");
    return;
  }
  const double ms = std::chrono::duration<double, std::milli>(now - previous).count();
  previous = now;
  std::array<Sample, 3> samples{};
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i] = TakeSample(operations[i]);
    totals[i].count += samples[i].count;
    totals[i].microseconds += samples[i].microseconds;
    totals[i].maximum_us = std::max(totals[i].maximum_us, samples[i].maximum_us);
  }
  const uint64_t faults = stale_recoveries.exchange(0, std::memory_order_relaxed);
  window_faults += faults;
  frame_times[count++] = ms;
  if (ms > 25.0) {
    ++hitches;
    if (ms > 100.0) ++severe_hitches;
    const bool log = ms > 100.0 ? logged_severe++ < 4 : logged_hitches++ < 8;
    if (log) {
      REXLOG_INFO("PERF hitch: frame={} elapsed_s={:.3f} frame_ms={:.3f} "
                  "pipeline_n={} pipeline_ms={:.3f} gpu_wait_ms={:.3f} "
                  "read_n={} read_ms={:.3f} read_max_ms={:.3f} stale_recoveries={}",
                  frame, std::chrono::duration<double>(now - session_start).count(), ms,
                  samples[0].count, samples[0].microseconds / 1000.0,
                  samples[1].microseconds / 1000.0, samples[2].count,
                  samples[2].microseconds / 1000.0, samples[2].maximum_us / 1000.0, faults);
    }
  }
  if (count != frame_times.size()) return;
  const double elapsed = std::chrono::duration<double>(now - window_start).count();
  const double average = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / count;
  std::sort(frame_times.begin(), frame_times.end());
  REXLOG_INFO("PERF summary: frame={} elapsed_s={:.3f} window_s={:.3f} fps={:.2f} "
              "avg_ms={:.3f} p95_ms={:.3f} p99_ms={:.3f} max_ms={:.3f} "
              "hitches={} severe={} suppressed={} pipeline_n={} pipeline_ms={:.3f} "
              "pipeline_max_ms={:.3f} gpu_wait_ms={:.3f} gpu_wait_max_ms={:.3f} "
              "read_n={} read_ms={:.3f} read_max_ms={:.3f} stale_recoveries={}",
              frame, std::chrono::duration<double>(now - session_start).count(), elapsed,
              count / elapsed, average, frame_times[284], frame_times[296], frame_times.back(),
              hitches, severe_hitches,
              logged_hitches - std::min(logged_hitches, size_t(8)) +
                  logged_severe - std::min(logged_severe, size_t(4)),
              totals[0].count, totals[0].microseconds / 1000.0, totals[0].maximum_us / 1000.0,
              totals[1].microseconds / 1000.0, totals[1].maximum_us / 1000.0,
              totals[2].count, totals[2].microseconds / 1000.0, totals[2].maximum_us / 1000.0,
              window_faults);
  LogHostContext(elapsed);
  count = hitches = severe_hitches = logged_hitches = logged_severe = 0;
  window_faults = 0;
  totals = {};
  window_start = now;
}

}  // namespace rex::perf::hitch
