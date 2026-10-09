#include <rex/perf/sampler.h>

#include <rex/platform.h>

#if REX_PLATFORM_PS5
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>

#include <rex/logging.h>
#endif

namespace rex::perf::sampler {

#if REX_PLATFORM_PS5
namespace {

struct Sample {
  uint64_t rip;
  uint64_t stack[kSampleStackWords];
};

constexpr size_t kRingSize = 1 << 16;
Sample ring[kRingSize];
std::atomic<uint64_t> written{0};
uint64_t flushed = 0;

std::atomic<bool> started{false};
pthread_t target;
uint32_t period_ns = 0;
std::string file_path;

// See exception_handler_ps5.cpp: the machine context sits 0x40 bytes into the
// signal context on the console.
constexpr size_t kMachineContextOffset = 0x40;
// Where the title's code is mapped (the eboot loads at 0x400000); stack words
// in this range are kept as likely return addresses.
constexpr uint64_t kCodeFirst = 0x400000;
constexpr uint64_t kCodeEnd = 0x20000000;
constexpr int kStackScanWords = 48;

void OnSignal(int, siginfo_t*, void* signal_context) {
  const auto* mc = reinterpret_cast<const mcontext_t*>(static_cast<uint8_t*>(signal_context) +
                                                       kMachineContextOffset);
  Sample& sample = ring[written.fetch_add(1, std::memory_order_relaxed) % kRingSize];
  sample.rip = uint64_t(mc->mc_rip);
  const auto* stack = reinterpret_cast<const uint64_t*>(mc->mc_rsp);
  int found = 0;
  for (int i = 0; i < kStackScanWords && found < kSampleStackWords; ++i) {
    if (stack[i] >= kCodeFirst && stack[i] < kCodeEnd) {
      sample.stack[found++] = stack[i];
    }
  }
  for (; found < kSampleStackWords; ++found) {
    sample.stack[found] = 0;
  }
}

void* SamplerThread(void*) {
  const timespec period = {0, long(period_ns)};
  while (true) {
    nanosleep(&period, nullptr);
    pthread_kill(target, SIGPROF);
  }
  return nullptr;
}

}  // namespace

void StartForCurrentThread(uint32_t hz, std::string_view path) {
  if (!hz || path.empty() || started.exchange(true)) {
    return;
  }
  target = pthread_self();
  period_ns = 1000000000u / std::min<uint32_t>(hz, 10000);
  file_path = std::string(path);
  // Start each run with an empty file.
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd >= 0) close(fd);

  struct sigaction action {};
  action.sa_sigaction = OnSignal;
  action.sa_flags = SA_SIGINFO | SA_RESTART;
  sigemptyset(&action.sa_mask);
  sigaction(SIGPROF, &action, nullptr);
  pthread_t thread;
  if (pthread_create(&thread, nullptr, SamplerThread, nullptr) != 0) {
    REXLOG_WARN("CPU sampler: could not start the sampler thread");
    return;
  }
  pthread_detach(thread);
  REXLOG_INFO("CPU sampler: {} Hz into {}", hz, file_path);
}

void Flush() {
  if (!started.load(std::memory_order_relaxed)) {
    return;
  }
  // Leave the newest sample alone; the handler may still be writing it.
  const uint64_t end = written.load(std::memory_order_relaxed);
  if (end <= flushed + 1) {
    return;
  }
  const uint64_t last = end - 1;
  uint64_t first = flushed;
  if (last - first > kRingSize) {
    first = last - kRingSize;  // Overrun: keep the newest.
  }
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) {
    return;
  }
  for (uint64_t i = first; i < last;) {
    const size_t slot = i % kRingSize;
    const size_t count = std::min<uint64_t>(last - i, kRingSize - slot);
    write(fd, &ring[slot], count * sizeof(Sample));
    i += count;
  }
  close(fd);
  flushed = last;
}

#else

void StartForCurrentThread(uint32_t, std::string_view) {}
void Flush() {}

#endif

}  // namespace rex::perf::sampler
