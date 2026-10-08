#pragma once

#include <chrono>
#include <cstdint>

namespace rex::perf::hitch {

enum class Operation { kPipelineCompile, kGpuWait, kFileRead };

bool Enabled();
void RecordOperation(Operation operation, uint64_t microseconds);
void RecordStaleProtectionRecovery();
// Called only by the GPU command processor at the guest swap boundary.
void RecordSwap();

class ScopedOperation {
 public:
  explicit ScopedOperation(Operation operation) : operation_(operation), enabled_(Enabled()) {
    if (enabled_) start_ = std::chrono::steady_clock::now();
  }
  ~ScopedOperation() {
    if (enabled_) {
      RecordOperation(operation_, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start_).count()));
    }
  }
  ScopedOperation(const ScopedOperation&) = delete;
  ScopedOperation& operator=(const ScopedOperation&) = delete;

 private:
  Operation operation_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace rex::perf::hitch
