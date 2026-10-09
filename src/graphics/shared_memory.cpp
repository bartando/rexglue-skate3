/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <cstring>
#include <utility>

#include <rex/assert.h>
#include <rex/bit.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/shared_memory.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/perf/counter.h>
#include <rex/memory.h>

REXCVAR_DEFINE_BOOL(
    shared_memory_gpu_exact_watch_diagnostic, false, "GPU",
    "Report how many watched ranges intersect explicit GPU writes exactly "
    "versus only sharing a host page.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    shared_memory_filter_gpu_page_only_watches, false, "GPU",
    "Do not invalidate a watched range when an explicit GPU write only shares "
    "a host page with it and does not intersect its exact byte range.")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(
    shared_memory_cpu_invalidation_widen_kb, 16, "GPU",
    "How far a CPU write fault widens shared memory invalidation, in KB "
    "(rounded to a power-of-two number of host pages, at most 64). Wider means "
    "fewer access violations but more re-uploads of unchanged neighbours. "
    "Upstream used 64 host pages, which is 256 KB on 4 KB pages but 1 MB on "
    "Apple Silicon's 16 KB pages.")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(
    shared_memory_upload_shadow_max_mb, 64, "GPU",
    "Most memory, in MB, the CPU copy of uploaded pages may use. Past it the "
    "least recently used 64-page chunk is freed. 0 disables the copy, which "
    "disables upload hoisting and hot pages; -1 means no limit.")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(
    shared_memory_hot_pages, false, "GPU",
    "Stop write-protecting pages the CPU rewrites every frame; check the bytes "
    "each request needs against the last upload instead. Pays off where "
    "changing page protection is expensive (PS5).")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

namespace rex::graphics {

SharedMemory::SharedMemory(memory::Memory& memory) : memory_(memory) {
  page_size_log2_ = rex::log2_ceil(uint32_t(rex::memory::page_size()));
}

SharedMemory::~SharedMemory() {
  ShutdownCommon();
}

void SharedMemory::InitializeCommon() {
  num_system_page_flags_ = ((kBufferSize >> page_size_log2_) + 63) / 64;
  {
    uint32_t widen_pages =
        uint32_t(std::max(REXCVAR_GET(shared_memory_cpu_invalidation_widen_kb), int32_t(1))) *
        1024 >> page_size_log2_;
    widen_pages = std::clamp(widen_pages, uint32_t(1), uint32_t(64));
    cpu_invalidation_widen_pages_ = uint32_t(1) << rex::log2_floor(widen_pages);
  }
  valid_buffer_a_.assign(num_system_page_flags_, 0);
  valid_buffer_b_.assign(num_system_page_flags_, 0);
  system_page_flags_valid_and_gpu_written_.assign(num_system_page_flags_, 0);
  window_accessed_pages_.assign(num_system_page_flags_, 0);
  window_fully_accessed_pages_.assign(num_system_page_flags_, 0);
  window_inline_uploaded_pages_.assign(num_system_page_flags_, 0);
  upload_shadow_valid_.assign(num_system_page_flags_, 0);
  {
    const int32_t max_mb = REXCVAR_GET(shared_memory_upload_shadow_max_mb);
    upload_shadow_max_chunks_ =
        max_mb < 0 ? SIZE_MAX
                   : (size_t(max_mb) << 20) / upload_shadow_chunk_bytes();
    rex::perf::cpu_profile::upload_shadow_cap_bytes.store(
        max_mb < 0 ? 0 : upload_shadow_max_chunks_ * upload_shadow_chunk_bytes(),
        std::memory_order_relaxed);
  }
  // Hot pages compare against the shadow; without it they would upload on
  // every request.
  hot_pages_enabled_ = REXCVAR_GET(shared_memory_hot_pages) && upload_shadow_max_chunks_ != 0;
  hot_tracking_ = hot_pages_enabled_;
  if (hot_pages_enabled_) {
    hot_pages_.assign(num_system_page_flags_, 0);
    hot_excluded_pages_.assign(num_system_page_flags_, 0);
    cpu_invalidation_frame_.assign(kBufferSize >> page_size_log2_, 0);
    cpu_invalidation_streak_.assign(kBufferSize >> page_size_log2_, 0);
  }
  window_accessed_any_ = false;
  window_inline_uploaded_any_ = false;
  active_valid_flags_.store(valid_buffer_a_.data(), std::memory_order_relaxed);
  staging_valid_flags_.store(valid_buffer_b_.data(), std::memory_order_relaxed);
  gpu_written_data_dirty_.store(false, std::memory_order_relaxed);
  dirty_blocks_.store(0, std::memory_order_relaxed);

  memory_invalidation_callback_handle_ =
      memory_.RegisterPhysicalMemoryInvalidationCallback(MemoryInvalidationCallbackThunk, this);
}

void SharedMemory::InitializeSparseHostGpuMemory(uint32_t granularity_log2) {
  assert_true(granularity_log2 <= kBufferSizeLog2);
  assert_true(host_gpu_memory_sparse_granularity_log2_ == UINT32_MAX);
  host_gpu_memory_sparse_granularity_log2_ = granularity_log2;
  host_gpu_memory_sparse_allocated_.resize(
      size_t(1) << (std::max(kBufferSizeLog2 - granularity_log2, uint32_t(6)) - 6));
}

void SharedMemory::ShutdownCommon() {
  ReleaseTraceDownloadRanges();
  upload_shadow_chunks_.clear();
  upload_shadow_chunk_used_frame_.clear();
  upload_shadow_chunk_count_ = 0;
  rex::perf::cpu_profile::upload_shadow_bytes.store(0, std::memory_order_relaxed);

  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, false);
  assert_true(global_watches_.empty());
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();

  if (memory_invalidation_callback_handle_ != nullptr) {
    memory_.UnregisterPhysicalMemoryInvalidationCallback(memory_invalidation_callback_handle_);
    memory_invalidation_callback_handle_ = nullptr;
  }

  if (host_gpu_memory_sparse_used_bytes_) {
    host_gpu_memory_sparse_used_bytes_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_used_mb", 0);
  }
  if (host_gpu_memory_sparse_allocations_) {
    host_gpu_memory_sparse_allocations_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_allocations", 0);
  }
  host_gpu_memory_sparse_allocated_.clear();
  host_gpu_memory_sparse_allocated_.shrink_to_fit();
  host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;

  active_valid_flags_.store(nullptr, std::memory_order_relaxed);
  staging_valid_flags_.store(nullptr, std::memory_order_relaxed);
  valid_buffer_a_.clear();
  valid_buffer_a_.shrink_to_fit();
  valid_buffer_b_.clear();
  valid_buffer_b_.shrink_to_fit();
  system_page_flags_valid_and_gpu_written_.clear();
  system_page_flags_valid_and_gpu_written_.shrink_to_fit();
  num_system_page_flags_ = 0;
  gpu_written_data_dirty_.store(false, std::memory_order_relaxed);
  dirty_blocks_.store(0, std::memory_order_relaxed);
}

void SharedMemory::InvalidateAllPages() {
  auto global_lock = global_critical_region_.Acquire();

  uint64_t* active = active_valid_flags_.load(std::memory_order_relaxed);
  uint64_t* staging = staging_valid_flags_.load(std::memory_order_relaxed);
  if (active && num_system_page_flags_) {
    std::memset(active, 0, num_system_page_flags_ * sizeof(uint64_t));
  }
  if (staging && num_system_page_flags_) {
    std::memset(staging, 0, num_system_page_flags_ * sizeof(uint64_t));
  }
  if (!system_page_flags_valid_and_gpu_written_.empty()) {
    std::memset(system_page_flags_valid_and_gpu_written_.data(), 0,
                num_system_page_flags_ * sizeof(uint64_t));
  }

  // Force a refresh on the next frame-end sync.
  dirty_blocks_.store(UINT32_MAX, std::memory_order_relaxed);
  gpu_written_data_dirty_.store(true, std::memory_order_relaxed);
}

void SharedMemory::SetSystemPageBlocksValidWithGpuDataWritten() {
  if (!gpu_written_data_dirty_.load(std::memory_order_relaxed)) {
    return;
  }

  uint64_t* staging = staging_valid_flags_.load(std::memory_order_acquire);
  if (!staging || !num_system_page_flags_) {
    gpu_written_data_dirty_.store(false, std::memory_order_relaxed);
    dirty_blocks_.store(0, std::memory_order_relaxed);
    return;
  }

  uint32_t dirty_mask = dirty_blocks_.exchange(0, std::memory_order_relaxed);
  uint32_t dirty_count = rex::bit_count(dirty_mask);
  if (dirty_count == 0 || dirty_count > 16) {
    std::memcpy(staging, system_page_flags_valid_and_gpu_written_.data(),
                num_system_page_flags_ * sizeof(uint64_t));
  } else {
    while (dirty_mask) {
      uint32_t block_index;
      rex::bit_scan_forward(dirty_mask, &block_index);
      dirty_mask &= ~(uint32_t(1) << block_index);
      uint32_t entry_offset = block_index * 64;
      if (entry_offset >= num_system_page_flags_) {
        continue;
      }
      uint32_t entry_count = std::min(uint32_t(64), num_system_page_flags_ - entry_offset);
      std::memcpy(staging + entry_offset,
                  system_page_flags_valid_and_gpu_written_.data() + entry_offset,
                  entry_count * sizeof(uint64_t));
    }
  }

  uint64_t* old_active = active_valid_flags_.exchange(staging, std::memory_order_acq_rel);
  staging_valid_flags_.store(old_active, std::memory_order_release);
  gpu_written_data_dirty_.store(false, std::memory_order_relaxed);
}

void SharedMemory::ClearCache() {
  // Keeping GPU-written data, so "invalidated by GPU".
  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, true);
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();
  SetSystemPageBlocksValidWithGpuDataWritten();
}

SharedMemory::GlobalWatchHandle SharedMemory::RegisterGlobalWatch(GlobalWatchCallback callback,
                                                                  void* callback_context) {
  GlobalWatch* watch = new GlobalWatch;
  watch->callback = callback;
  watch->callback_context = callback_context;

  auto global_lock = global_critical_region_.Acquire();
  global_watches_.push_back(watch);

  return reinterpret_cast<GlobalWatchHandle>(watch);
}

void SharedMemory::UnregisterGlobalWatch(GlobalWatchHandle handle) {
  auto watch = reinterpret_cast<GlobalWatch*>(handle);

  {
    auto global_lock = global_critical_region_.Acquire();
    auto it = std::find(global_watches_.begin(), global_watches_.end(), watch);
    assert_false(it == global_watches_.end());
    if (it != global_watches_.end()) {
      global_watches_.erase(it);
    }
  }

  delete watch;
}

SharedMemory::WatchHandle SharedMemory::WatchMemoryRange(uint32_t start, uint32_t length,
                                                         WatchCallback callback,
                                                         void* callback_context,
                                                         void* callback_data,
                                                         uint64_t callback_argument) {
  if (length == 0 || start >= kBufferSize) {
    return nullptr;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t watch_page_first = start >> page_size_log2_;
  uint32_t watch_page_last = (start + length - 1) >> page_size_log2_;
  uint32_t bucket_first = watch_page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  uint32_t bucket_last = watch_page_last << page_size_log2_ >> kWatchBucketSizeLog2;

  auto global_lock = global_critical_region_.Acquire();

  // Allocate the range.
  WatchRange* range = watch_range_first_free_;
  if (range != nullptr) {
    watch_range_first_free_ = range->next_free;
  } else {
    if (watch_range_pools_.empty() || watch_range_current_pool_allocated_ >= kWatchRangePoolSize) {
      watch_range_pools_.push_back(new WatchRange[kWatchRangePoolSize]);
      watch_range_current_pool_allocated_ = 0;
    }
    range = &(watch_range_pools_.back()[watch_range_current_pool_allocated_++]);
  }
  range->callback = callback;
  range->callback_context = callback_context;
  range->callback_data = callback_data;
  range->callback_argument = callback_argument;
  range->address_first = start;
  range->address_last = start + length - 1;
  range->page_first = watch_page_first;
  range->page_last = watch_page_last;

  // Allocate and link the nodes.
  WatchNode* node_previous = nullptr;
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_node_first_free_;
    if (node != nullptr) {
      watch_node_first_free_ = node->next_free;
    } else {
      if (watch_node_pools_.empty() || watch_node_current_pool_allocated_ >= kWatchNodePoolSize) {
        watch_node_pools_.push_back(new WatchNode[kWatchNodePoolSize]);
        watch_node_current_pool_allocated_ = 0;
      }
      node = &(watch_node_pools_.back()[watch_node_current_pool_allocated_++]);
    }
    node->range = range;
    node->range_node_next = nullptr;
    if (node_previous != nullptr) {
      node_previous->range_node_next = node;
    } else {
      range->node_first = node;
    }
    node_previous = node;
    node->bucket_node_previous = nullptr;
    node->bucket_node_next = watch_buckets_[i];
    if (watch_buckets_[i] != nullptr) {
      watch_buckets_[i]->bucket_node_previous = node;
    }
    watch_buckets_[i] = node;
  }

  if (hot_pages_enabled_) {
    DemoteHotPages(watch_page_first, watch_page_last);
  }
  if (zero_copy_) {
    // No upload protects these pages, so the watch has to: a CPU write must
    // fault for it to fire. Hot pages are left unprotected; OnFrameEnd fires
    // their watches.
    uint32_t run_first = UINT32_MAX;
    for (uint32_t page = watch_page_first; page <= watch_page_last + 1; ++page) {
      const bool protect = page <= watch_page_last && !IsHotPage(page);
      if (protect && run_first == UINT32_MAX) {
        run_first = page;
      } else if (!protect && run_first != UINT32_MAX) {
        memory().EnablePhysicalMemoryAccessCallbacks(
            run_first << page_size_log2_, (page - run_first) << page_size_log2_, true, false);
        run_first = UINT32_MAX;
      }
    }
  }

  return reinterpret_cast<WatchHandle>(range);
}

bool SharedMemory::RequestRangeIfBytesEqual(uint32_t start, uint32_t length,
                                            const uint8_t* expected) {
  if (expected == nullptr || length == 0 || start >= kBufferSize ||
      length > kBufferSize - start) {
    return false;
  }
  auto global_lock = global_critical_region_.Acquire();
  const uint8_t* guest_bytes =
      memory_.TranslatePhysical<const uint8_t*>(start);
  if (guest_bytes == nullptr ||
      std::memcmp(guest_bytes, expected, length) != 0) {
    return false;
  }
  // RequestRange acquires the recursive global critical region internally.
  // Keeping this outer lock prevents a CPU invalidation from racing the exact
  // comparison and residency refresh.
  if (!RequestRange(start, length)) {
    return false;
  }
  guest_bytes = memory_.TranslatePhysical<const uint8_t*>(start);
  return guest_bytes != nullptr &&
         std::memcmp(guest_bytes, expected, length) == 0;
}

bool SharedMemory::GuestBytesEqual(uint32_t start, uint32_t length,
                                   const uint8_t* expected) {
  if (expected == nullptr || length == 0 || start >= kBufferSize ||
      length > kBufferSize - start) {
    return false;
  }
  auto global_lock = global_critical_region_.Acquire();
  const uint8_t* guest_bytes =
      memory_.TranslatePhysical<const uint8_t*>(start);
  return guest_bytes != nullptr &&
         std::memcmp(guest_bytes, expected, length) == 0;
}

bool SharedMemory::CopyGuestBytes(uint32_t start, uint32_t length,
                                  std::vector<uint8_t>& bytes_out) {
  bytes_out.clear();
  if (length == 0 || start >= kBufferSize ||
      length > kBufferSize - start) {
    return false;
  }
  auto global_lock = global_critical_region_.Acquire();
  const uint8_t* guest_bytes =
      memory_.TranslatePhysical<const uint8_t*>(start);
  if (guest_bytes == nullptr) {
    return false;
  }
  bytes_out.assign(guest_bytes, guest_bytes + length);
  return true;
}

const uint8_t* SharedMemory::TranslatePhysical(uint32_t address) const {
  return memory_.TranslatePhysical<const uint8_t*>(address);
}

void SharedMemory::UnwatchMemoryRange(WatchHandle handle) {
  auto global_lock = global_critical_region_.Acquire();
  UnlinkWatchRange(reinterpret_cast<WatchRange*>(handle));
}

void SharedMemory::FireWatches(uint32_t page_first, uint32_t page_last, bool invalidated_by_gpu,
                               bool exact_address_range_valid, uint32_t exact_address_first,
                               uint32_t exact_address_last) {
  uint32_t address_first = page_first << page_size_log2_;
  uint32_t address_last = (page_last << page_size_log2_) + ((1 << page_size_log2_) - 1);
  uint32_t bucket_first = address_first >> kWatchBucketSizeLog2;
  uint32_t bucket_last = address_last >> kWatchBucketSizeLog2;
  const bool exact_gpu_watch_diagnostic =
      invalidated_by_gpu && exact_address_range_valid &&
      REXCVAR_GET(shared_memory_gpu_exact_watch_diagnostic);
  const bool filter_gpu_page_only_watches =
      invalidated_by_gpu && exact_address_range_valid &&
      REXCVAR_GET(shared_memory_filter_gpu_page_only_watches);
  const bool inspect_exact_gpu_watches =
      exact_gpu_watch_diagnostic || filter_gpu_page_only_watches;

  auto global_lock = global_critical_region_.Acquire();
  firing_address_first_ = address_first;
  firing_address_last_ = address_last;

  // Fire global watches.
  for (const auto global_watch : global_watches_) {
    global_watch->callback(global_lock, global_watch->callback_context, address_first, address_last,
                           invalidated_by_gpu);
  }

  // Fire per-range watches.
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_buckets_[i];
    while (node != nullptr) {
      WatchRange* range = node->range;
      // Store the next node now since when the callback is triggered, the links
      // will be broken.
      node = node->bucket_node_next;
      if (page_first <= range->page_last && page_last >= range->page_first) {
        if (inspect_exact_gpu_watches) {
          // A watched range may have a node in multiple 4 MiB buckets. Process
          // it only in the first bucket touched by this FireWatches call so a
          // filtered page-only collision is counted once and left linked.
          uint32_t range_bucket_first =
              range->address_first >> kWatchBucketSizeLog2;
          if (i != std::max(bucket_first, range_bucket_first)) {
            continue;
          }
          const bool exact_intersection =
              exact_address_first <= range->address_last &&
              exact_address_last >= range->address_first;
          if (exact_intersection) {
            ++gpu_exact_watch_hits_since_report_;
          } else {
            ++gpu_page_only_watch_hits_since_report_;
            if (filter_gpu_page_only_watches) {
              ++gpu_page_only_watch_filtered_since_report_;
              continue;
            }
          }
        }
        range->callback(global_lock, range->callback_context, range->callback_data,
                        range->callback_argument, invalidated_by_gpu);
        UnlinkWatchRange(range);
      }
    }
  }

  if (exact_gpu_watch_diagnostic &&
      ++gpu_exact_watch_writes_since_report_ >= 256) {
    REXGPU_INFO(
        "Shared memory exact GPU watches: writes={} exact={} page_only={} "
        "filtered={}",
        gpu_exact_watch_writes_since_report_,
        gpu_exact_watch_hits_since_report_,
        gpu_page_only_watch_hits_since_report_,
        gpu_page_only_watch_filtered_since_report_);
    gpu_exact_watch_writes_since_report_ = 0;
    gpu_exact_watch_hits_since_report_ = 0;
    gpu_page_only_watch_hits_since_report_ = 0;
    gpu_page_only_watch_filtered_since_report_ = 0;
  }
}

namespace {
void SetPageBits(std::vector<uint64_t>& bits, uint32_t page_first, uint32_t page_last) {
  for (uint32_t block = page_first >> 6; block <= page_last >> 6; ++block) {
    uint64_t mask = ~uint64_t(0);
    if (block == page_first >> 6) {
      mask &= ~uint64_t(0) << (page_first & 63);
    }
    if (block == page_last >> 6) {
      mask &= ~uint64_t(0) >> (63 - (page_last & 63));
    }
    bits[block] |= mask;
  }
}

void ClearPageBits(std::vector<uint64_t>& bits, uint32_t page_first, uint32_t page_last) {
  for (uint32_t block = page_first >> 6; block <= page_last >> 6; ++block) {
    uint64_t mask = ~uint64_t(0);
    if (block == page_first >> 6) {
      mask &= ~uint64_t(0) << (page_first & 63);
    }
    if (block == page_last >> 6) {
      mask &= ~uint64_t(0) >> (63 - (page_last & 63));
    }
    bits[block] &= ~mask;
  }
}

bool PageBit(const std::vector<uint64_t>& bits, uint32_t page) {
  return (bits[page >> 6] >> (page & 63)) & 1;
}
}  // namespace

void SharedMemory::RangeWrittenByGpu(uint32_t start, uint32_t length) {
  if (length == 0 || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t end = start + length - 1;
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = end >> page_size_log2_;

  // Trigger modification callbacks so, for instance, resolved data is loaded to
  // the texture.
  FireWatches(page_first, page_last, true, true, start, end);
  if (zero_copy_) {
    // The GPU wrote guest memory itself; there is no copy to keep valid.
    return;
  }

  // Mark the range as valid (so pages are not reuploaded until modified by the
  // CPU) and watch it so the CPU can reuse it and this will be caught.
  MakeRangeValid(start, length, true);
  NoteGpuAccess(start, length);
  ClearPageBits(upload_shadow_valid_, page_first, page_last);
}

bool SharedMemory::AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                    uint32_t length_allocations) {
  assert_always(
      "Sparse host GPU memory allocation has been initialized, but the "
      "implementation doesn't provide AllocateSparseHostGpuMemoryRange");
  return false;
}

void SharedMemory::MakeRangeValid(uint32_t start, uint32_t length, bool written_by_gpu) {
  rex::perf::cpu_profile::OwnerScope profile(rex::perf::CounterId::kCpuSharedMemoryMakeValidUs);
  if (length == 0 || start >= kBufferSize || zero_copy_) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t last = start + length - 1;
  uint32_t valid_page_first = start >> page_size_log2_;
  uint32_t valid_page_last = last >> page_size_log2_;
  uint32_t valid_block_first = valid_page_first >> 6;
  uint32_t valid_block_last = valid_page_last >> 6;

  bool any_hot = false;
  {
    auto global_lock = global_critical_region_.Acquire();
    uint64_t* valid_flags = active_valid_flags_.load(std::memory_order_relaxed);

    for (uint32_t i = valid_block_first; i <= valid_block_last; ++i) {
      uint64_t valid_bits = UINT64_MAX;
      if (i == valid_block_first) {
        valid_bits &= ~((uint64_t(1) << (valid_page_first & 63)) - 1);
      }
      if (i == valid_block_last && (valid_page_last & 63) != 63) {
        valid_bits &= (uint64_t(1) << ((valid_page_last & 63) + 1)) - 1;
      }
      if (hot_pages_enabled_) {
        if (written_by_gpu) {
          // The GPU copy is now authoritative and the shadow is stale.
          hot_pages_[i] &= ~valid_bits;
        } else if (hot_pages_[i] & valid_bits) {
          any_hot = true;
          valid_bits &= ~hot_pages_[i];
        }
      }
      if (valid_flags) {
        valid_flags[i] |= valid_bits;
      }
      uint64_t old_gpu_written = system_page_flags_valid_and_gpu_written_[i];
      uint64_t new_gpu_written =
          written_by_gpu ? (old_gpu_written | valid_bits) : (old_gpu_written & ~valid_bits);
      if (new_gpu_written != old_gpu_written) {
        system_page_flags_valid_and_gpu_written_[i] = new_gpu_written;
        gpu_written_data_dirty_.store(true, std::memory_order_relaxed);
        dirty_blocks_.fetch_or(uint32_t(1) << (i >> 6), std::memory_order_relaxed);
      }
    }
  }

  if (memory_invalidation_callback_handle_) {
    if (!any_hot) {
      memory().EnablePhysicalMemoryAccessCallbacks(
          valid_page_first << page_size_log2_,
          (valid_page_last - valid_page_first + 1) << page_size_log2_, true, false);
    } else {
      // Hot pages stay unprotected; protect the runs between them.
      uint32_t run_first = UINT32_MAX;
      for (uint32_t page = valid_page_first; page <= valid_page_last + 1; ++page) {
        const bool protect = page <= valid_page_last && !IsHotPage(page);
        if (protect && run_first == UINT32_MAX) {
          run_first = page;
        } else if (!protect && run_first != UINT32_MAX) {
          memory().EnablePhysicalMemoryAccessCallbacks(
              run_first << page_size_log2_, (page - run_first) << page_size_log2_, true, false);
          run_first = UINT32_MAX;
        }
      }
    }
  }
}

void SharedMemory::UnlinkWatchRange(WatchRange* range) {
  uint32_t bucket = range->page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  WatchNode* node = range->node_first;
  while (node != nullptr) {
    WatchNode* node_next = node->range_node_next;
    if (node->bucket_node_previous != nullptr) {
      node->bucket_node_previous->bucket_node_next = node->bucket_node_next;
    } else {
      watch_buckets_[bucket] = node->bucket_node_next;
    }
    if (node->bucket_node_next != nullptr) {
      node->bucket_node_next->bucket_node_previous = node->bucket_node_previous;
    }
    node->next_free = watch_node_first_free_;
    watch_node_first_free_ = node;
    node = node_next;
    ++bucket;
  }
  range->next_free = watch_range_first_free_;
  watch_range_first_free_ = range;
}


void SharedMemory::NoteGpuAccess(uint32_t start, uint32_t length) {
  if (!length || start >= kBufferSize || window_accessed_pages_.empty()) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  const uint32_t page_size = uint32_t(1) << page_size_log2_;
  const uint32_t end = start + length;
  const uint32_t page_first = start >> page_size_log2_;
  const uint32_t page_last = (end - 1) >> page_size_log2_;
  SetPageBits(window_accessed_pages_, page_first, page_last);
  window_accessed_any_ = true;

  const uint32_t offset_first = start & (page_size - 1);
  const uint32_t offset_end = end - (page_last << page_size_log2_);
  uint32_t full_first = page_first;
  uint32_t full_last = page_last;
  if (page_first == page_last) {
    if (offset_first || offset_end != page_size) {
      NotePartialPageAccess(page_first, offset_first, offset_end);
      return;
    }
  } else {
    if (offset_first) {
      NotePartialPageAccess(page_first, offset_first, page_size);
      ++full_first;
    }
    if (offset_end != page_size) {
      NotePartialPageAccess(page_last, 0, offset_end);
      --full_last;
    }
  }
  if (full_first <= full_last) {
    SetPageBits(window_fully_accessed_pages_, full_first, full_last);
  }
}

void SharedMemory::NotePartialPageAccess(uint32_t page, uint32_t offset_first,
                                         uint32_t offset_end) {
  // Bounds the cost per page; collapsing to the hull is conservative.
  constexpr size_t kMaxIntervals = 16;
  auto& intervals = window_partial_page_access_[page];
  auto it = std::lower_bound(intervals.begin(), intervals.end(), offset_first,
                             [](const auto& interval, uint32_t offset) {
                               return interval.second < offset;
                             });
  auto merge_end = it;
  while (merge_end != intervals.end() && merge_end->first <= offset_end) {
    offset_first = std::min(offset_first, merge_end->first);
    offset_end = std::max(offset_end, merge_end->second);
    ++merge_end;
  }
  it = intervals.erase(it, merge_end);
  intervals.emplace(it, offset_first, offset_end);
  if (intervals.size() > kMaxIntervals) {
    intervals = {{intervals.front().first, intervals.back().second}};
  }
}

void SharedMemory::ResetGpuAccessWindow() {
  if (window_accessed_any_) {
    std::fill(window_accessed_pages_.begin(), window_accessed_pages_.end(), 0);
    std::fill(window_fully_accessed_pages_.begin(), window_fully_accessed_pages_.end(), 0);
    window_partial_page_access_.clear();
    window_accessed_any_ = false;
  }
  if (window_inline_uploaded_any_) {
    std::fill(window_inline_uploaded_pages_.begin(), window_inline_uploaded_pages_.end(), 0);
    window_inline_uploaded_any_ = false;
  }
}

bool SharedMemory::UploadChangesAccessedBytes(
    const std::vector<std::pair<uint32_t, const uint8_t*>>& new_pages) const {
  if (!window_accessed_any_) {
    return false;
  }
  for (const auto& [page, data] : new_pages) {
    if (!PageBit(window_accessed_pages_, page)) {
      continue;
    }
    if (PageBit(window_inline_uploaded_pages_, page)) {
      return true;
    }
    if (!PageBit(upload_shadow_valid_, page)) {
      return true;
    }
    const uint8_t* shadow = UploadShadowPage(page);
    if (!shadow) {
      return true;
    }
    auto bytes_changed = [&](uint32_t offset_first, uint32_t offset_end) {
      return std::memcmp(shadow + offset_first, data + offset_first,
                         offset_end - offset_first) != 0;
    };
    bool changed = false;
    if (PageBit(window_fully_accessed_pages_, page)) {
      changed = bytes_changed(0, uint32_t(1) << page_size_log2_);
    } else if (auto partial = window_partial_page_access_.find(page);
               partial != window_partial_page_access_.end()) {
      for (const auto& [offset_first, offset_end] : partial->second) {
        if (bytes_changed(offset_first, offset_end)) {
          changed = true;
          break;
        }
      }
    } else {
      changed = bytes_changed(0, uint32_t(1) << page_size_log2_);
    }
    if (changed) {
      return true;
    }
  }
  return false;
}

void SharedMemory::NoteUploadedPages(
    const std::vector<std::pair<uint32_t, const uint8_t*>>& pages, bool hoisted) {
  const size_t page_size = size_t(1) << page_size_log2_;
  for (const auto& [page, data] : pages) {
    if (uint8_t* shadow = AllocateUploadShadowPage(page)) {
      std::memcpy(shadow, data, page_size);
      upload_shadow_valid_[page >> 6] |= uint64_t(1) << (page & 63);
    }
    if (!hoisted) {
      window_inline_uploaded_pages_[page >> 6] |= uint64_t(1) << (page & 63);
      window_inline_uploaded_any_ = true;
    }
  }
}

const uint8_t* SharedMemory::UploadShadowPage(uint32_t page) const {
  const size_t chunk = page >> 6;
  if (chunk >= upload_shadow_chunks_.size() || !upload_shadow_chunks_[chunk]) {
    return nullptr;
  }
  return upload_shadow_chunks_[chunk].get() + (size_t(page & 63) << page_size_log2_);
}

uint8_t* SharedMemory::AllocateUploadShadowPage(uint32_t page) {
  const size_t chunk = page >> 6;
  if (chunk >= upload_shadow_chunks_.size()) {
    const size_t chunk_count = ((kBufferSize >> page_size_log2_) + 63) >> 6;
    upload_shadow_chunks_.resize(chunk_count);
    upload_shadow_chunk_used_frame_.resize(chunk_count, 0);
  }
  std::unique_ptr<uint8_t[]>& storage = upload_shadow_chunks_[chunk];
  if (!storage) {
    if (!upload_shadow_max_chunks_) {
      return nullptr;
    }
    if (upload_shadow_chunk_count_ >= upload_shadow_max_chunks_) {
      EvictUploadShadowChunk(chunk);
    }
    storage.reset(new uint8_t[upload_shadow_chunk_bytes()]);
    ++upload_shadow_chunk_count_;
    rex::perf::cpu_profile::upload_shadow_bytes.store(
        upload_shadow_chunk_count_ * upload_shadow_chunk_bytes(), std::memory_order_relaxed);
  }
  NoteUploadShadowUse(page);
  return storage.get() + (size_t(page & 63) << page_size_log2_);
}

void SharedMemory::EvictUploadShadowChunk(size_t keep_chunk) {
  size_t victim = SIZE_MAX;
  for (size_t chunk = 0; chunk < upload_shadow_chunks_.size(); ++chunk) {
    if (chunk != keep_chunk && upload_shadow_chunks_[chunk] &&
        (victim == SIZE_MAX ||
         upload_shadow_chunk_used_frame_[chunk] < upload_shadow_chunk_used_frame_[victim])) {
      victim = chunk;
    }
  }
  if (victim == SIZE_MAX) {
    return;
  }
  upload_shadow_chunks_[victim].reset();
  --upload_shadow_chunk_count_;
  upload_shadow_valid_[victim] = 0;
  rex::perf::cpu_profile::AddOwnerCount(rex::perf::CounterId::kCpuSharedMemoryShadowEvictions, 1);
}

bool SharedMemory::RequestRanges(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  bool result = RequestRangesUntracked(ranges, count);
  for (size_t i = 0; ranges && i < count; ++i) {
    NoteGpuAccess(ranges[i].first, ranges[i].second);
  }
  return result;
}

bool SharedMemory::RequestRangesUntracked(const std::pair<uint32_t, uint32_t>* ranges,
                                          size_t count) {
  if (ranges == nullptr || !count || zero_copy_) {
    return true;
  }
  if (count == 1) {
    return RequestRangeUntracked(ranges[0].first, ranges[0].second);
  }
  CheckDemotedPages();

  // Some texture or buffer is empty, for example - safe to draw in this case.
  std::vector<std::pair<uint32_t, uint32_t>> merged_ranges;
  merged_ranges.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    uint32_t start = ranges[i].first;
    uint32_t length = ranges[i].second;
    if (!length) {
      continue;
    }
    if (start > kBufferSize || (kBufferSize - start) < length) {
      return false;
    }
    merged_ranges.emplace_back(start, length);
  }
  if (merged_ranges.empty()) {
    return true;
  }

  SCOPE_profile_cpu_f("gpu");

  std::sort(merged_ranges.begin(), merged_ranges.end(),
            [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
              return a.first < b.first;
            });
  size_t merged_write = 0;
  for (size_t i = 1; i < merged_ranges.size(); ++i) {
    std::pair<uint32_t, uint32_t>& range_previous = merged_ranges[merged_write];
    const std::pair<uint32_t, uint32_t>& range_current = merged_ranges[i];
    uint64_t previous_end = uint64_t(range_previous.first) + uint64_t(range_previous.second);
    uint64_t current_start = uint64_t(range_current.first);
    if (current_start <= previous_end) {
      uint64_t current_end = current_start + uint64_t(range_current.second);
      if (current_end > previous_end) {
        range_previous.second = uint32_t(current_end - uint64_t(range_previous.first));
      }
    } else {
      merged_ranges[++merged_write] = range_current;
    }
  }
  merged_ranges.resize(merged_write + 1);

  for (const std::pair<uint32_t, uint32_t>& range : merged_ranges) {
    if (!EnsureHostGpuMemoryAllocated(range.first, range.second)) {
      return false;
    }
  }

  uint64_t* valid_flags = active_valid_flags_.load(std::memory_order_acquire);
  if (valid_flags) {
    bool all_valid = true;
    for (const std::pair<uint32_t, uint32_t>& range : merged_ranges) {
      if (!range.second) {
        continue;
      }
      uint32_t page_first = range.first >> page_size_log2_;
      uint32_t page_last = (range.first + range.second - 1) >> page_size_log2_;
      uint32_t block_first = page_first >> 6;
      uint32_t block_last = page_last >> 6;
      for (uint32_t i = block_first; i <= block_last; ++i) {
        uint64_t block_valid = valid_flags[i];
        if (i == block_first) {
          uint64_t block_before = (uint64_t(1) << (page_first & 63)) - 1;
          block_valid |= block_before;
        }
        if (i == block_last && (page_last & 63) != 63) {
          uint64_t block_inside = (uint64_t(1) << ((page_last & 63) + 1)) - 1;
          block_valid |= ~block_inside;
        }
        if (block_valid != UINT64_MAX) {
          all_valid = false;
          break;
        }
      }
      if (!all_valid) {
        break;
      }
    }
    if (all_valid) {
      COUNT_profile_set("gpu/shared_memory/request_ranges_count", uint32_t(count));
      COUNT_profile_set("gpu/shared_memory/request_ranges_merged_count",
                        uint32_t(merged_ranges.size()));
      COUNT_profile_set("gpu/shared_memory/request_ranges_upload_count", 0);
      return true;
    }
  }

  upload_ranges_.clear();
  auto append_upload_range = [this](uint32_t page_start, uint32_t page_count) {
    if (!page_count) {
      return;
    }
    if (!upload_ranges_.empty()) {
      std::pair<uint32_t, uint32_t>& last_upload_range = upload_ranges_.back();
      if (last_upload_range.first + last_upload_range.second == page_start) {
        last_upload_range.second += page_count;
        return;
      }
    }
    upload_ranges_.emplace_back(page_start, page_count);
  };
  {
    auto global_lock = global_critical_region_.Acquire();
    valid_flags = active_valid_flags_.load(std::memory_order_relaxed);
    for (const std::pair<uint32_t, uint32_t>& range : merged_ranges) {
      uint32_t page_first = range.first >> page_size_log2_;
      uint32_t page_last = (range.first + range.second - 1) >> page_size_log2_;
      uint32_t block_first = page_first >> 6;
      uint32_t block_last = page_last >> 6;
      uint32_t range_start = UINT32_MAX;
      for (uint32_t i = block_first; i <= block_last; ++i) {
        uint64_t block_valid = valid_flags ? valid_flags[i] : 0;
        // Consider pages in the block outside the requested range valid.
        if (i == block_first) {
          uint64_t block_before = (uint64_t(1) << (page_first & 63)) - 1;
          block_valid |= block_before;
        }
        if (i == block_last && (page_last & 63) != 63) {
          uint64_t block_inside = (uint64_t(1) << ((page_last & 63) + 1)) - 1;
          block_valid |= ~block_inside;
        }

        while (true) {
          uint32_t block_page;
          if (range_start == UINT32_MAX) {
            // Check if need to open a new range.
            if (!rex::bit_scan_forward(~block_valid, &block_page)) {
              break;
            }
            range_start = (i << 6) + block_page;
          } else {
            // Check if need to close the range.
            // Ignore the valid pages before the beginning of the range.
            uint64_t block_valid_from_start = block_valid;
            if (i == (range_start >> 6)) {
              block_valid_from_start &= ~((uint64_t(1) << (range_start & 63)) - 1);
            }
            if (!rex::bit_scan_forward(block_valid_from_start, &block_page)) {
              break;
            }
            append_upload_range(range_start, (i << 6) + block_page - range_start);
            // In the next iteration within this block, consider this range
            // valid since it has been queued for upload.
            block_valid |= (uint64_t(1) << block_page) - 1;
            range_start = UINT32_MAX;
          }
        }
      }
      if (range_start != UINT32_MAX) {
        append_upload_range(range_start, page_last + 1 - range_start);
      }
    }
  }

  COUNT_profile_set("gpu/shared_memory/request_ranges_count", uint32_t(count));
  COUNT_profile_set("gpu/shared_memory/request_ranges_merged_count",
                    uint32_t(merged_ranges.size()));
  COUNT_profile_set("gpu/shared_memory/request_ranges_upload_count",
                    uint32_t(upload_ranges_.size()));

  if (hot_pages_enabled_) {
    FilterUnchangedHotPages(merged_ranges.data(), merged_ranges.size());
  }
  if (upload_ranges_.empty()) {
    return true;
  }
  WidenUploadRanges();

  return UploadRanges(upload_ranges_);
}

void SharedMemory::WidenUploadRanges() {
  if (cpu_invalidation_widen_pages_ <= 1) {
    return;
  }
  const uint32_t window_mask = cpu_invalidation_widen_pages_ - 1;
  const uint32_t page_last_in_buffer = (kBufferSize >> page_size_log2_) - 1;
  // Read without the global lock, like the all-valid fast path: a page
  // validated concurrently is only uploaded once more.
  const uint64_t* valid_flags = active_valid_flags_.load(std::memory_order_acquire);
  memory::BaseHeap* physical_heap = memory().GetPhysicalHeap();
  auto widenable = [&](uint32_t page) {
    if (IsHotPage(page) ||
        (valid_flags && (valid_flags[page >> 6] & (uint64_t(1) << (page & 63))))) {
      return false;
    }
    // Uncommitted guest memory is inaccessible on the host; never read it.
    const uint32_t address = page << page_size_log2_;
    return physical_heap->QueryRangeAccess(address,
                                           address + (uint32_t(1) << page_size_log2_) - 1) !=
           rex::memory::PageAccess::kNoAccess;
  };
  widened_upload_ranges_.clear();
  for (auto [first, count] : upload_ranges_) {
    uint32_t last = first + count - 1;
    const uint32_t window_first = first & ~window_mask;
    const uint32_t window_last = std::min(last | window_mask, page_last_in_buffer);
    while (first > window_first && widenable(first - 1)) {
      --first;
    }
    while (last < window_last && widenable(last + 1)) {
      ++last;
    }
    widened_upload_ranges_.emplace_back(first, last + 1 - first);
  }
  std::sort(widened_upload_ranges_.begin(), widened_upload_ranges_.end());
  size_t merged = 0;
  for (size_t i = 1; i < widened_upload_ranges_.size(); ++i) {
    auto& previous = widened_upload_ranges_[merged];
    const auto& range = widened_upload_ranges_[i];
    if (range.first <= previous.first + previous.second) {
      previous.second =
          std::max(previous.first + previous.second, range.first + range.second) - previous.first;
    } else {
      widened_upload_ranges_[++merged] = range;
    }
  }
  widened_upload_ranges_.resize(merged + 1);
  for (const auto& range : widened_upload_ranges_) {
    if (!EnsureHostGpuMemoryAllocated(range.first << page_size_log2_,
                                      range.second << page_size_log2_)) {
      return;  // Keep the unwidened ranges.
    }
  }
  upload_ranges_.swap(widened_upload_ranges_);
}

void SharedMemory::EnableZeroCopyHotPages() {
  hot_pages_.assign(num_system_page_flags_, 0);
  hot_excluded_pages_.assign(num_system_page_flags_, 0);
  cpu_invalidation_frame_.assign(kBufferSize >> page_size_log2_, 0);
  cpu_invalidation_streak_.assign(kBufferSize >> page_size_log2_, 0);
  hot_tracking_ = true;
}

void SharedMemory::OnFrameEnd() {
  frame_index_.fetch_add(1, std::memory_order_relaxed);
  if (!zero_copy_ || !hot_tracking_) {
    return;
  }
  auto global_lock = global_critical_region_.Acquire();
  for (uint32_t block = 0; block < uint32_t(hot_pages_.size()); ++block) {
    uint64_t bits = hot_pages_[block];
    while (bits) {
      const uint32_t first = uint32_t(rex::tzcnt(bits));
      const uint64_t from_first = bits >> first;
      const uint32_t count = from_first == UINT64_MAX ? 64 - first
                                                      : uint32_t(rex::tzcnt(~from_first));
      const uint32_t page_first = (block << 6) + first;
      FireWatches(page_first, page_first + count - 1, false);
      bits &= count + first >= 64 ? 0 : ~uint64_t(0) << (first + count);
    }
  }
}

void SharedMemory::NoteCpuInvalidation(uint32_t page_first, uint32_t page_last) {
  // Called with the global lock held, from a guest write fault.
  if (!hot_tracking_) {
    return;
  }
  const uint32_t frame = frame_index_.load(std::memory_order_relaxed);
  for (uint32_t page = page_first; page <= page_last; ++page) {
    uint32_t& last_frame = cpu_invalidation_frame_[page];
    if (last_frame == frame) {
      continue;
    }
    uint8_t& streak = cpu_invalidation_streak_[page];
    streak = last_frame + 1 == frame ? uint8_t(std::min(streak + 1, 255)) : uint8_t(1);
    last_frame = frame;
    if (streak >= kHotPageStreak && (zero_copy_ || !PageBit(hot_excluded_pages_, page))) {
      // The fault that got here already unprotected and unwatched the page,
      // and fired (so removed) every watch on it.
      hot_pages_[page >> 6] |= uint64_t(1) << (page & 63);
    }
  }
}

void SharedMemory::DemoteHotPages(uint32_t page_first, uint32_t page_last) {
  // Called with the global lock held, from WatchMemoryRange.
  for (uint32_t page = page_first; page <= page_last; ++page) {
    if (!IsHotPage(page)) {
      continue;
    }
    hot_pages_[page >> 6] &= ~(uint64_t(1) << (page & 63));
    hot_excluded_pages_[page >> 6] |= uint64_t(1) << (page & 63);
    // From here on a guest write faults and fires the new watch. A write since
    // the page was last compared would not have; CheckDemotedPages catches it.
    // Firing now would release the watch before its owner stores the handle.
    memory().EnablePhysicalMemoryAccessCallbacks(page << page_size_log2_,
                                                 uint32_t(1) << page_size_log2_, true, false);
    pending_demotion_checks_.push_back(page);
  }
}

void SharedMemory::CheckDemotedPages() {
  if (pending_demotion_checks_.empty()) {
    return;
  }
  auto global_lock = global_critical_region_.Acquire();
  const uint32_t page_size = uint32_t(1) << page_size_log2_;
  for (uint32_t page : pending_demotion_checks_) {
    const uint32_t address = page << page_size_log2_;
    const uint8_t* guest = memory_.TranslatePhysical<const uint8_t*>(address);
    if (PageBit(upload_shadow_valid_, page) && guest &&
        std::memcmp(guest, UploadShadowPage(page), page_size) == 0) {
      continue;
    }
    MemoryInvalidationCallback(address, page_size, true);
  }
  pending_demotion_checks_.clear();
}

void SharedMemory::FilterUnchangedHotPages(const std::pair<uint32_t, uint32_t>* requested,
                                           size_t count) {
  const uint32_t page_size = uint32_t(1) << page_size_log2_;
  auto unchanged = [&](uint32_t page) {
    if (!PageBit(upload_shadow_valid_, page)) {
      return false;
    }
    const uint8_t* shadow = UploadShadowPage(page);
    NoteUploadShadowUse(page);
    const uint32_t page_start = page << page_size_log2_;
    const uint32_t page_end = page_start + page_size;
    const uint8_t* guest = memory_.TranslatePhysical<const uint8_t*>(page_start);
    if (!guest) {
      return false;
    }
    // Only the bytes this request reads have to match what the GPU holds; the
    // next request touching other bytes of the page checks those.
    for (size_t i = 0; i < count; ++i) {
      const uint32_t first = std::max(requested[i].first, page_start);
      const uint32_t end = std::min(requested[i].first + requested[i].second, page_end);
      if (first < end && std::memcmp(guest + (first - page_start),
                                     shadow + (first - page_start), end - first) != 0) {
        return false;
      }
    }
    return true;
  };
  hot_filtered_ranges_.clear();
  bool any_skipped = false;
  for (auto [first, page_count] : upload_ranges_) {
    uint32_t run_first = UINT32_MAX;
    for (uint32_t page = first; page <= first + page_count; ++page) {
      bool upload = page < first + page_count;
      if (upload && IsHotPage(page)) {
        if (unchanged(page)) {
          upload = false;
          any_skipped = true;
          rex::perf::cpu_profile::AddOwnerCount(
              rex::perf::CounterId::kCpuSharedMemoryHotSkippedPages, 1);
        } else {
          rex::perf::cpu_profile::AddOwnerCount(
              rex::perf::CounterId::kCpuSharedMemoryHotUploadedPages, 1);
        }
      }
      if (upload && run_first == UINT32_MAX) {
        run_first = page;
      } else if (!upload && run_first != UINT32_MAX) {
        hot_filtered_ranges_.emplace_back(run_first, page - run_first);
        run_first = UINT32_MAX;
      }
    }
  }
  if (any_skipped) {
    upload_ranges_.swap(hot_filtered_ranges_);
  }
}

bool SharedMemory::RequestRange(uint32_t start, uint32_t length) {
  bool result = RequestRangeUntracked(start, length);
  NoteGpuAccess(start, length);
  return result;
}

bool SharedMemory::RequestRangeUntracked(uint32_t start, uint32_t length) {
  // Some texture or buffer is empty, for example - safe to draw in this case.
  if (!length) {
    return true;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return false;
  }
  if (zero_copy_) {
    return true;
  }

  SCOPE_profile_cpu_f("gpu");
  rex::perf::cpu_profile::OwnerScope profile(rex::perf::CounterId::kCpuSharedMemoryRequestUs);
  CheckDemotedPages();

  if (!EnsureHostGpuMemoryAllocated(start, length)) {
    return false;
  }

  const uint32_t page_first = start >> page_size_log2_;
  const uint32_t page_last = (start + length - 1) >> page_size_log2_;
  const uint32_t block_first = page_first >> 6;
  const uint32_t block_last = page_last >> 6;

  uint64_t* valid_flags = active_valid_flags_.load(std::memory_order_acquire);
  if (valid_flags) {
    bool all_valid = true;
    for (uint32_t i = block_first; i <= block_last; ++i) {
      uint64_t block_valid = valid_flags[i];
      if (i == block_first) {
        uint64_t block_before = (uint64_t(1) << (page_first & 63)) - 1;
        block_valid |= block_before;
      }
      if (i == block_last && (page_last & 63) != 63) {
        uint64_t block_inside = (uint64_t(1) << ((page_last & 63) + 1)) - 1;
        block_valid |= ~block_inside;
      }
      if (block_valid != UINT64_MAX) {
        all_valid = false;
        break;
      }
    }
    if (all_valid) {
      COUNT_profile_set("gpu/shared_memory/request_ranges_count", 1);
      COUNT_profile_set("gpu/shared_memory/request_ranges_merged_count", 1);
      COUNT_profile_set("gpu/shared_memory/request_ranges_upload_count", 0);
      return true;
    }
  }

  upload_ranges_.clear();
  auto append_upload_range = [this](uint32_t page_start, uint32_t page_count) {
    if (!page_count) {
      return;
    }
    if (!upload_ranges_.empty()) {
      std::pair<uint32_t, uint32_t>& last_upload_range = upload_ranges_.back();
      if (last_upload_range.first + last_upload_range.second == page_start) {
        last_upload_range.second += page_count;
        return;
      }
    }
    upload_ranges_.emplace_back(page_start, page_count);
  };
  {
    auto global_lock = global_critical_region_.Acquire();
    valid_flags = active_valid_flags_.load(std::memory_order_relaxed);
    uint32_t range_start = UINT32_MAX;
    for (uint32_t i = block_first; i <= block_last; ++i) {
      uint64_t block_valid = valid_flags ? valid_flags[i] : 0;
      // Consider pages in the block outside the requested range valid.
      if (i == block_first) {
        uint64_t block_before = (uint64_t(1) << (page_first & 63)) - 1;
        block_valid |= block_before;
      }
      if (i == block_last && (page_last & 63) != 63) {
        uint64_t block_inside = (uint64_t(1) << ((page_last & 63) + 1)) - 1;
        block_valid |= ~block_inside;
      }

      while (true) {
        uint32_t block_page;
        if (range_start == UINT32_MAX) {
          if (!rex::bit_scan_forward(~block_valid, &block_page)) {
            break;
          }
          range_start = (i << 6) + block_page;
        } else {
          uint64_t block_valid_from_start = block_valid;
          if (i == (range_start >> 6)) {
            block_valid_from_start &= ~((uint64_t(1) << (range_start & 63)) - 1);
          }
          if (!rex::bit_scan_forward(block_valid_from_start, &block_page)) {
            break;
          }
          append_upload_range(range_start, (i << 6) + block_page - range_start);
          block_valid |= (uint64_t(1) << block_page) - 1;
          range_start = UINT32_MAX;
        }
      }
    }
    if (range_start != UINT32_MAX) {
      append_upload_range(range_start, page_last + 1 - range_start);
    }
  }

  COUNT_profile_set("gpu/shared_memory/request_ranges_count", 1);
  COUNT_profile_set("gpu/shared_memory/request_ranges_merged_count", 1);
  COUNT_profile_set("gpu/shared_memory/request_ranges_upload_count",
                    uint32_t(upload_ranges_.size()));

  if (hot_pages_enabled_) {
    const std::pair<uint32_t, uint32_t> requested(start, length);
    FilterUnchangedHotPages(&requested, 1);
  }
  if (upload_ranges_.empty()) {
    return true;
  }
  WidenUploadRanges();

  rex::perf::cpu_profile::OwnerScope upload_profile(
      rex::perf::CounterId::kCpuSharedMemoryUploadUs);
  if (rex::perf::cpu_profile::owner_thread &&
      rex::perf::cpu_profile::enabled.load(std::memory_order_relaxed)) {
    for (const auto& range : upload_ranges_) {
      rex::perf::cpu_profile::AddOwnerCount(rex::perf::CounterId::kCpuSharedMemoryUploadPages,
                                            range.second);
      for (uint32_t page = 0; page < range.second; ++page) {
        rex::perf::cpu_profile::frame_uploaded_pages.push_back(range.first + page);
      }
    }
  }
  return UploadRanges(upload_ranges_);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallbackThunk(
    void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range) {
  return reinterpret_cast<SharedMemory*>(context_ptr)
      ->MemoryInvalidationCallback(physical_address_start, length, exact_range);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallback(
    uint32_t physical_address_start, uint32_t length, bool exact_range) {
  if (length == 0 || physical_address_start >= kBufferSize) {
    return std::make_pair(uint32_t(0), UINT32_MAX);
  }
  length = std::min(length, kBufferSize - physical_address_start);
  uint32_t physical_address_last = physical_address_start + (length - 1);

  uint32_t page_first = physical_address_start >> page_size_log2_;
  uint32_t page_last = physical_address_last >> page_size_log2_;
  uint32_t block_first = page_first >> 6;
  uint32_t block_last = page_last >> 6;

  auto global_lock = global_critical_region_.Acquire();

  if (!exact_range) {
    // Check if a somewhat wider range (up to the configured widening window) can be
    // invalidated (up to the configured widening window) - if no GPU-written data nearby that was not intended to be
    // invalidated since it's not in sync with CPU memory and can't be
    // reuploaded. It's a lot cheaper to upload some excess data than to catch
    // access violations - with 4 KB callbacks, 58410824 (being a
    // software-rendered game) runs at 4 FPS on Intel Core i7-3770, with 64 KB,
    // the CPU game code takes 3 ms to run per frame, but with 256 KB, it's 0.7
    // ms.
    const uint32_t widen_mask = cpu_invalidation_widen_pages_ - 1;
    if (page_first & widen_mask) {
      uint64_t gpu_written_start = system_page_flags_valid_and_gpu_written_[block_first];
      gpu_written_start &= (uint64_t(1) << (page_first & 63)) - 1;
      // Pages before the widening window act as a barrier, like GPU-written ones.
      gpu_written_start |= (uint64_t(1) << (page_first & 63 & ~widen_mask)) - 1;
      page_first = (page_first & ~uint32_t(63)) + (64 - rex::lzcnt(gpu_written_start));
    }
    if ((page_last & widen_mask) != widen_mask) {
      uint64_t gpu_written_end = system_page_flags_valid_and_gpu_written_[block_last];
      gpu_written_end &= ~((uint64_t(1) << ((page_last & 63) + 1)) - 1);
      uint32_t window_last = (page_last | widen_mask) & 63;
      if (window_last != 63) {
        gpu_written_end |= ~((uint64_t(1) << (window_last + 1)) - 1);
      }
      page_last =
          (page_last & ~uint32_t(63)) + (std::max(rex::tzcnt(gpu_written_end), uint8_t(1)) - 1);
    }
  }

  uint32_t dirty_blocks_mask = 0;
  uint64_t* valid_flags = active_valid_flags_.load(std::memory_order_relaxed);
  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t invalidate_bits = UINT64_MAX;
    if (i == block_first) {
      invalidate_bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      invalidate_bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    if (valid_flags) {
      valid_flags[i] &= ~invalidate_bits;
    }
    system_page_flags_valid_and_gpu_written_[i] &= ~invalidate_bits;
    dirty_blocks_mask |= uint32_t(1) << (i >> 6);
  }
  gpu_written_data_dirty_.store(true, std::memory_order_relaxed);
  dirty_blocks_.fetch_or(dirty_blocks_mask, std::memory_order_relaxed);

  if (!exact_range) {
    // A guest write fault (explicit invalidations pass exact ranges).
    NoteCpuInvalidation(page_first, page_last);
  }

  FireWatches(page_first, page_last, false);

  return std::make_pair(page_first << page_size_log2_, (page_last - page_first + 1)
                                                           << page_size_log2_);
}

void SharedMemory::PrepareForTraceDownload() {
  ReleaseTraceDownloadRanges();
  assert_true(trace_download_ranges_.empty());
  assert_zero(trace_download_page_count_);

  // Invalidate the entire memory CPU->GPU memory copy so all the history
  // doesn't have to be written into every frame trace, and collect the list of
  // ranges with data modified on the GPU.

  uint32_t fire_watches_range_start = UINT32_MAX;
  uint32_t gpu_written_range_start = UINT32_MAX;
  auto global_lock = global_critical_region_.Acquire();
  uint64_t* valid_flags = active_valid_flags_.load(std::memory_order_relaxed);
  for (uint32_t i = 0; i < num_system_page_flags_; ++i) {
    uint64_t previously_valid_block = valid_flags ? valid_flags[i] : 0;
    uint64_t gpu_written_block = system_page_flags_valid_and_gpu_written_[i];
    if (valid_flags) {
      valid_flags[i] = gpu_written_block;
    }

    // Fire watches on the invalidated pages.
    uint64_t fire_watches_block = previously_valid_block & ~gpu_written_block;
    uint64_t fire_watches_break_block = ~fire_watches_block;
    while (true) {
      uint32_t fire_watches_block_page;
      if (!rex::bit_scan_forward(fire_watches_range_start == UINT32_MAX ? fire_watches_block
                                                                        : fire_watches_break_block,
                                 &fire_watches_block_page)) {
        break;
      }
      uint32_t fire_watches_page = (i << 6) + fire_watches_block_page;
      if (fire_watches_range_start == UINT32_MAX) {
        fire_watches_range_start = fire_watches_page;
      } else {
        FireWatches(fire_watches_range_start, fire_watches_page - 1, false);
        fire_watches_range_start = UINT32_MAX;
      }
      uint64_t fire_watches_block_mask = ~((uint64_t(1) << fire_watches_block_page) - 1);
      fire_watches_block &= fire_watches_block_mask;
      fire_watches_break_block &= fire_watches_block_mask;
    }

    // Add to the GPU-written ranges.
    uint64_t gpu_written_break_block = ~gpu_written_block;
    while (true) {
      uint32_t gpu_written_block_page;
      if (!rex::bit_scan_forward(
              gpu_written_range_start == UINT32_MAX ? gpu_written_block : gpu_written_break_block,
              &gpu_written_block_page)) {
        break;
      }
      uint32_t gpu_written_page = (i << 6) + gpu_written_block_page;
      if (gpu_written_range_start == UINT32_MAX) {
        gpu_written_range_start = gpu_written_page;
      } else {
        uint32_t gpu_written_range_length = gpu_written_page - gpu_written_range_start;
        // Call EnsureHostGpuMemoryAllocated in case the page was marked as
        // GPU-written not as a result to an actual write to the shared memory
        // buffer, but, for instance, by resolving with resolution scaling (to a
        // separate buffer).
        if (EnsureHostGpuMemoryAllocated(gpu_written_range_start << page_size_log2_,
                                         gpu_written_range_length << page_size_log2_)) {
          trace_download_ranges_.push_back(
              std::make_pair(gpu_written_range_start << page_size_log2_,
                             gpu_written_range_length << page_size_log2_));
          trace_download_page_count_ += gpu_written_range_length;
        }
        gpu_written_range_start = UINT32_MAX;
      }
      uint64_t gpu_written_block_mask = ~((uint64_t(1) << gpu_written_block_page) - 1);
      gpu_written_block &= gpu_written_block_mask;
      gpu_written_break_block &= gpu_written_block_mask;
    }
  }
  uint32_t page_count = kBufferSize >> page_size_log2_;
  if (fire_watches_range_start != UINT32_MAX) {
    FireWatches(fire_watches_range_start, page_count - 1, false);
  }
  if (gpu_written_range_start != UINT32_MAX) {
    uint32_t gpu_written_range_length = page_count - gpu_written_range_start;
    if (EnsureHostGpuMemoryAllocated(gpu_written_range_start << page_size_log2_,
                                     gpu_written_range_length << page_size_log2_)) {
      trace_download_ranges_.push_back(std::make_pair(gpu_written_range_start << page_size_log2_,
                                                      gpu_written_range_length << page_size_log2_));
      trace_download_page_count_ += gpu_written_range_length;
    }
  }
}

void SharedMemory::ReleaseTraceDownloadRanges() {
  trace_download_ranges_.clear();
  trace_download_ranges_.shrink_to_fit();
  trace_download_page_count_ = 0;
}

bool SharedMemory::EnsureHostGpuMemoryAllocated(uint32_t start, uint32_t length) {
  if (host_gpu_memory_sparse_granularity_log2_ == UINT32_MAX) {
    return true;
  }
  if (!length) {
    return true;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return false;
  }
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = (start + length - 1) >> page_size_log2_;
  uint32_t allocation_first =
      page_first << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_;
  uint32_t allocation_last =
      page_last << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_;
  while (true) {
    std::pair<size_t, size_t> allocation_range =
        rex::bit::GetNextRangeUnset(host_gpu_memory_sparse_allocated_.data(), allocation_first,
                                    allocation_last - allocation_first + 1);
    if (!allocation_range.second) {
      break;
    }
    if (!AllocateSparseHostGpuMemoryRange(uint32_t(allocation_range.first),
                                          uint32_t(allocation_range.second))) {
      return false;
    }
    rex::bit::SetRange(host_gpu_memory_sparse_allocated_.data(), allocation_range.first,
                       allocation_range.second);
    ++host_gpu_memory_sparse_allocations_;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_allocations",
                      host_gpu_memory_sparse_allocations_);
    host_gpu_memory_sparse_used_bytes_ += uint32_t(allocation_range.second)
                                          << host_gpu_memory_sparse_granularity_log2_;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_used_mb",
                      (host_gpu_memory_sparse_used_bytes_ + ((1 << 20) - 1)) >> 20);
    allocation_first = uint32_t(allocation_range.first + allocation_range.second);
  }
  return true;
}

}  // namespace rex::graphics
