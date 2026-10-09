#pragma once
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/memory.h>
#include <rex/thread/mutex.h>

namespace rex::graphics {

// Manages memory for unconverted textures, resolve targets, vertex and index
// buffers that can be accessed from shaders with Xenon physical addresses, with
// system page size granularity.
class SharedMemory {
 public:
  static constexpr uint32_t kBufferSizeLog2 = 29;
  static constexpr uint32_t kBufferSize = 1 << kBufferSizeLog2;

  virtual ~SharedMemory();
  // Call in the implementation-specific ClearCache.
  virtual void ClearCache();
  void SetSystemPageBlocksValidWithGpuDataWritten();
  void InvalidateAllPages();

  typedef void (*GlobalWatchCallback)(const std::unique_lock<std::recursive_mutex>& global_lock,
                                      void* context, uint32_t address_first, uint32_t address_last,
                                      bool invalidated_by_gpu);
  typedef void* GlobalWatchHandle;
  // Registers a callback invoked when something is invalidated in the GPU
  // memory copy by the CPU or (if triggered explicitly - such as by a resolve)
  // by the GPU. It will be fired for writes to pages previously requested, but
  // may also be fired regardless of whether it was used by GPU emulation - for
  // example, if the game changes protection level of a memory range containing
  // the watched range.
  //
  // The callback is called within the global critical region.
  GlobalWatchHandle RegisterGlobalWatch(GlobalWatchCallback callback, void* callback_context);
  void UnregisterGlobalWatch(GlobalWatchHandle handle);
  typedef void (*WatchCallback)(const std::unique_lock<std::recursive_mutex>& global_lock,
                                void* context, void* data, uint64_t argument,
                                bool invalidated_by_gpu);
  typedef void* WatchHandle;
  // Registers a callback invoked when the specified memory range is invalidated
  // in the GPU memory copy by the CPU or (if triggered explicitly - such as by
  // a resolve) by the GPU. It will be fired for writes to pages previously
  // requested, but may also be fired regardless of whether it was used by GPU
  // emulation - for example, if the game changes protection level of a memory
  // range containing the watched range.
  //
  // Generally the context is the subsystem pointer (for example, the texture
  // cache), the data is the object (such as a texture), and the argument is
  // additional subsystem/object-specific data (such as whether the range
  // belongs to the base mip level or to the rest of the mips).
  //
  // Called with the global critical region locked. Do NOT watch or unwatch
  // ranges from within it! The watch for the callback is cancelled after the
  // callback - the handle becomes invalid.
  WatchHandle WatchMemoryRange(uint32_t start, uint32_t length, WatchCallback callback,
                               void* callback_context, void* callback_data,
                               uint64_t callback_argument);
  // Unregisters previously registered watched memory range.
  void UnwatchMemoryRange(WatchHandle handle);

  // Checks if the range has been updated, uploads new data if needed and
  // ensures the host GPU memory backing the range are resident. Returns true if
  // the range has been fully updated and is usable.
  bool RequestRanges(const std::pair<uint32_t, uint32_t>* ranges, size_t count);
  bool RequestRange(uint32_t start, uint32_t length);
  // Under the shared-memory global critical region, verifies exact guest
  // bytes, refreshes residency, then verifies once more before releasing the
  // lock. Used by narrow deferred-replay guards where unrelated writes on the
  // same host page must not invalidate an unchanged payload.
  bool RequestRangeIfBytesEqual(uint32_t start, uint32_t length,
                                const uint8_t* expected);
  // Observer-only exact comparison against CPU guest physical memory. Unlike
  // RequestRangeIfBytesEqual, this never allocates sparse GPU backing,
  // uploads, inserts barriers, or ends a render pass.
  bool GuestBytesEqual(uint32_t start, uint32_t length,
                       const uint8_t* expected);
  bool CopyGuestBytes(uint32_t start, uint32_t length,
                      std::vector<uint8_t>& bytes_out);

  const uint8_t* DebugTranslatePhysical(uint32_t address) const {
    return memory_.TranslatePhysical<const uint8_t*>(address);
  }

  // Marks the range and, if not exact_range, potentially its surroundings
  // (to up to the first GPU-written page, as an access violation exception
  // count optimization) as modified by the CPU, also invalidating GPU-written
  // pages directly in the range.
  std::pair<uint32_t, uint32_t> MemoryInvalidationCallback(uint32_t physical_address_start,
                                                           uint32_t length, bool exact_range);

  // Marks the range as containing GPU-generated data (such as resolves),
  // triggering modification callbacks, making it valid (so pages are not
  // copied from the main memory until they're modified by the CPU) and
  // protecting it. Before writing anything from the GPU side, RequestRange must
  // be called, to make sure, if the GPU writes don't overwrite *everything* in
  // the pages they touch, the CPU data is properly loaded to the unmodified
  // regions in those pages.
  void RangeWrittenByGpu(uint32_t start, uint32_t length);

  // Records that GPU work in the current submission reads or writes the range.
  // Requests and GPU writes are recorded automatically; callers that read the
  // buffer without requesting (cached residency) must note the access
  // themselves. A backend may only move an upload ahead of earlier work in the
  // submission if none of that work touched the uploaded pages.
  void NoteGpuAccess(uint32_t start, uint32_t length);

  uint32_t page_size_log2() const { return page_size_log2_; }
  // Inclusive address range whose watches are being fired. Only meaningful
  // inside a watch callback.
  uint32_t firing_address_first() const { return firing_address_first_; }
  uint32_t firing_address_last() const { return firing_address_last_; }
  const uint8_t* TranslatePhysical(uint32_t address) const;
  void ResetGpuAccessWindow();
  // Advances the frame index that hot-page detection counts streaks in. In
  // zero-copy mode also fires the watches on hot pages (see zero_copy_).
  void OnFrameEnd();
  bool zero_copy() const { return zero_copy_; }

 protected:
  // Whether moving an upload ahead of earlier work in the window could change
  // what that work read. new_pages are (page, staged contents) pairs. A page
  // touched earlier is only safe if its previous contents are known, it wasn't
  // uploaded in order earlier in the window, and the bytes the earlier work
  // accessed are identical.
  bool UploadChangesAccessedBytes(
      const std::vector<std::pair<uint32_t, const uint8_t*>>& new_pages) const;
  // Records the new contents of uploaded pages. Pages uploaded in order rather
  // than hoisted can't be hoisted again in the window, since a later hoisted
  // copy would land before them.
  void NoteUploadedPages(const std::vector<std::pair<uint32_t, const uint8_t*>>& pages,
                         bool hoisted);

  // The buffer is guest memory itself (VulkanSharedMemory zero copy): nothing
  // is uploaded, requests are no-ops, and only watches protect pages.
  bool zero_copy_ = false;
  // Zero copy still write-protects watched pages (textures). Pages the CPU
  // rewrites every frame turn hot: their watches no longer protect them and
  // instead fire once per frame, saving a fault and two protection changes per
  // page and frame. A texture there sees a mid-frame CPU write a frame late.
  void EnableZeroCopyHotPages();
  void set_cpu_invalidation_widen_pages(uint32_t pages) { cpu_invalidation_widen_pages_ = pages; }

  SharedMemory(memory::Memory& memory);
  // Call in implementation-specific initialization.
  void InitializeCommon();
  void InitializeSparseHostGpuMemory(uint32_t granularity_log2);
  // Call last in implementation-specific shutdown, also callable from the
  // destructor.
  void ShutdownCommon();

  // Sparse allocations are 4 MB, so not too many of them are allocated, but
  // also not to waste too much memory for padding (with 16 MB there's too
  // much).
  static constexpr uint32_t kHostGpuMemoryOptimalSparseAllocationLog2 = 22;
  static_assert(kHostGpuMemoryOptimalSparseAllocationLog2 <= kBufferSizeLog2);

  memory::Memory& memory() const { return memory_; }



  uint32_t host_gpu_memory_sparse_granularity_log2() const {
    return host_gpu_memory_sparse_granularity_log2_;
  }

  // Allocations in the host buffer are aligned the same way as in the guest
  // physical memory (for instance, if an allocation is 64 KB, it can represent
  // 0-64 KB, 64-128 KB, 128-192 KB in the guest memory, and so on, but not
  // something like 16-80 KB. This is assumed by the rules for texture data
  // access in the texture cache.
  virtual bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                uint32_t length_allocations);

  // Mark the memory range as updated and protect it.
  void MakeRangeValid(uint32_t start, uint32_t length, bool written_by_gpu);

  // Uploads a range of host pages - only called if host GPU sparse memory
  // allocation succeeded if needed. While uploading, MakeRangeValid must be
  // called for each successfully uploaded range as early as possible, before
  // the memcpy, to make sure invalidation that happened during the CPU -> GPU
  // memcpy isn't missed (upload_page_ranges is in pages because of this -
  // MakeRangeValid has page granularity). upload_page_ranges are sorted in
  // ascending address order, so front and back can be used to determine the
  // overall bounds of pages to be uploaded.
  virtual bool UploadRanges(
      const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) = 0;

  const std::vector<std::pair<uint32_t, uint32_t>>& trace_download_ranges() {
    return trace_download_ranges_;
  }
  uint32_t trace_download_page_count() const { return trace_download_page_count_; }
  // Fills trace_download_ranges() and trace_download_page_count() with
  // GPU-written ranges that need to be downloaded, and also invalidates
  // non-GPU-written ranges so only the needed data - not the all the collected
  // data - will be written in the trace. trace_download_page_count() will be 0
  // if nothing to download.
  void PrepareForTraceDownload();
  // Release memory used for trace download ranges, to be called after
  // downloading or in cases when download is dropped.
  void ReleaseTraceDownloadRanges();

 private:
  memory::Memory& memory_;

  // Log2 of invalidation granularity (the system page size, but the dependency
  // on it is not hard - the access callback takes a range as an argument, and
  // touched pages of the buffer of this size will be invalidated).
  uint32_t page_size_log2_;
  uint32_t cpu_invalidation_widen_pages_ = 64;
  uint32_t firing_address_first_ = 0;
  uint32_t firing_address_last_ = 0;

  bool EnsureHostGpuMemoryAllocated(uint32_t start, uint32_t length);
  bool RequestRangeUntracked(uint32_t start, uint32_t length);
  bool RequestRangesUntracked(const std::pair<uint32_t, uint32_t>* ranges, size_t count);

  std::vector<uint64_t> window_accessed_pages_;
  // Accessed pages are either fully accessed or have sorted, disjoint
  // [first, end) intervals of accessed byte offsets. Gaps matter: guest
  // resource headers the CPU keeps updating often share pages with the data.
  std::vector<uint64_t> window_fully_accessed_pages_;
  std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>
      window_partial_page_access_;
  std::vector<uint64_t> window_inline_uploaded_pages_;
  bool window_accessed_any_ = false;
  bool window_inline_uploaded_any_ = false;
  void NotePartialPageAccess(uint32_t page, uint32_t offset_first, uint32_t offset_end);
  // CPU copy of what the last upload of each page put in the buffer, valid
  // until the GPU writes the page. Allocated in 64-page chunks on first upload:
  // a single 512 MB reservation fails in a PS5 title's memory budget. Capped
  // by shared_memory_upload_shadow_max_mb; past the cap the least recently
  // used chunk is freed. Losing a page's shadow only makes uploads to it
  // non-hoistable and hot-page checks upload instead of compare.
  std::vector<std::unique_ptr<uint8_t[]>> upload_shadow_chunks_;
  std::vector<uint32_t> upload_shadow_chunk_used_frame_;
  size_t upload_shadow_chunk_count_ = 0;
  size_t upload_shadow_max_chunks_ = SIZE_MAX;
  std::vector<uint64_t> upload_shadow_valid_;
  // The shadow of a page, or nullptr if its chunk is not allocated.
  const uint8_t* UploadShadowPage(uint32_t page) const;
  uint8_t* AllocateUploadShadowPage(uint32_t page);
  void NoteUploadShadowUse(uint32_t page) {
    upload_shadow_chunk_used_frame_[page >> 6] = frame_index_.load(std::memory_order_relaxed);
  }
  void EvictUploadShadowChunk(size_t keep_chunk);
  size_t upload_shadow_chunk_bytes() const { return size_t(64) << page_size_log2_; }

  // Hot pages (shared_memory_hot_pages). A page the CPU write-faults on in
  // kHotPageStreak consecutive frames stops being write-protected and is never
  // marked valid. Requests compare the bytes they need against the upload
  // shadow instead, and upload the page only if they differ. This removes the
  // fault, unprotect, upload and re-protect cycle for data rewritten every
  // frame, which costs ~33 us per protection change on the PS5.
  //
  // A watch registered over a hot page demotes it: the page is protected again
  // and checked against the shadow before the next request, firing watches if
  // it changed. GPU writes demote it too, since the shadow no longer matches.
  static constexpr uint8_t kHotPageStreak = 4;
  bool hot_pages_enabled_ = false;
  std::atomic<uint32_t> frame_index_{1};
  std::vector<uint64_t> hot_pages_;
  // Pages a watch demoted. They never become hot again: textures reloaded
  // every frame would otherwise demote and re-promote them in a loop, paying a
  // protection change each time.
  std::vector<uint64_t> hot_excluded_pages_;
  std::vector<uint32_t> cpu_invalidation_frame_;
  std::vector<uint8_t> cpu_invalidation_streak_;
  std::vector<uint32_t> pending_demotion_checks_;
  std::vector<std::pair<uint32_t, uint32_t>> hot_filtered_ranges_;
  void NoteCpuInvalidation(uint32_t page_first, uint32_t page_last);
  void DemoteHotPages(uint32_t page_first, uint32_t page_last);
  void CheckDemotedPages();
  // Drops hot pages from upload_ranges_ whose requested bytes match the
  // upload shadow. requested are byte ranges.
  void FilterUnchangedHotPages(const std::pair<uint32_t, uint32_t>* requested, size_t count);
  bool IsHotPage(uint32_t page) const {
    return hot_tracking_ && ((hot_pages_[page >> 6] >> (page & 63)) & 1);
  }
  // Hot-page streaks are tracked (shared_memory_hot_pages or zero copy).
  bool hot_tracking_ = false;
  uint32_t host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;
  std::vector<uint64_t> host_gpu_memory_sparse_allocated_;
  uint32_t host_gpu_memory_sparse_allocations_ = 0;
  uint32_t host_gpu_memory_sparse_used_bytes_ = 0;

  void* memory_invalidation_callback_handle_ = nullptr;
  void* memory_data_provider_handle_ = nullptr;

  // Ranges that need to be uploaded, generated by GetRangesToUpload (a
  // persistently allocated vector).
  std::vector<std::pair<uint32_t, uint32_t>> upload_ranges_;
  std::vector<std::pair<uint32_t, uint32_t>> widened_upload_ranges_;
  // Grows upload_ranges_ over neighbouring invalid pages inside the CPU
  // invalidation widening window, so a window opened by one write fault is
  // uploaded and write-protected again with one Protect call instead of one
  // per page. On the PS5 a call costs ~33 us however many pages it covers.
  void WidenUploadRanges();

  // GPU-written memory downloading for traces. <Start address, length>.
  std::vector<std::pair<uint32_t, uint32_t>> trace_download_ranges_;
  uint32_t trace_download_page_count_ = 0;

  // Mutex between the guest memory subsystem and the command processor, to be
  // locked when checking or updating validity of pages/ranges and when firing
  // watches.
  rex::thread::global_critical_region global_critical_region_;

  // ***************************************************************************
  // Things below should be fully protected by global_critical_region.
  // ***************************************************************************

  // Double-buffered valid-page flags for lockless checks in RequestRanges.
  std::vector<uint64_t> valid_buffer_a_;
  std::vector<uint64_t> valid_buffer_b_;
  std::atomic<uint64_t*> active_valid_flags_{nullptr};
  std::atomic<uint64_t*> staging_valid_flags_{nullptr};
  // Subset of valid pages containing data written by the GPU.
  std::vector<uint64_t> system_page_flags_valid_and_gpu_written_;
  // Dirty state tracking for frame-end page-state refresh.
  std::atomic<bool> gpu_written_data_dirty_{false};
  std::atomic<uint32_t> dirty_blocks_{0};
  uint32_t num_system_page_flags_ = 0;

  static std::pair<uint32_t, uint32_t> MemoryInvalidationCallbackThunk(
      void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range);

  struct GlobalWatch {
    GlobalWatchCallback callback;
    void* callback_context;
  };
  std::vector<GlobalWatch*> global_watches_;
  struct WatchNode;
  // Watched range placed by other GPU subsystems.
  struct WatchRange {
    union {
      struct {
        WatchCallback callback;
        void* callback_context;
        void* callback_data;
        uint64_t callback_argument;
        WatchNode* node_first;
        uint32_t address_first;
        uint32_t address_last;
        uint32_t page_first;
        uint32_t page_last;
      };
      WatchRange* next_free;
    };
  };
  // Node for faster checking of watches when pages have been written to - all
  // 512 MB are split into smaller equally sized buckets, and then ranges are
  // linearly checked.
  struct WatchNode {
    union {
      struct {
        WatchRange* range;
        // Link to another node of this watched range in the next bucket.
        WatchNode* range_node_next;
        // Links to nodes belonging to other watched ranges in the bucket.
        WatchNode* bucket_node_previous;
        WatchNode* bucket_node_next;
      };
      WatchNode* next_free;
    };
  };
  static constexpr uint32_t kWatchBucketSizeLog2 = 22;
  static constexpr uint32_t kWatchBucketCount = 1 << (kBufferSizeLog2 - kWatchBucketSizeLog2);
  WatchNode* watch_buckets_[kWatchBucketCount] = {};
  // Allocation from pools - taking new WatchRanges and WatchNodes from the free
  // list, and if there are none, creating a pool if the current one is fully
  // used, and linearly allocating from the current pool.
  static constexpr uint32_t kWatchRangePoolSize = 8192;
  static constexpr uint32_t kWatchNodePoolSize = 8192;
  std::vector<WatchRange*> watch_range_pools_;
  std::vector<WatchNode*> watch_node_pools_;
  uint32_t watch_range_current_pool_allocated_ = 0;
  uint32_t watch_node_current_pool_allocated_ = 0;
  WatchRange* watch_range_first_free_ = nullptr;
  WatchNode* watch_node_first_free_ = nullptr;
  uint32_t gpu_exact_watch_writes_since_report_ = 0;
  uint64_t gpu_exact_watch_hits_since_report_ = 0;
  uint64_t gpu_page_only_watch_hits_since_report_ = 0;
  uint64_t gpu_page_only_watch_filtered_since_report_ = 0;
  // Triggers the watches (global and per-range), removing triggered range
  // watches. Exact byte bounds are optional and currently supplied only for
  // explicit GPU writes; CPU invalidation stays conservatively page-based.
  void FireWatches(uint32_t page_first, uint32_t page_last, bool invalidated_by_gpu,
                   bool exact_address_range_valid = false, uint32_t exact_address_first = 0,
                   uint32_t exact_address_last = 0);
  // Unlinks and frees the range and its nodes. Call this in the global critical
  // region.
  void UnlinkWatchRange(WatchRange* range);
};

}  // namespace rex::graphics
