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

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rex/math.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>
#include <rex/string.h>

#if REX_PLATFORM_ANDROID
#include <string.h>

#include <dlfcn.h>
#include <sys/ioctl.h>

#include <linux/ashmem.h>

// TODO(tomc): Android or maybe na. idk
// #include "xenia/base/main_android.h"
#endif

#if REX_PLATFORM_MAC || REX_PLATFORM_PS5
#define ftruncate64 ftruncate
#define mmap64 mmap
#endif

#if REX_PLATFORM_PS5
#include <ps5platform/kernel.h>
#endif

namespace rex {
namespace memory {

#if REX_PLATFORM_PS5
// Native-title direct arena, adapted from holdmysocks/mcla-recomp (GPL-3.0-or-later).
namespace {
constexpr FileMappingHandle kDirectMemoryHandle = 0x44495243;
constexpr size_t kDirectAlignment = 0x200000;
int64_t direct_start = -1;
size_t direct_length = 0;
std::atomic<uint64_t> protect_calls{0};
std::atomic<uint64_t> protect_site_calls[kPs5ProtectSiteCount]{};
}
#endif

namespace {

#if REX_PLATFORM_MAC || REX_PLATFORM_PS5
void AlignHostPageRange(void*& base_address, size_t& length) {
  const uintptr_t page_mask = uintptr_t(page_size() - 1);
  const uintptr_t start = reinterpret_cast<uintptr_t>(base_address);
  const uintptr_t end = start + length;
  const uintptr_t aligned_start = start & ~page_mask;
  const uintptr_t aligned_end = (end + page_mask) & ~page_mask;
  base_address = reinterpret_cast<void*>(aligned_start);
  length = static_cast<size_t>(aligned_end - aligned_start);
}
#endif

}  // namespace

// Convert filesystem path to valid shm_open name (must start with /, no other slashes)
static std::string MakeShmName(const std::filesystem::path& path) {
  std::string name = path.string();
  for (char& c : name) {
    if (c == '/')
      c = '_';
  }
  if (name.empty() || name[0] != '/') {
    name.insert(name.begin(), '/');
  }
  return name;
}

#if REX_PLATFORM_ANDROID
// May be null if no dynamically loaded functions are required.
static void* libandroid_;
// API 26+.
static int (*android_ASharedMemory_create_)(const char* name, size_t size);

void AndroidInitialize() {
  if (rex::GetAndroidApiLevel() >= 26) {
    libandroid_ = dlopen("libandroid.so", RTLD_NOW);
    assert_not_null(libandroid_);
    if (libandroid_) {
      android_ASharedMemory_create_ = reinterpret_cast<decltype(android_ASharedMemory_create_)>(
          dlsym(libandroid_, "ASharedMemory_create"));
      assert_not_null(android_ASharedMemory_create_);
    }
  }
}

void AndroidShutdown() {
  android_ASharedMemory_create_ = nullptr;
  if (libandroid_) {
    dlclose(libandroid_);
    libandroid_ = nullptr;
  }
}
#endif

size_t page_size() {
  return getpagesize();
}
size_t allocation_granularity() {
  return page_size();
}

uint32_t ToPosixProtectFlags(PageAccess access) {
  switch (access) {
    case PageAccess::kNoAccess:
      return PROT_NONE;
    case PageAccess::kReadOnly:
      return PROT_READ;
    case PageAccess::kReadWrite:
      return PROT_READ | PROT_WRITE;
    case PageAccess::kExecuteReadOnly:
      // Statically recompiled guest code executes from the host image, never
      // from guest pages, so execute requests are served without PROT_EXEC.
      // This keeps the process W^X: no path through this layer can create
      // executable memory.
      return PROT_READ;
    case PageAccess::kExecuteReadWrite:
      return PROT_READ | PROT_WRITE;
    default:
      assert_unhandled_case(access);
      return PROT_NONE;
  }
}

// TODO(tomc): this needs to go somewhere else. we should utilize the platform namespace more.
#if REX_PLATFORM_LINUX
namespace {

struct LinuxMapEntry {
  uintptr_t start = 0;
  uintptr_t end = 0;
  char perms[5] = {};
};

// Parse a line from /proc/self/maps into a LinuxMapEntry
static bool ParseProcMapsLine(const std::string& line, LinuxMapEntry& out) {
  out = LinuxMapEntry{};
  unsigned long long start = 0, end = 0;
  char perms[5] = {};
  const int matched = std::sscanf(line.c_str(), "%llx-%llx %4s", &start, &end, perms);
  if (matched < 3)
    return false;
  out.start = static_cast<uintptr_t>(start);
  out.end = static_cast<uintptr_t>(end);
  std::memcpy(out.perms, perms, sizeof(out.perms));
  return out.start < out.end;
}

// Find the mapping entry in /proc/self/maps that contains the given address
static bool FindEntryForAddress(void* address, LinuxMapEntry& out_entry) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(address);
  std::ifstream maps("/proc/self/maps");
  if (!maps.is_open())
    return false;
  std::string line;
  while (std::getline(maps, line)) {
    LinuxMapEntry e;
    if (!ParseProcMapsLine(line, e))
      continue;
    if (addr >= e.start && addr < e.end) {
      out_entry = e;
      return true;
    }
  }
  return false;
}

// Check if [base, base+length) is fully covered by existing mappings (no gaps)
static bool IsRangeFullyMapped(void* base_address, size_t length) {
  if (!base_address || length == 0)
    return false;

  const uintptr_t begin = reinterpret_cast<uintptr_t>(base_address);
  const uintptr_t end = begin + length;
  if (end < begin) {  // overflow check
    return false;
  }

  std::ifstream maps("/proc/self/maps");
  if (!maps.is_open())
    return false;

  uintptr_t cursor = begin;
  std::string line;
  while (std::getline(maps, line)) {
    LinuxMapEntry e;
    if (!ParseProcMapsLine(line, e))
      continue;
    if (e.end <= cursor)
      continue;
    if (e.start > cursor)
      return false;  // gap found
    cursor = e.end;
    if (cursor >= end)
      return true;
  }
  return cursor >= end;
}

// Convert /proc/self/maps permission chars to PageAccess
static PageAccess PermsToPageAccess(const char perms[5]) {
  const bool r = perms[0] == 'r';
  const bool w = perms[1] == 'w';
  const bool x = perms[2] == 'x';

  if (!r && !w && !x)
    return PageAccess::kNoAccess;
  if (x)
    return w ? PageAccess::kExecuteReadWrite : PageAccess::kExecuteReadOnly;
  return w ? PageAccess::kReadWrite : PageAccess::kReadOnly;
}

}  // namespace
#endif  // REX_PLATFORM_LINUX

void* AllocFixed(void* base_address, size_t length, AllocationType allocation_type,
                 PageAccess access) {
  // Emulates Windows VirtualAlloc behavior:
  // - Reserve: create PROT_NONE mapping to hold address space
  // - Commit on existing reservation: mprotect to enable access (EEXIST path)
  // - New allocation: mmap with MAP_FIXED_NOREPLACE (never silently replace)
  const uint32_t prot_requested = ToPosixProtectFlags(access);

  // Determine initial protection based on allocation type
  int prot_initial = 0;
  switch (allocation_type) {
    case AllocationType::kReserve:
      prot_initial = PROT_NONE;
      break;
    case AllocationType::kCommit:
    case AllocationType::kReserveCommit:
    default:
      prot_initial = static_cast<int>(prot_requested);
      break;
  }

#if REX_PLATFORM_MAC || REX_PLATFORM_PS5
  if (base_address &&
      (allocation_type == AllocationType::kCommit ||
       allocation_type == AllocationType::kReserveCommit)) {
    void* protect_base = base_address;
    size_t protect_length = length;
    AlignHostPageRange(protect_base, protect_length);
    if (mprotect(protect_base, protect_length, static_cast<int>(prot_requested)) == 0) {
      return base_address;
    }
#if REX_PLATFORM_PS5
    // Never replace an existing guest alias with anonymous memory on failure.
    return nullptr;
#endif
  }
#endif

  // Build flags - always use MAP_FIXED_NOREPLACE for fixed addresses
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_FIXED_NOREPLACE)
  if (base_address) {
    flags |= MAP_FIXED_NOREPLACE;
  }
#else
  if (base_address) {
    flags |= MAP_FIXED;
  }
#endif

  void* result = mmap(base_address, length, prot_initial, flags, -1, 0);
  if (result != MAP_FAILED) {
    return result;
  }
#if defined(MAP_FIXED_NOREPLACE) && REX_PLATFORM_LINUX
  // Handle EEXIST: address already has a mapping (e.g., from prior Reserve)
  // This is the "commit on existing reservation" path
  if (errno == EEXIST && base_address &&
      (allocation_type == AllocationType::kCommit ||
       allocation_type == AllocationType::kReserveCommit)) {
    // Verify the entire range is mapped before using mprotect
    if (IsRangeFullyMapped(base_address, length)) {
      if (mprotect(base_address, length, static_cast<int>(prot_requested)) == 0) {
        return base_address;
      }
    }
  }
#endif

  return nullptr;
}

bool DeallocFixed(void* base_address, size_t length, DeallocationType deallocation_type) {
  switch (deallocation_type) {
    case DeallocationType::kDecommit: {
      // Decommit: remove access first, then release physical pages
#if REX_PLATFORM_MAC || REX_PLATFORM_PS5
      AlignHostPageRange(base_address, length);
#endif
      if (mprotect(base_address, length, PROT_NONE) != 0) {
        return false;
      }
#if defined(MADV_DONTNEED)
      (void)madvise(base_address, length, MADV_DONTNEED);
#endif
      return true;
    }
    case DeallocationType::kRelease: {
      return munmap(base_address, length) == 0;
    }
    default:
      // how we get here? :(
      assert_always();
      return false;
  }
}

bool Protect(void* base_address, size_t length, PageAccess access, PageAccess* out_old_access) {
  if (out_old_access) {
    *out_old_access = PageAccess::kNoAccess;
  }

#if REX_PLATFORM_LINUX
  // NOTE(tomc): we may want to look at doing this differently. it should work for now
  //             but there is a TOCTOU window between reading and changing.
  //             This really shouldn't be an issue since VirtualProtect on Windows isn't truly
  //             atomic in a mutli-threaded process either, but it's something to be aware of.
  // Query old access before changing, if the caller needs it
  if (out_old_access) {
    LinuxMapEntry e;
    if (FindEntryForAddress(base_address, e)) {
      *out_old_access = PermsToPageAccess(e.perms);
    }
  }
#endif

  uint32_t prot = ToPosixProtectFlags(access);
#if REX_PLATFORM_MAC || REX_PLATFORM_PS5
  AlignHostPageRange(base_address, length);
#endif
#if REX_PLATFORM_PS5
  protect_calls.fetch_add(1, std::memory_order_relaxed);
  const int site = Ps5ProtectSite();
  if (site >= 0 && site < kPs5ProtectSiteCount) protect_site_calls[site].fetch_add(1, std::memory_order_relaxed);
#endif
  return mprotect(base_address, length, prot) == 0;
}

bool QueryProtect(void* base_address, size_t& length, PageAccess& access_out) {
#if !REX_PLATFORM_LINUX
  access_out = PageAccess::kNoAccess;
  length = 0;
  return false;
#else
  access_out = PageAccess::kNoAccess;
  length = 0;

  LinuxMapEntry e;
  if (!FindEntryForAddress(base_address, e)) {
    return false;
  }

  const uintptr_t addr = reinterpret_cast<uintptr_t>(base_address);
  length = static_cast<size_t>(e.end - addr);
  access_out = PermsToPageAccess(e.perms);

  return true;
#endif
}

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path& path, size_t length,
                                          PageAccess access, bool commit) {
#if REX_PLATFORM_ANDROID
  // TODO(Triang3l): Check if memfd can be used instead on API 30+.
  if (android_ASharedMemory_create_) {
    int sharedmem_fd = android_ASharedMemory_create_(path.c_str(), length);
    return sharedmem_fd >= 0 ? static_cast<FileMappingHandle>(sharedmem_fd)
                             : kFileMappingHandleInvalid;
  }

  // Use /dev/ashmem on API versions below 26, which added ASharedMemory.
  // /dev/ashmem was disabled on API 29 for apps targeting it.
  // https://chromium.googlesource.com/chromium/src/+/master/third_party/ashmem/ashmem-dev.c
  int ashmem_fd = open("/" ASHMEM_NAME_DEF, O_RDWR);
  if (ashmem_fd < 0) {
    return kFileMappingHandleInvalid;
  }
  char ashmem_name[ASHMEM_NAME_LEN];
  strlcpy(ashmem_name, path.c_str(), rex::countof(ashmem_name));
  if (ioctl(ashmem_fd, ASHMEM_SET_NAME, ashmem_name) < 0 ||
      ioctl(ashmem_fd, ASHMEM_SET_SIZE, length) < 0) {
    close(ashmem_fd);
    return kFileMappingHandleInvalid;
  }
  return static_cast<FileMappingHandle>(ashmem_fd);
#else
  int oflag;
  switch (access) {
    case PageAccess::kNoAccess:
      oflag = 0;
      break;
    case PageAccess::kReadOnly:
    case PageAccess::kExecuteReadOnly:
      oflag = O_RDONLY;
      break;
    case PageAccess::kReadWrite:
    case PageAccess::kExecuteReadWrite:
      oflag = O_RDWR;
      break;
    default:
      assert_always();
      return kFileMappingHandleInvalid;
  }
  oflag |= O_CREAT;
#if REX_PLATFORM_PS5
  // An installed title's flexible-memory budget cannot hold the 4.5 GiB arena.
  // Use direct memory, which supports the same shared aliases without that budget.
  if (direct_start >= 0 || length > SIZE_MAX - (kDirectAlignment - 1)) {
    return kFileMappingHandleInvalid;
  }
  const size_t bytes = (length + kDirectAlignment - 1) & ~(kDirectAlignment - 1);
  int64_t start = -1;
  const int32_t status = sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes,
      kDirectAlignment, PS5_KERNEL_DIRECT_TYPE_CPU, &start);
  if (status != 0) {
    std::fprintf(stderr, "PS5 guest arena allocation failed: bytes=0x%zx status=0x%x\n", bytes, unsigned(status));
    return kFileMappingHandleInvalid;
  }
  direct_start = start;
  direct_length = bytes;
  return kDirectMemoryHandle;
#else
  auto full_path = MakeShmName(path);
  int ret = shm_open(full_path.c_str(), oflag, 0777);
  if (ret < 0) {
    return kFileMappingHandleInvalid;
  }
  if (ftruncate64(ret, static_cast<off_t>(length)) != 0) {
    close(ret);
    shm_unlink(full_path.c_str());
    return kFileMappingHandleInvalid;
  }
  return static_cast<FileMappingHandle>(ret);
#endif  // PS5 direct-memory arena
#endif
}

void CloseFileMappingHandle(FileMappingHandle handle, const std::filesystem::path& path) {
#if REX_PLATFORM_PS5
  if (handle == kDirectMemoryHandle) {
    if (direct_start >= 0) {
      const int32_t status = sceKernelReleaseDirectMemory(direct_start, direct_length);
      if (status != 0) std::fprintf(stderr, "PS5 arena release failed: 0x%x\n", unsigned(status));
      else { direct_start = -1; direct_length = 0; }
    }
    return;
  }
#endif
  close(static_cast<int>(handle));
#if !REX_PLATFORM_ANDROID && !REX_PLATFORM_PS5
  auto full_path = MakeShmName(path);
  shm_unlink(full_path.c_str());
#endif
}

void* MapFileView(FileMappingHandle handle, void* base_address, size_t length, PageAccess access,
                  size_t file_offset) {
  // file_offset must be page-aligned
  const size_t page = page_size();
  if (file_offset % page != 0) {
    return nullptr;
  }

  int flags = MAP_SHARED;

  // For file views, we need MAP_FIXED to replace existing reservations.
  // The emulator reserves address space first, then maps file views into it.
  // MAP_FIXED_NOREPLACE would fail with EEXIST in this case.
  if (base_address) {
    flags |= MAP_FIXED;
  }

  uint32_t prot = ToPosixProtectFlags(access);
#if REX_PLATFORM_PS5
  if (handle == kDirectMemoryHandle) {
    if (direct_start < 0 || file_offset > direct_length || length > direct_length - file_offset) return nullptr;
    void* address = base_address;
    const int32_t status = sceKernelMapDirectMemory(&address, length, int(prot),
        base_address ? PS5_KERNEL_MAP_FIXED : 0, direct_start + int64_t(file_offset), page);
    if (status != 0) return nullptr;
    if (base_address && address != base_address) {
      sceKernelMunmap(address, length);
      return nullptr;
    }
    return address;
  }
#endif
  void* result = mmap64(base_address, length, prot, flags, static_cast<int>(handle),
                        static_cast<off_t>(file_offset));
  if (result == MAP_FAILED) {
    return nullptr;
  }

  // Verify we got the address we asked for
  if (base_address && result != base_address) {
    munmap(result, length);
    return nullptr;
  }

  return result;
}

bool UnmapFileView(FileMappingHandle handle, void* base_address, size_t length) {
#if REX_PLATFORM_PS5
  if (handle == kDirectMemoryHandle) return sceKernelMunmap(base_address, length) == 0;
#endif
  return munmap(base_address, length) == 0;
}

#if REX_PLATFORM_PS5
void* ReserveFileMappingRange(FileMappingHandle handle, size_t length) {
  if (handle != kDirectMemoryHandle) return nullptr;
  void* address = reinterpret_cast<void*>(uintptr_t{0x1000000000ull});
  const int32_t status = sceKernelReserveVirtualRange(&address, length, 0, PS5_KERNEL_DIRECT_ALIGNMENT);
  if (status != 0) return nullptr;
  const uintptr_t start = reinterpret_cast<uintptr_t>(address);
  if (start < 0x300000000ull && start + length > 0x200000000ull) {
    sceKernelMunmap(address, length);
    return nullptr;
  }
  return address;
}
bool ReleaseFileMappingRange(FileMappingHandle handle, void* address, size_t length) {
  return handle == kDirectMemoryHandle && sceKernelMunmap(address, length) == 0;
}
int& Ps5ProtectSite() { static thread_local int site = 0; return site; }
uint64_t Ps5ProtectCallCount() { return protect_calls.load(std::memory_order_relaxed); }
uint64_t Ps5ProtectCallCount(int site) {
  return site >= 0 && site < kPs5ProtectSiteCount ? protect_site_calls[site].load(std::memory_order_relaxed) : 0;
}
#endif

}  // namespace memory
}  // namespace rex
