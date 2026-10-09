#include <rex/net/socket.h>
#include <rex/platform.h>

static_assert(REX_PLATFORM_LINUX || REX_PLATFORM_MAC || REX_PLATFORM_PS5, "This file is POSIX-only");

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstring>

namespace rex::net {

int socket_close(SocketHandle handle) {
  return close(static_cast<int>(handle));
}

int socket_ioctl(SocketHandle handle, uint32_t cmd, uint8_t* arg) {
#if REX_PLATFORM_PS5
  // The title sandbox refuses socket ioctls with EACCES. FIONBIO is the one
  // titles rely on; the guest's u32 flag is big-endian, so test it as raw bytes.
  constexpr uint32_t kFionbio = 0x8004667E;
  if (cmd == kFionbio) {
    uint32_t enable;
    std::memcpy(&enable, arg, sizeof(enable));
    int fd = static_cast<int>(handle);
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0) {
      return -1;
    }
    flags = enable ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, flags);
  }
#endif
  return ioctl(static_cast<int>(handle), cmd, arg);
}

}  // namespace rex::net
