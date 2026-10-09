/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <vector>
#include <string>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/kernel/xam/module.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xsocket.h>
// #include <rex/system/xnet.h>

#include <rex/net/socket.h>

// Standard socket types used by Xbox API emulation
#if REX_PLATFORM_WIN32
#include <WinSock2.h>

#include <WS2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <poll.h>
#include <sys/socket.h>
#if REX_PLATFORM_MAC || REX_PLATFORM_PS5
#ifdef IPPROTO_TCP
#undef IPPROTO_TCP
#endif
#ifdef IPPROTO_UDP
#undef IPPROTO_UDP
#endif
#endif
#endif

REXCVAR_DEFINE_INT32(net_instance, 0, "Network",
                     "Instance number when several titles share a host for system link "
                     "(0-7): picks the host ports of the guest's ports below 1024")
    .range(0, 7);

namespace rex::system {

uint32_t LocalInstance() {
  return uint32_t(REXCVAR_GET(net_instance));
}

namespace {

struct Association {
  uint32_t peer_address;
  uint32_t instance;
  uint64_t key;
};
std::mutex associations_mutex;
std::vector<Association> associations;

// Datagrams of an association start with this tag and the session key.
constexpr uint32_t kAssociationMagic = 0x52584E4B;  // "RXNK"
constexpr size_t kAssociationHeaderSize = 12;

uint32_t NativeErrorToWsa() {
#if REX_PLATFORM_WIN32
  return uint32_t(WSAGetLastError());
#else
  switch (errno) {
    case EWOULDBLOCK:
#if EAGAIN != EWOULDBLOCK
    case EAGAIN:
#endif
      return 10035;  // WSAEWOULDBLOCK
    case EINPROGRESS:
      return 10036;
    case EALREADY:
      return 10037;
    case ENOTSOCK:
      return 10038;
    case EMSGSIZE:
      return 10040;
    case ENOPROTOOPT:
      return 10042;
    case EAFNOSUPPORT:
      return 10047;
    case EADDRINUSE:
      return 10048;
    case EADDRNOTAVAIL:
      return 10049;
    case ENETUNREACH:
      return 10051;
    case ECONNABORTED:
      return 10053;
    case ECONNRESET:
      return 10054;
    case ENOBUFS:
      return 10055;
    case EISCONN:
      return 10056;
    case ENOTCONN:
      return 10057;
    case ETIMEDOUT:
      return 10060;
    case ECONNREFUSED:
      return 10061;
    case EHOSTUNREACH:
      return 10065;
    case EACCES:
      return 10013;
    default:
      return 10022;  // WSAEINVAL
  }
#endif
}

// A native IPv4 address from a host port and address (numbers).
sockaddr_in NativeHostAddress(uint16_t host_port, uint32_t address) {
  sockaddr_in native = {};
#if REX_PLATFORM_MAC || REX_PLATFORM_PS5
  native.sin_len = sizeof(native);
#endif
  native.sin_family = AF_INET;
  native.sin_port = htons(host_port);
  native.sin_addr.s_addr = htonl(address);
  return native;
}

// The same from a guest port, on this instance's ports.
sockaddr_in NativeAddress(uint16_t guest_port, uint32_t address) {
  return NativeHostAddress(GuestToHostPort(guest_port), address);
}

}  // namespace

bool IsVirtualAddress(uint32_t address) {
  return (address & 0xFF000000) == kVirtualAddressBase;
}

uint32_t AssociationAddress(uint32_t peer_address, uint32_t instance, uint64_t key) {
  std::lock_guard<std::mutex> lock(associations_mutex);
  for (size_t i = 0; i < associations.size(); ++i) {
    const Association& association = associations[i];
    if (association.peer_address == peer_address && association.instance == instance &&
        association.key == key) {
      return kVirtualAddressBase + uint32_t(i) + 1;
    }
  }
  associations.push_back({peer_address, instance, key});
  const uint32_t address = kVirtualAddressBase + uint32_t(associations.size());
  REXSYS_INFO("XNet association {}.{}.{}.{} instance {} key {:016X} -> 1.0.0.{}",
              peer_address >> 24, (peer_address >> 16) & 0xFF, (peer_address >> 8) & 0xFF,
              peer_address & 0xFF, instance, key, address & 0xFFFFFF);
  return address;
}

bool ResolveAssociation(uint32_t virtual_address, uint32_t* peer_address, uint32_t* instance,
                        uint64_t* key) {
  if (!IsVirtualAddress(virtual_address)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(associations_mutex);
  const uint32_t index = (virtual_address & 0xFFFFFF) - 1;
  if (index >= associations.size()) {
    return false;
  }
  *peer_address = associations[index].peer_address;
  *instance = associations[index].instance;
  *key = associations[index].key;
  return true;
}

void XSocket::CountTraffic(bool sent, uint32_t address) {
  ++(sent ? sent_by_address_ : received_by_address_)[address];
  const uint64_t now = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
  if (now - traffic_summary_ms_ < 5000) {
    return;
  }
  traffic_summary_ms_ = now;
  const auto describe = [](const std::map<uint32_t, uint32_t>& counts) {
    std::string text;
    for (const auto& [peer, count] : counts) {
      text += fmt::format(" {}.{}.{}.{}={}", peer >> 24, (peer >> 16) & 0xFF, (peer >> 8) & 0xFF,
                          peer & 0xFF, count);
    }
    return text.empty() ? std::string(" -") : text;
  };
  REXSYS_INFO("socket {:X} traffic: sent{} | received{}", handle(), describe(sent_by_address_),
              describe(received_by_address_));
  sent_by_address_.clear();
  received_by_address_.clear();
}

int XSocket::IoFlags(uint32_t flags) const {
#if REX_PLATFORM_WIN32
  return int(flags);
#else
  return int(flags) | (non_blocking_ ? MSG_DONTWAIT : 0);
#endif
}

int XSocket::Track(int result) {
  last_wsa_error_ = result < 0 ? NativeErrorToWsa() : 0;
  return result;
}

XSocket::XSocket(KernelState* kernel_state) : XObject(kernel_state, kObjectType) {}

XSocket::XSocket(KernelState* kernel_state, uint64_t native_handle)
    : XObject(kernel_state, kObjectType), native_handle_(native_handle) {}

XSocket::~XSocket() {
  Close();
}

X_STATUS XSocket::Initialize(AddressFamily af, Type type, Protocol proto) {
  af_ = af;
  type_ = type;
  proto_ = proto;

  if (proto == Protocol::IPPROTO_VDP) {
    // VDP is a layer on top of UDP.
    proto = Protocol::IPPROTO_UDP;
  }

  native_handle_ = socket(af, type, proto);
  if (native_handle_ == -1) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::Close() {
  REXSYS_INFO("socket {:X}: close", handle());
  int ret = rex::net::socket_close(native_handle_);
  if (ret != 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::SetOption(uint32_t level, uint32_t optname, void* optval_ptr, uint32_t optlen) {
  if (level == 0xFFFF && (optname == 0x5801 || optname == 0x5802)) {
    // Disable socket encryption
    secure_ = false;
    return X_STATUS_SUCCESS;
  }

  if (level == 0xFFFF && optname == 0x5803) {
    return X_STATUS_SUCCESS;  // SO_GRANTINSECURE: no XNet security here
  }
  // Winsock numbering and big-endian values on the guest side.
  int native_level = int(level);
  int native_name = int(optname);
  if (level == 0xFFFF) {
    native_level = SOL_SOCKET;
    switch (optname) {
      case 0x0004:
        native_name = SO_REUSEADDR;
        break;
      case 0x0008:
        native_name = SO_KEEPALIVE;
        break;
      case 0x0020:
        native_name = SO_BROADCAST;
        break;
      case 0x1001:
        native_name = SO_SNDBUF;
        break;
      case 0x1002:
        native_name = SO_RCVBUF;
        break;
      default:
        REXSYS_WARN("setsockopt: unsupported socket option {:04X}, ignored", optname);
        return X_STATUS_SUCCESS;
    }
  }
  int value = 0;
  if (optlen >= 4) {
    value = int(rex::byte_swap(*reinterpret_cast<uint32_t*>(optval_ptr)));
  } else if (optlen == 1) {
    value = *reinterpret_cast<uint8_t*>(optval_ptr);
  }
  REXSYS_INFO("socket {:X}: setsockopt level {:X} option {:X} = {}", handle(), level, optname,
              value);
  int ret = Track(setsockopt(native_handle_, native_level, native_name,
                             reinterpret_cast<char*>(&value), sizeof(value)));
  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  // SO_BROADCAST
  if (level == 0xFFFF && optname == 0x0020) {
    broadcast_socket_ = true;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::IOControl(uint32_t cmd, uint8_t* arg_ptr) {
  constexpr uint32_t kFionbio = 0x8004667E;
  if (cmd == kFionbio) {
    non_blocking_ = arg_ptr && rex::byte_swap(*reinterpret_cast<uint32_t*>(arg_ptr)) != 0;
  }
  int ret = Track(rex::net::socket_ioctl(native_handle_, cmd, arg_ptr));
#if !REX_PLATFORM_WIN32
  if (ret < 0 && cmd == kFionbio) {
    return X_STATUS_SUCCESS;  // Emulated through IoFlags.
  }
#endif
  if (ret < 0) {
    // TODO: Get last error
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::Connect(N_XSOCKADDR* name, int name_len) {
  // sa_data holds the big-endian port and address of an AF_INET address.
  uint16_t port;
  uint32_t address;
  std::memcpy(&port, name->sa_data, sizeof(port));
  std::memcpy(&address, name->sa_data + 2, sizeof(address));
  sockaddr_in native = NativeAddress(ntohs(port), ntohl(address));
  int ret = Track(connect(native_handle_, reinterpret_cast<sockaddr*>(&native), sizeof(native)));
  const uint32_t target = ntohl(address);
  REXSYS_INFO("socket {:X}: connect {}.{}.{}.{}:{} -> {} (WSA {})", handle(), target >> 24,
              (target >> 16) & 0xFF, (target >> 8) & 0xFF, target & 0xFF, ntohs(port), ret,
              last_wsa_error_);
  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::Bind(N_XSOCKADDR_IN* name, int name_len) {
  (void)name_len;
  sockaddr_in native = NativeAddress(name->sin_port, name->sin_addr);
  int ret = Track(bind(native_handle_, reinterpret_cast<sockaddr*>(&native), sizeof(native)));
  if (ret < 0) {
    REXSYS_WARN("bind to port {} (host {}) failed: WSA error {}", uint16_t(name->sin_port),
                ntohs(native.sin_port), last_wsa_error_);
    return X_STATUS_UNSUCCESSFUL;
  }

  REXSYS_INFO("socket {:X}: bound to port {} (host {}), type {} protocol {}", handle(),
              uint16_t(name->sin_port), ntohs(native.sin_port), uint32_t(type_),
              uint32_t(proto_));
  bound_ = true;
  bound_port_ = name->sin_port;

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::GetSockName(N_XSOCKADDR_IN* name) {
  sockaddr_in native = {};
  socklen_t native_len = sizeof(native);
  if (Track(getsockname(native_handle_, reinterpret_cast<sockaddr*>(&native), &native_len)) < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }
  name->sin_family = AF_INET;
  name->sin_port = HostToGuestPort(ntohs(native.sin_port));
  name->sin_addr = ntohl(native.sin_addr.s_addr);
  std::memset(name->x_sin_zero, 0, sizeof(name->x_sin_zero));
  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::Listen(int backlog) {
  int ret = Track(listen(native_handle_, backlog));
  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

object_ref<XSocket> XSocket::Accept(N_XSOCKADDR* name, int* name_len) {
#if !REX_PLATFORM_WIN32
  if (non_blocking_) {
    pollfd pending = {int(native_handle_), POLLIN, 0};
    if (poll(&pending, 1, 0) <= 0) {
      last_wsa_error_ = 10035;  // WSAEWOULDBLOCK
      std::memset(name, 0, *name_len);
      *name_len = 0;
      return nullptr;
    }
  }
#endif
  sockaddr_in n_sockaddr = {};
  socklen_t n_name_len = sizeof(n_sockaddr);
  intptr_t ret = Track(int(accept(native_handle_, reinterpret_cast<sockaddr*>(&n_sockaddr),
                                  &n_name_len)));
  if (ret == -1) {
    std::memset(name, 0, *name_len);
    *name_len = 0;
    return nullptr;
  }

  // Guest layout: AF_INET, then the big-endian port and address.
  name->address_family = AF_INET;
  const uint16_t port = htons(HostToGuestPort(ntohs(n_sockaddr.sin_port)));
  std::memcpy(name->sa_data, &port, sizeof(port));
  std::memcpy(name->sa_data + 2, &n_sockaddr.sin_addr.s_addr, sizeof(uint32_t));
  *name_len = sizeof(sockaddr_in);

  // Create a kernel object to represent the new socket, and copy parameters
  // over.
  auto socket = object_ref<XSocket>(new XSocket(kernel_state_, ret));
  socket->af_ = af_;
  socket->type_ = type_;
  socket->proto_ = proto_;

  return socket;
}

int XSocket::Shutdown(int how) {
  return shutdown(native_handle_, how);
}

int XSocket::Recv(uint8_t* buf, uint32_t buf_len, uint32_t flags) {
  return Track(int(recv(native_handle_, reinterpret_cast<char*>(buf), buf_len, IoFlags(flags))));
}

int XSocket::RecvFrom(uint8_t* buf, uint32_t buf_len, uint32_t flags, N_XSOCKADDR_IN* from,
                      uint32_t* from_len) {
  // Pop from secure packets first
  // TODO(DrChat): Enable when I commit XNet
  /*
  {
    std::lock_guard<std::mutex> lock(incoming_packet_mutex_);
    if (incoming_packets_.size()) {
      packet* pkt = (packet*)incoming_packets_.front();
      int data_len = pkt->data_len;
      std::memcpy(buf, pkt->data, std::min((uint32_t)pkt->data_len, buf_len));

      from->sin_family = 2;
      from->sin_addr = pkt->src_ip;
      from->sin_port = pkt->src_port;

      incoming_packets_.pop();
      uint8_t* pkt_ui8 = (uint8_t*)pkt;
      delete[] pkt_ui8;

      return data_len;
    }
  }
  */

  sockaddr_in nfrom;
  socklen_t nfromlen = sizeof(sockaddr_in);
  receive_buffer_.resize(size_t(buf_len) + kAssociationHeaderSize);
  int ret = Track(int(recvfrom(native_handle_, reinterpret_cast<char*>(receive_buffer_.data()),
                               receive_buffer_.size(), IoFlags(flags), (sockaddr*)&nfrom,
                               &nfromlen)));
  uint32_t source_address = ntohl(nfrom.sin_addr.s_addr);
  const uint32_t source_instance = InstanceOfHostPort(ntohs(nfrom.sin_port));
  if (ret >= int(kAssociationHeaderSize) &&
      rex::byte_swap(*reinterpret_cast<uint32_t*>(receive_buffer_.data())) == kAssociationMagic) {
    // A peer's datagram for one session: it arrives from that association.
    const uint64_t key = rex::byte_swap(*reinterpret_cast<uint64_t*>(receive_buffer_.data() + 4));
    source_address = AssociationAddress(source_address, source_instance, key);
    ret -= int(kAssociationHeaderSize);
    std::memcpy(buf, receive_buffer_.data() + kAssociationHeaderSize,
                std::min<size_t>(size_t(ret), buf_len));
  } else if (ret > 0) {
    std::memcpy(buf, receive_buffer_.data(), std::min<size_t>(size_t(ret), buf_len));
    if (source_instance != LocalInstance()) {
      // Another instance, maybe on this host: replies have to find its ports.
      source_address = AssociationAddress(source_address, source_instance, 0);
    }
  }
  ret = std::min(ret, int(buf_len));
  ++receive_calls_;
  if (receive_calls_ % 256 == 1) {
    REXSYS_INFO("socket {:X}: recvfrom call {}", handle(), receive_calls_);
  }
  if (ret < 0 && last_wsa_error_ != 10035 && logged_errors_ < 10) {
    ++logged_errors_;
    REXSYS_WARN("socket {:X}: recvfrom failed: WSA {} (errno {})", handle(), last_wsa_error_,
                errno);
  }
  if (ret >= 0 && logged_receives_ < 40) {
    ++logged_receives_;
    const uint32_t source = source_address;
    std::string head;
    for (int i = 0; i < std::min(ret, 8); ++i) {
      head += fmt::format("{:02X}", buf[i]);
    }
    REXSYS_INFO("socket {:X}: received {} bytes from {}.{}.{}.{}:{} [{}]", handle(), ret,
                source >> 24, (source >> 16) & 0xFF, (source >> 8) & 0xFF, source & 0xFF,
                HostToGuestPort(ntohs(nfrom.sin_port)), head);
  }
  if (ret >= 0) {
    CountTraffic(false, source_address);
  }
  if (from && ret >= 0) {
    from->sin_family = AF_INET;
    from->sin_addr = source_address;
    from->sin_port = HostToGuestPort(ntohs(nfrom.sin_port));
    std::memset(from->x_sin_zero, 0, sizeof(from->x_sin_zero));
  }

  if (from_len) {
    *from_len = nfromlen;
  }

  return ret;
}

int XSocket::Send(const uint8_t* buf, uint32_t buf_len, uint32_t flags) {
  const int ret = Track(
      int(send(native_handle_, reinterpret_cast<const char*>(buf), buf_len, IoFlags(flags))));
  if (logged_sends_ < 40) {
    ++logged_sends_;
    REXSYS_INFO("socket {:X}: sent {} bytes (connected) -> {} (WSA {})", handle(), buf_len, ret,
                last_wsa_error_);
  }
  return ret;
}

int XSocket::SendTo(uint8_t* buf, uint32_t buf_len, uint32_t flags, N_XSOCKADDR_IN* to,
                    uint32_t to_len) {
  // Send 2 copies of the packet: One to XNet (for network security) and an
  // unencrypted copy for other Xenia hosts.
  // TODO(DrChat): Enable when I commit XNet.
  /*
  auto xam = kernel_state()->GetKernelModule<xam::XamModule>("xam.xex");
  auto xnet = xam->xnet();
  if (xnet) {
    xnet->SendPacket(this, to, buf, buf_len);
  }
  */

  (void)to_len;
  sockaddr_in nto = {};
  const uint8_t* payload = buf;
  size_t payload_len = buf_len;
  std::vector<uint8_t> tagged;
  uint32_t destination_instance = 0;
  if (to) {
    uint32_t address = to->sin_addr;
    uint32_t peer_address;
    uint64_t key = 0;
    if (ResolveAssociation(address, &peer_address, &destination_instance, &key) && key) {
      tagged.resize(kAssociationHeaderSize + buf_len);
      const uint32_t magic = rex::byte_swap(kAssociationMagic);
      const uint64_t key_be = rex::byte_swap(key);
      std::memcpy(tagged.data(), &magic, sizeof(magic));
      std::memcpy(tagged.data() + 4, &key_be, sizeof(key_be));
      std::memcpy(tagged.data() + kAssociationHeaderSize, buf, buf_len);
      payload = tagged.data();
      payload_len = tagged.size();
    }
    if (IsVirtualAddress(address)) {
      address = peer_address;
    }
    nto = NativeHostAddress(GuestToHostPort(to->sin_port, destination_instance), address);
  }
  int ret = Track(int(sendto(native_handle_, reinterpret_cast<const char*>(payload), payload_len,
                             IoFlags(flags), to ? reinterpret_cast<sockaddr*>(&nto) : nullptr,
                             to ? socklen_t(sizeof(nto)) : 0)));
  if (to && to->sin_addr == INADDR_BROADCAST && to->sin_port < 1024) {
    // Broadcasts reach every instance's ports.
    for (uint32_t instance = 1; instance < kMaxInstances; ++instance) {
      sockaddr_in other =
          NativeHostAddress(GuestToHostPort(to->sin_port, instance), INADDR_BROADCAST);
      sendto(native_handle_, reinterpret_cast<const char*>(payload), payload_len, IoFlags(flags),
             reinterpret_cast<sockaddr*>(&other), sizeof(other));
    }
  }
  if (ret > 0 && !tagged.empty()) {
    ret = int(buf_len);
  }
  if (to && ret >= 0) {
    CountTraffic(true, to->sin_addr);
  }
  if (to && logged_sends_ < 40) {
    ++logged_sends_;
    const uint32_t target = to->sin_addr;
    REXSYS_INFO("socket {:X}: sent {} bytes to {}.{}.{}.{}:{} -> {} (WSA {})", handle(), buf_len,
                target >> 24, (target >> 16) & 0xFF, (target >> 8) & 0xFF, target & 0xFF,
                uint16_t(to->sin_port), ret, last_wsa_error_);
  }
  return ret;
}

bool XSocket::QueuePacket(uint32_t src_ip, uint16_t src_port, const uint8_t* buf, size_t len) {
  packet* pkt = reinterpret_cast<packet*>(new uint8_t[sizeof(packet) + len]);
  pkt->src_ip = src_ip;
  pkt->src_port = src_port;

  pkt->data_len = (uint16_t)len;
  std::memcpy(pkt->data, buf, len);

  std::lock_guard<std::mutex> lock(incoming_packet_mutex_);
  incoming_packets_.push((uint8_t*)pkt);

  // TODO: Limit on number of incoming packets?
  return true;
}

}  // namespace rex::system
