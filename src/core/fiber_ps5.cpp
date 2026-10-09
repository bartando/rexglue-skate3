/**
 * @file        core/fiber_ps5.cpp
 * @brief       Fiber support for PlayStation 5 homebrew: thread fibers only.
 *
 * Every guest thread converts itself to a fiber when it starts, so that much
 * has to exist. Creating further fibers and switching between them is done
 * with getcontext/makecontext/swapcontext on the other POSIX platforms; on the
 * console the saved context does not have the layout the payload SDKs declare
 * (see exception_handler_ps5.cpp), so those calls are not used here. A title
 * that creates fibers needs a hand-written context switch instead.
 */

#include <rex/platform.h>
#if REX_PLATFORM_PS5

#include <rex/thread/fiber.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

namespace rex::thread {

thread_local Fiber* Fiber::tls_current_ = nullptr;

Fiber* Fiber::ConvertCurrentThread() {
  auto* f = new Fiber();
  f->is_thread_fiber_ = true;
  tls_current_ = f;
  return f;
}

Fiber* Fiber::Create(size_t stack_size, void (*entry)(void*), void* arg) {
  (void)stack_size;
  (void)entry;
  (void)arg;
  std::fputs("rex::thread::Fiber::Create is not implemented on PS5\n", stderr);
  return nullptr;
}

void Fiber::SwitchTo(Fiber* target) {
  (void)target;
  std::fputs("rex::thread::Fiber::SwitchTo is not implemented on PS5\n", stderr);
  std::abort();
}

void Fiber::Destroy() {
  if (is_thread_fiber_) {
    tls_current_ = nullptr;
  } else {
    assert(this != tls_current_ && "Destroy called on the currently running fiber");
  }
  delete this;
}

}  // namespace rex::thread

#endif  // REX_PLATFORM_PS5
