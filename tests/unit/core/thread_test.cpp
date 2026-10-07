/**
 * @file        tests/unit/core/thread_test.cpp
 * @brief       Unit tests for rex::thread::Thread
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <rex/thread.h>

using rex::thread::Thread;

TEST_CASE("rex::thread::Thread - resume right after a suspended create", "[thread]") {
  // Resuming immediately races the new thread's own startup, which only loses
  // when that thread is preempted at the wrong moment, so oversubscribe the
  // cores. A lost resume parks the thread forever, hence the bounded wait.
  constexpr int kIterations = 300;
  const unsigned creators = std::max(4u, std::thread::hardware_concurrency() * 2);
  std::atomic<int> ran{0};
  std::atomic<int> stuck{0};
  std::vector<std::thread> workers;
  for (unsigned c = 0; c < creators; ++c) {
    workers.emplace_back([&] {
      for (int i = 0; i < kIterations; ++i) {
        Thread::CreationParameters params;
        params.stack_size = 64 * 1024;
        params.create_suspended = true;
        auto thread = Thread::Create(params, [&ran] { ran.fetch_add(1); });
        if (!thread) {
          continue;
        }
        thread->Resume();
        if (rex::thread::Wait(thread.get(), false, std::chrono::seconds(2)) !=
            rex::thread::WaitResult::kSuccess) {
          stuck.fetch_add(1);
          thread->Resume();
          rex::thread::Wait(thread.get(), false);
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  CHECK(stuck.load() == 0);
  CHECK(ran.load() == int(creators) * kIterations);
}
