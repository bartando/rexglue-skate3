/**
 * @file        rex/ui/overlay/fullscreen_hint.h
 *
 * @brief       Short-lived banner telling how to leave fullscreen.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#pragma once
#include <chrono>

#include <rex/ui/imgui_dialog.h>

namespace rex::ui {

// Shown for a few seconds whenever the window goes fullscreen, since there is
// no title bar left to close or restore it with.
class FullscreenHintDialog : public ImGuiDialog {
 public:
  explicit FullscreenHintDialog(ImGuiDrawer* imgui_drawer);

  void Show();
  bool WantsContinuousRepaint() const override { return IsDrawActive(); }

 protected:
  void OnDraw(ImGuiIO& io) override;

 private:
  std::chrono::steady_clock::time_point shown_at_;
};

}  // namespace rex::ui
