/**
 * @file        ui/overlay/fullscreen_hint.cpp
 *
 * @brief       Short-lived banner telling how to leave fullscreen. See
 *              fullscreen_hint.h for details.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#include <rex/ui/overlay/fullscreen_hint.h>

#include <algorithm>

#include <imgui.h>

#include <rex/platform.h>

namespace rex::ui {
namespace {

constexpr float kVisibleSeconds = 4.0f;
constexpr float kFadeSeconds = 0.6f;

#if REX_PLATFORM_WIN32
constexpr const char* kHintText = "F11 or Alt+Enter: windowed      Alt+F4: quit";
#else
constexpr const char* kHintText = "F11 or Alt+Enter: windowed";
#endif

}  // namespace

FullscreenHintDialog::FullscreenHintDialog(ImGuiDrawer* imgui_drawer)
    : ImGuiDialog(imgui_drawer) {
  SetDrawActive(false);
}

void FullscreenHintDialog::Show() {
  shown_at_ = std::chrono::steady_clock::now();
  SetDrawActive(true);
}

void FullscreenHintDialog::OnDraw(ImGuiIO& io) {
  const float age =
      std::chrono::duration<float>(std::chrono::steady_clock::now() - shown_at_).count();
  if (age >= kVisibleSeconds) {
    SetDrawActive(false);
    return;
  }
  const float alpha = std::clamp((kVisibleSeconds - age) / kFadeSeconds, 0.0f, 1.0f);

  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.04f),
                          ImGuiCond_Always, ImVec2(0.5f, 0.0f));
  ImGui::SetNextWindowBgAlpha(0.6f);
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(18.0f, 10.0f));
  ImGui::PushFont(nullptr, 20.0f);
  if (ImGui::Begin("##fullscreen_hint", nullptr,
                   ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                       ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted(kHintText);
  }
  ImGui::End();
  ImGui::PopFont();
  ImGui::PopStyleVar(3);
}

}  // namespace rex::ui
