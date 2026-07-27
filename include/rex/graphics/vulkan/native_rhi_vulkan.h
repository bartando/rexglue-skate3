#pragma once

// Vulkan implementation of the native-render RHI (rex/graphics/native_rhi.h).
// Constructed and owned by the Vulkan command processor; built on the CP's
// deferred command buffer, its barrier queue and its submission counters.
// Unlike the D3D12 backend (a thin passthrough), this backend owns all the
// translation the interface promises: render-pass scoping, descriptor sets
// per the frozen set/binding plan, explicit-state -> (stage, access,
// layout) mapping, and the
// negative-viewport y-flip.

#include <cstdint>

#include <rex/graphics/native_rhi.h>
#include <rex/ui/vulkan/api.h>

namespace rex::graphics::vulkan {

class VulkanCommandProcessor;

// Exact attachment signature of an already-open guest render scope that NRHI
// may borrow for one replacement draw. The caller obtains these values from
// the guest render-target cache after it has updated and entered the render
// scope.
//
// color_attachment_count is the highest active color slot plus one, so
// VK_FORMAT_UNDEFINED entries preserve holes in the guest MRT layout.
// Dynamic rendering uses the attachment formats and leaves render_pass null.
// Classic rendering supplies the exact render pass handle; the formats still
// describe its subpass and are used to reject incompatible cached pipelines.
struct NativeRhiBorrowedRenderScopeDesc {
  static constexpr uint32_t kMaxColorAttachments = 4;

  bool dynamic_rendering = false;
  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkFormat color_attachment_formats[kMaxColorAttachments] = {};
  uint32_t color_attachment_count = 0;
  VkFormat depth_attachment_format = VK_FORMAT_UNDEFINED;
  VkFormat stencil_attachment_format = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT;
  // Usually UINT64_MAX. The host-render-target 2x fallback uses Vulkan 4x
  // rasterization with samples 0 and 3 enabled (0b1001), matching the guest
  // pipeline cache.
  uint64_t sample_mask = UINT64_MAX;
};

// Converts a backend attachment format into the backend-independent NRHI
// contract exposed to title-side replacement matchers.
nrhi::Format NativeRhiFormatFromVkFormat(VkFormat format);

// Creates the device wrapper (call once; destroy with DestroyNativeRhiDevice
// after all GPU work completed - AwaitAllQueueOperationsCompletion).
nrhi::Device* CreateNativeRhiDevice(VulkanCommandProcessor* command_processor);
void DestroyNativeRhiDevice(nrhi::Device* device);

// Per-frame, inside the guest-output refresher (submission open): wraps the
// presenter's guest output image (cached by VkImage identity), drains
// completed retirement, advances the internal root-constant ring region and
// resets the frame Cmd latches. guest_output_ever_written_previously selects
// the wrapper's initial tracked layout (UNDEFINED on the image's first ever
// write, the presenter's internal SHADER_READ_ONLY_OPTIMAL otherwise).
// guest_output_out receives the wrapped texture.
nrhi::Cmd* NativeRhiBeginFrame(nrhi::Device* device, VkImage guest_output_image,
                               VkImageView guest_output_image_view,
                               bool guest_output_ever_written_previously, uint32_t width,
                               uint32_t height, nrhi::Texture** guest_output_out);

// Vulkan-only frame epilogue: ends any render pass still open in the frame
// Cmd, flushes pending clears that no draw consumed, and submits the command
// processor's queued barriers (including the app's release barrier of the
// guest output back to nrhi::ResourceState::kGuestOutput). Call after the
// native render callback, before EndSubmission.
void NativeRhiEndFrame(nrhi::Device* device);

// Borrows the guest render scope currently open in the command processor.
// This does not begin/end a frame or render pass, submit barriers, transition
// attachments, or alter render targets. It only resets NRHI's per-draw binding
// latches and makes Cmd drawing/binding operations target the existing scope.
//
// Replacement pipeline variants are cached lazily for the exact borrowed
// attachment signature. The caller must prepare all buffers/textures and their
// resource states before the guest scope is entered. SetRenderTargets, clears,
// copies and barriers are rejected while borrowed. End must be called before
// the command processor resumes guest drawing.
nrhi::Cmd* NativeRhiBeginBorrowedRenderScope(
    nrhi::Device* device, const NativeRhiBorrowedRenderScopeDesc& desc);
void NativeRhiEndBorrowedRenderScope(nrhi::Device* device);

}  // namespace rex::graphics::vulkan
