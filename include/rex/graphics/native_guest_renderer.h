#pragma once

#include <array>
#include <cstdint>

#include <rex/graphics/native_rhi.h>

namespace rex::graphics {

enum class NativeGuestOutputBackend : uint32_t {
  kUnknown = 0,
  kD3D12,
  kVulkan,
};

// Per-frame context handed to the registered native guest-output renderer.
// Backend-agnostic: all device access and command recording goes through the
// native-render RHI (rex/graphics/native_rhi.h); the command processor owns
// the nrhi::Device and the per-frame nrhi::Cmd.
struct NativeGuestOutputRenderContext {
  NativeGuestOutputBackend backend = NativeGuestOutputBackend::kUnknown;
  uint32_t guest_output_width = 0;
  uint32_t guest_output_height = 0;
  uint32_t display_width = 0;
  uint32_t display_height = 0;

  // Device-level RHI: resource/pipeline creation, submission counters.
  // Stable across frames for the lifetime of the graphics system.
  nrhi::Device* device = nullptr;
  // Frame-scoped command recording into the command processor's deferred
  // command list. Only valid during the callback.
  nrhi::Cmd* cmd = nullptr;
  // The presenter's guest output image, wrapped as an RHI texture. In
  // nrhi::ResourceState::kGuestOutput at entry (or kCommon the very first
  // frame it exists); must be returned to kGuestOutput before the callback
  // returns true. The pointer is stable while the underlying image is (it
  // changes on output resize).
  nrhi::Texture* guest_output = nullptr;
};

using NativeGuestOutputRenderer = bool (*)(const NativeGuestOutputRenderContext& context,
                                           void* user_data);

void SetNativeGuestOutputRenderer(NativeGuestOutputRenderer renderer, void* user_data);
bool TryRenderNativeGuestOutput(const NativeGuestOutputRenderContext& context);
// Whether a renderer is registered at all; command processors skip RHI
// setup entirely when none is (non-Skate titles / renderer disabled).
bool HasNativeGuestOutputRenderer();

// True while the registered renderer actually replaced the last presented
// frame (false when it yields to the emulated output).
bool IsNativeGuestOutputActive();

// ---- Wide guest output ----------------------------------------------------
// Display aspect the native renderer wants to draw at when it is wider than
// the guest frontbuffer's own aspect (ultrawide). 0 = disabled. Set once at
// app startup; consumed by the command processors when sizing the guest
// output for a swap.
void SetNativeGuestOutputWideAspect(double aspect);
double GetNativeGuestOutputWideAspect();
// Applies the wide aspect to this swap's guest output and display
// dimensions: widens guest_output_width to guest_output_height * aspect and
// makes the display aspect match. Only widens while a renderer is registered
// AND it served the last presented frame, so emulated fallback frames size
// back to the frontbuffer aspect and present pillarboxed by the presenter's
// letterbox path. Returns whether the output was widened; when it was and
// the renderer then yields the frame, the refresh callback must return false
// (keeping the previously presented image) instead of running the emulated
// blit, which writes frontbuffer-aspect content.
bool ApplyNativeGuestOutputWideAspect(uint32_t& guest_output_width, uint32_t guest_output_height,
                                      uint32_t& display_width, uint32_t& display_height);
// native_render_suppress_emulated_draws && IsNativeGuestOutputActive():
// command processors skip emulated draw/resolve execution (memexport draws,
// fences, queries and PM4 parsing still run).
bool ShouldSuppressEmulatedDraws();

// While the native guest-output renderer is active: should the emulated pass
// currently targeting `surface_pitch`-wide surfaces be suppressed? Driven by
// the native_render_suppress_mode cvar; shared by both command processors.
// Draw and resolve suppression must agree: executed passes need their
// resolves, suppressed passes leave garbage EDRAM that must never be copied
// out.
bool ShouldSuppressPassAtPitch(uint32_t surface_pitch);

// Within a suppression-EXEMPT pass: should a draw with no pixel shader
// (depth/stencil-only: shadow casters, z-prepasses) be skipped anyway?
// Driven by native_render_suppress_exempt_depth_only; their output feeds
// only suppressed scene passes (the native renderer shadows itself).
bool ShouldSuppressExemptDepthOnlyDraws();

// ---- Guest-output post-processor ------------------------------------------
// Host effect over the EMULATED guest-output path (e.g. the settings-menu
// backdrop blur): invoked by the command processors at the very end of the
// emulated gamma/FXAA refresh, after the guest output image has been fully
// written, with the same context contract as the renderer callback (the
// image arrives in kGuestOutput state and must be returned to it). When the
// native renderer handled the frame this is NOT invoked; the renderer
// applies its own effects inline. Only invoked while the request flag is set
// (a cheap gate so the common no-effect frame skips the RHI frame setup);
// the post-processor clears the flag itself once its effect has fully
// decayed.
using NativeGuestOutputPostProcessor = void (*)(const NativeGuestOutputRenderContext& context,
                                                void* user_data);
void SetNativeGuestOutputPostProcessor(NativeGuestOutputPostProcessor post_processor,
                                       void* user_data);
bool HasNativeGuestOutputPostProcessor();
void InvokeNativeGuestOutputPostProcessor(const NativeGuestOutputRenderContext& context);
void RequestNativeGuestOutputPostProcess(bool requested);
bool IsNativeGuestOutputPostProcessRequested();

// ---- In-order guest draw replacement --------------------------------------
// Selective replacement of one emulated draw in its original render scope.
// The command processor may first call the matcher with immutable guest draw
// identity before render-target preparation. Matchers needing the exact
// render-pass key can reject that probe and receive the normal late call.
// Only a match pays the cost of opening a borrowed native RHI command scope
// and invoking the renderer.
//
// The first implementation is Vulkan-only. Keeping the contract here avoids
// leaking backend command processor types into title code and leaves room for
// a D3D12 implementation later.
struct NativeGuestDrawContext {
  static constexpr uint32_t kMaxColorAttachments = 4;

  struct VertexFetchIdentity {
    uint32_t physical_address = 0;
    uint32_t byte_count = 0;
    uint32_t endian = 0;
    bool valid = false;

    bool operator==(const VertexFetchIdentity&) const = default;
  };

  NativeGuestOutputBackend backend = NativeGuestOutputBackend::kUnknown;
  // Command-processor frame containing this draw. Unlike title-side swap
  // counters, this is authoritative for asynchronously consumed PM4 work and
  // must be used to keep observer/replacement proofs from spanning frames.
  uint64_t backend_frame_sequence = 0;
  uint64_t vertex_shader_hash = 0;
  uint64_t pixel_shader_hash = 0;
  uint32_t primitive_type = 0;
  // Count submitted by the guest before primitive conversion.
  uint32_t guest_vertex_or_index_count = 0;
  // Count consumed by the host draw after primitive conversion.
  uint32_t vertex_or_index_count = 0;
  // Physical guest address used by DMA-indexed draws. This is a stronger
  // identity than the index count, which is commonly shared by unrelated
  // draws. Auto-indexed draws don't have a guest index buffer.
  uint32_t guest_index_base = 0;
  uint32_t surface_pitch = 0;
  uint32_t render_pass_key = 0;
  bool guest_index_base_valid = false;
  // Effective guest draw state used by the host pipeline. Depth control and
  // color mask are normalized exactly as they are for guest pipeline/render
  // target selection. Color and blend control retain the raw guest register
  // values because alpha test, alpha-to-mask and blend factors are part of a
  // replacement draw's identity.
  uint32_t normalized_depth_control = 0;
  uint32_t normalized_color_mask = 0;
  uint32_t color_control = 0;
  uint32_t blend_control_0 = 0;
  // Raw PA_SU_SC_MODE_CNTL used by the guest pipeline. Cull selection,
  // front-face winding, polygon mode, polygon offset and window-offset
  // behavior are required to prove a native scene pipeline's raster parity.
  uint32_t rasterizer_mode_control = 0;
  // Primitive processing has already resolved whether restart is effective
  // for this topology/index format. The register value is retained for
  // diagnostics even while restart is disabled.
  uint32_t primitive_restart_index = 0;
  bool primitive_restart_enabled = false;
  bool rasterizer_mode_control_valid = false;
  bool draw_state_contract_valid = false;
  // False for the optional early matcher probe, before render targets have
  // been updated. A matcher that requires the exact render-pass identity must
  // reject that probe; it will still receive the normal late call with this
  // true. The renderer is only invoked after the exact key and borrowed scope
  // are available.
  bool render_pass_key_valid = false;
  bool indexed = false;

  // Raw guest vertex-fetch bindings used by native scene observers to
  // disambiguate repeated geometry submitted with different live palettes.
  // These are copied values from the register file, never guest pointers.
  VertexFetchIdentity primary_vertex_fetch{};
  VertexFetchIdentity palette_vertex_fetch{};

  // Exact host attachment signature of the guest render scope being
  // borrowed. This is unavailable during the optional early matcher probe;
  // matchers requiring pipeline compatibility must reject that probe and
  // wait for the late call. Values are translated from the actual backend
  // render scope, never reconstructed from the title's render-pass key.
  std::array<nrhi::Format, kMaxColorAttachments> color_attachment_formats{};
  uint32_t color_attachment_count = 0;
  nrhi::Format depth_attachment_format = nrhi::Format::kUnknown;
  nrhi::Format stencil_attachment_format = nrhi::Format::kUnknown;
  uint32_t sample_count = 1;
  uint64_t sample_mask = UINT64_MAX;
  bool borrowed_attachment_contract_valid = false;

  // Null while matching. Valid only for the renderer invocation.
  nrhi::Device* device = nullptr;
  nrhi::Cmd* cmd = nullptr;
};

// Read-only diagnostic emitted immediately before the selective replacement
// eligibility gate. This exposes why a real guest draw did or did not reach a
// NativeGuestDrawMatcher without making title-specific shader hashes part of
// the backend. Observers must remain cheap and must not mutate GPU state.
struct NativeGuestDrawEligibilityContext {
  NativeGuestOutputBackend backend = NativeGuestOutputBackend::kUnknown;
  uint64_t backend_frame_sequence = 0;
  uint64_t vertex_shader_hash = 0;
  uint64_t pixel_shader_hash = 0;
  uint32_t primitive_type = 0;
  // Count submitted by the guest before primitive conversion.
  uint32_t guest_vertex_or_index_count = 0;
  // Count consumed by the host draw after primitive conversion.
  uint32_t vertex_or_index_count = 0;
  uint32_t guest_index_base = 0;
  uint32_t processed_index_buffer_type = 0;
  bool processed_index_buffer_present = false;
  bool shader_32bit_index_dma = false;
  bool memexport_writes_possible = false;
  bool native_rhi_device_available = false;
  bool draw_replacer_available = false;
  bool host_render_targets = false;
  bool eligible = false;
};

using NativeGuestDrawEligibilityObserver = void (*)(
    const NativeGuestDrawEligibilityContext& context, void* user_data);
void SetNativeGuestDrawEligibilityObserver(
    NativeGuestDrawEligibilityObserver observer, void* user_data);
bool HasNativeGuestDrawEligibilityObserver();
void ObserveNativeGuestDrawEligibility(
    const NativeGuestDrawEligibilityContext& context);

using NativeGuestDrawMatcher = bool (*)(const NativeGuestDrawContext& context,
                                        void* user_data);
// Returning false requests the original guest draw as a fallback. A renderer
// must therefore return false before recording any draw commands; command
// recording is not transactional and cannot be rolled back.
using NativeGuestDrawRenderer = bool (*)(const NativeGuestDrawContext& context,
                                         void* user_data);

void SetNativeGuestDrawReplacer(NativeGuestDrawMatcher matcher,
                                NativeGuestDrawRenderer renderer,
                                void* user_data);
bool HasNativeGuestDrawReplacer();
bool MatchesNativeGuestDraw(const NativeGuestDrawContext& context);
bool TryReplaceNativeGuestDraw(const NativeGuestDrawContext& context);

}  // namespace rex::graphics
