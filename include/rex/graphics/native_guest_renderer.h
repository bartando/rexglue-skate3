#pragma once

#include <array>
#include <cstdint>
#include <vector>

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
  // Authoritative command-processor frame whose tokens remain executable
  // during this callback. Title-side swap counters are not equivalent.
  uint64_t backend_frame_sequence = 0;
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

// Timing probe (see native_render_main_pass_suppression_benchmark). True only
// when the probe cvar is set and this is a framebuffer-sized pass. Independent
// of whether a native renderer is serving, unlike ShouldSuppressEmulatedDraws.
bool ShouldSuppressMainPassForBenchmark(uint32_t surface_pitch);

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

  // Raw guest render-target registers and their decoded identity at the draw
  // point. The raw values are retained so title observers can verify the
  // decode against the documented Xenos bit layout instead of treating a
  // backend render-pass cache key as guest target identity.
  struct RenderTargetState {
    uint32_t rb_color_info_0 = 0;
    uint32_t rb_depth_info = 0;
    uint32_t rb_surface_info = 0;
    uint32_t rb_modecontrol = 0;
    uint32_t color_edram_base = 0;
    uint32_t depth_edram_base = 0;
    uint32_t surface_pitch = 0;
    uint32_t edram_mode = 0;
    bool valid = false;

    bool operator==(const RenderTargetState&) const = default;
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
  RenderTargetState render_target_state{};

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

// Read-only state captured after guest pipeline, render-target, viewport,
// scissor and texture-fetch setup, immediately before draw submission. Unlike
// the selective replacement context, this is emitted for every host-render-
// target draw, including auto-indexed and fullscreen draws. It owns values
// only: no command list, backend object or live guest pointer is exposed.
struct NativeGuestDrawStateContext {
  static constexpr uint32_t kTextureFetchCount = 32;
  static constexpr uint32_t kTextureFetchWordCount = 6;

  NativeGuestDrawContext draw{};
  std::array<uint32_t, 2> viewport_offset{};
  std::array<uint32_t, 2> viewport_extent{};
  float viewport_min_depth = 0.0f;
  float viewport_max_depth = 1.0f;
  std::array<float, 3> ndc_scale{};
  std::array<float, 3> ndc_offset{};
  std::array<uint32_t, 2> scissor_offset{};
  std::array<uint32_t, 2> scissor_extent{};
  uint32_t active_texture_fetch_mask = 0;
  std::array<std::array<uint32_t, kTextureFetchWordCount>,
             kTextureFetchCount>
      texture_fetch_words{};
  bool viewport_scissor_valid = false;
  bool texture_fetches_valid = false;
  bool valid = false;
};

using NativeGuestDrawStateObserver = void (*)(
    const NativeGuestDrawStateContext& context, void* user_data);
// Cheap prefilter evaluated from already-available shader identities before
// render-scope and texture state are copied for the full observer context.
// This keeps targeted diagnostics from imposing full-state capture on every
// draw in a frame.
using NativeGuestDrawStateFilter = bool (*)(uint64_t vertex_shader_hash,
                                            uint64_t pixel_shader_hash,
                                            void* user_data);
void SetNativeGuestDrawStateObserver(NativeGuestDrawStateObserver observer,
                                     NativeGuestDrawStateFilter filter,
                                     void* user_data);
bool HasNativeGuestDrawStateObserver();
bool ShouldObserveNativeGuestDrawState(uint64_t vertex_shader_hash,
                                       uint64_t pixel_shader_hash);
void ObserveNativeGuestDrawState(
    const NativeGuestDrawStateContext& context);

// Value-owned copy of one exact Vulkan guest-shader translation selected for a
// draw. A microcode hash is not sufficient identity because translation
// modifications change the generated SPIR-V interface and behavior.
//
// This is an observation bridge, not a live backend-object escape hatch:
// shader modules, descriptor sets and VkPipelineLayout handles are
// intentionally not exposed. Consumers may retain the context after the
// callback without depending on pipeline-cache or command-processor lifetime.
struct NativeGuestShaderArtifactContext {
  struct TextureBinding {
    uint32_t binding = 0;
    uint32_t fetch_constant = 0;
    uint32_t dimension = 0;
    bool is_signed = false;
  };

  struct SamplerBinding {
    uint32_t binding = 0;
    uint32_t fetch_constant = 0;
    uint32_t mag_filter = 0;
    uint32_t min_filter = 0;
    uint32_t mip_filter = 0;
    uint32_t aniso_filter = 0;
  };

  NativeGuestOutputBackend backend = NativeGuestOutputBackend::kUnknown;
  uint64_t shader_hash = 0;
  uint64_t modification = 0;
  bool is_vertex_shader = false;

  // Exact translated SPIR-V words, including the SPIR-V header.
  std::vector<uint32_t> spirv;
  uint32_t used_texture_fetch_mask = 0;
  std::vector<TextureBinding> texture_bindings;
  std::vector<SamplerBinding> sampler_bindings;

  // Stable translated-shader descriptor contract. The first two sets contain
  // shared-memory/EDRAM and constants. Texture sets are stage-specific.
  uint32_t descriptor_set_shared_memory_and_edram = 0;
  uint32_t descriptor_set_constants = 1;
  uint32_t descriptor_set_textures_vertex = 2;
  uint32_t descriptor_set_textures_pixel = 3;
  uint32_t descriptor_set_count = 4;
  uint32_t shared_memory_binding = 0;
  uint32_t edram_binding = 1;
  uint32_t constant_buffer_system_binding = 0;
  uint32_t constant_buffer_float_vertex_binding = 1;
  uint32_t constant_buffer_float_pixel_binding = 2;
  uint32_t constant_buffer_bool_loop_binding = 3;
  uint32_t constant_buffer_fetch_binding = 4;
  uint32_t constant_buffer_count = 5;
  bool valid = false;
};

using NativeGuestShaderArtifactObserver = void (*)(
    const NativeGuestShaderArtifactContext& context, void* user_data);
// Cheap exact-translation filter. It is called only after the translation has
// succeeded, and before any vectors are copied.
using NativeGuestShaderArtifactFilter = bool (*)(
    uint64_t shader_hash, uint64_t modification, bool is_vertex_shader,
    void* user_data);
void SetNativeGuestShaderArtifactObserver(
    NativeGuestShaderArtifactObserver observer,
    NativeGuestShaderArtifactFilter filter, void* user_data);
bool HasNativeGuestShaderArtifactObserver();
bool ShouldObserveNativeGuestShaderArtifact(uint64_t shader_hash,
                                            uint64_t modification,
                                            bool is_vertex_shader);
void ObserveNativeGuestShaderArtifact(
    const NativeGuestShaderArtifactContext& context);

// Opaque, backend-owned snapshot of the translated guest draw state after the
// pipeline, descriptors, constants, dynamic state and render scope are all
// prepared. The token is valid only for backend_frame_sequence and deliberately
// cannot be dereferenced by title code. `valid` means the observation is
// structurally complete, not that the referenced resource contents are
// immutable for deferred replay. This phase-zero contract observes replay
// inputs only; it does not submit, replay or suppress anything.
struct NativeGuestTranslatedReplayTokenContext {
  static constexpr uint32_t kDescriptorSetCount = 4;
  static constexpr uint32_t kConstantBufferCount = 5;
  static constexpr uint32_t kMaxColorAttachments = 4;

  struct DescriptorSetIdentity {
    uint64_t generation = 0;
    bool valid = false;
  };

  struct ConstantBufferRange {
    uint64_t resource_generation = 0;
    uint64_t offset = 0;
    uint64_t range = 0;
    bool valid = false;
  };

  struct TextureResourceBinding {
    uint32_t descriptor_binding = 0;
    uint32_t fetch_constant = 0;
    uint32_t dimension = 0;
    bool is_vertex_shader = false;
    bool is_signed = false;
    std::array<uint32_t, 6> fetch_words{};
    uint64_t texture_key_hash = 0;
    uint32_t base_address = 0;
    uint32_t base_length = 0;
    uint32_t mip_address = 0;
    uint32_t mip_length = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth_or_array_size = 0;
    uint32_t format = 0;
    bool scaled_resolve = false;
    uint64_t image_view_generation = 0;
    uint64_t content_generation = 0;
    bool resident = false;

    bool operator==(const TextureResourceBinding&) const = default;
  };

  struct SamplerResourceBinding {
    uint32_t descriptor_binding = 0;
    uint32_t fetch_constant = 0;
    uint64_t parameters = 0;
    bool is_vertex_shader = false;

    bool operator==(const SamplerResourceBinding&) const = default;
  };

  NativeGuestOutputBackend backend = NativeGuestOutputBackend::kUnknown;
  uint64_t backend_frame_sequence = 0;
  uint64_t opaque_token = 0;
  uint64_t pipeline_generation = 0;
  uint64_t pipeline_layout_generation = 0;
  uint64_t vertex_shader_hash = 0;
  uint64_t pixel_shader_hash = 0;
  uint64_t vertex_shader_modification = 0;
  uint64_t pixel_shader_modification = 0;

  std::array<DescriptorSetIdentity, kDescriptorSetCount> descriptor_sets{};
  std::array<ConstantBufferRange, kConstantBufferCount> constant_buffers{};
  uint32_t descriptor_set_valid_mask = 0;
  uint32_t constant_buffer_valid_mask = 0;

  // Value-owned constant snapshots used to prove whether repeated tiled
  // submissions are one logical draw. Float values are packed in ascending
  // used-register order, matching the translated shader constant buffers.
  // The mechanical mask marks only backend-generated NDC / EDRAM-address
  // bytes that may be normalized by a separately proven untiled replay.
  std::vector<uint8_t> system_constants;
  std::vector<uint8_t> system_constants_mechanical_mask;
  uint32_t system_constants_ndc_scale_byte_offset = 0;
  uint32_t system_constants_ndc_offset_byte_offset = 0;
  std::array<uint64_t, 4> vertex_float_constant_usage{};
  std::array<uint64_t, 4> pixel_float_constant_usage{};
  std::vector<uint32_t> vertex_float_constants;
  std::vector<uint32_t> pixel_float_constants;
  std::array<uint32_t, 40> bool_loop_constants{};
  std::array<uint32_t, 32 * 6> fetch_constants{};
  std::array<float, 3> ndc_scale{};
  std::array<float, 3> ndc_offset{};
  bool constant_contents_valid = false;
  // Exact shader-used resource evidence. Transient VkDescriptorSet identity is
  // intentionally not content identity; these vectors describe what the
  // descriptors actually reference.
  std::vector<TextureResourceBinding> texture_resources;
  std::vector<SamplerResourceBinding> sampler_resources;
  bool resource_contents_valid = false;

  uint32_t guest_primitive_type = 0;
  uint32_t host_primitive_type = 0;
  uint32_t processed_index_buffer_type = 0;
  uint32_t host_index_format = 0;
  uint32_t guest_vertex_or_index_count = 0;
  uint32_t host_vertex_or_index_count = 0;
  uint32_t guest_index_base = 0;
  bool indexed = false;
  bool guest_index_base_valid = false;

  std::array<float, 6> viewport{};
  std::array<int32_t, 2> scissor_offset{};
  std::array<uint32_t, 2> scissor_extent{};
  float depth_bias_constant_factor = 0.0f;
  float depth_bias_slope_factor = 0.0f;
  std::array<float, 4> blend_constants{};
  uint32_t stencil_compare_mask_front = 0;
  uint32_t stencil_compare_mask_back = 0;
  uint32_t stencil_write_mask_front = 0;
  uint32_t stencil_write_mask_back = 0;
  uint32_t stencil_reference_front = 0;
  uint32_t stencil_reference_back = 0;

  std::array<nrhi::Format, kMaxColorAttachments> color_attachment_formats{};
  uint32_t color_attachment_count = 0;
  nrhi::Format depth_attachment_format = nrhi::Format::kUnknown;
  nrhi::Format stencil_attachment_format = nrhi::Format::kUnknown;
  uint32_t sample_count = 1;
  uint64_t sample_mask = UINT64_MAX;
  bool dynamic_rendering = false;
  // True only when the backend has snapshotted or guarded every mutable
  // resource referenced by this token through the output callback. Descriptor
  // handle lifetime alone is insufficient: shared memory, texture contents /
  // layouts and GuestDMA index bytes may change after token emission.
  bool resources_stable_for_deferred_replay = false;
  bool valid = false;
};

using NativeGuestTranslatedReplayTokenObserver = void (*)(
    const NativeGuestTranslatedReplayTokenContext& context, void* user_data);
using NativeGuestTranslatedReplayTokenFilter = bool (*)(
    uint64_t vertex_shader_hash, uint64_t pixel_shader_hash, void* user_data);
void SetNativeGuestTranslatedReplayTokenObserver(
    NativeGuestTranslatedReplayTokenObserver observer,
    NativeGuestTranslatedReplayTokenFilter filter, void* user_data);
bool HasNativeGuestTranslatedReplayTokenObserver();
bool ShouldCaptureNativeGuestTranslatedReplayToken(
    uint64_t vertex_shader_hash, uint64_t pixel_shader_hash);
void ObserveNativeGuestTranslatedReplayToken(
    const NativeGuestTranslatedReplayTokenContext& context);

// Caller-owned private attachments for one diagnostic translated replay.
// They must belong to the NativeGuestOutputRenderContext device and remain
// valid for the duration of the call. Replay never touches guest_output,
// presents, or suppresses the corresponding guest draw.
struct NativeGuestTranslatedReplayTarget {
  nrhi::Texture* color = nullptr;
  nrhi::Texture* depth_stencil = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  nrhi::Format color_format = nrhi::Format::kUnknown;
  nrhi::Format depth_stencil_format = nrhi::Format::kUnknown;
  uint32_t sample_count = 1;
  bool clear_color = false;
  bool clear_depth_stencil = false;
  std::array<float, 4> clear_color_value{};
  float clear_depth_value = 1.0f;
  uint32_t clear_stencil_value = 0;
};

enum class NativeGuestTranslatedReplayResult : uint32_t {
  kSucceeded = 0,
  kUnsupportedBackend,
  kInvalidContext,
  kInvalidToken,
  kExpiredToken,
  kResourcesNotStable,
  kIncompatibleTarget,
  kIncompleteBindings,
  kUnsupportedDraw,
  kRenderScopeUnavailable,
};

// Backend-only dispatch installed for the duration of the guest-output
// callback. Keeping the callback and its Vulkan state private avoids exposing
// VkPipeline, VkDescriptorSet, VkBuffer, or VkImage handles to title code.
using NativeGuestTranslatedReplayBackend = NativeGuestTranslatedReplayResult (*)(
    const NativeGuestOutputRenderContext& output_context,
    const NativeGuestTranslatedReplayTokenContext& token,
    const NativeGuestTranslatedReplayTarget& target, void* user_data);
void SetNativeGuestTranslatedReplayBackend(
    NativeGuestTranslatedReplayBackend backend, void* user_data);

// Atomic ordered replay of a guarded current-frame packet. The backend
// preflights the complete vector before recording any replay command, then
// opens one render scope, clears once, records all draws, and closes once.
using NativeGuestTranslatedReplayBatchBackend =
    NativeGuestTranslatedReplayResult (*)(
        const NativeGuestOutputRenderContext& output_context,
        const std::vector<NativeGuestTranslatedReplayTokenContext>& tokens,
        const NativeGuestTranslatedReplayTarget& target, void* user_data);
void SetNativeGuestTranslatedReplayBatchBackend(
    NativeGuestTranslatedReplayBatchBackend backend, void* user_data);

// Read-only backend query for the complete, ordered, current-frame PS328
// tile-one candidate family. Returns false and clears `tokens` unless the
// backend has proven the entire three-block repetition and every tile-one
// candidate has a private stable-resource guard.
using NativeGuestGuardedPs328Tile1TokenQuery = bool (*)(
    uint64_t backend_frame_sequence,
    std::vector<NativeGuestTranslatedReplayTokenContext>& tokens,
    void* user_data);
void SetNativeGuestGuardedPs328Tile1TokenQuery(
    NativeGuestGuardedPs328Tile1TokenQuery query, void* user_data);
bool TryGetCurrentFrameGuardedPs328Tile1ReplayTokens(
    uint64_t backend_frame_sequence,
    std::vector<NativeGuestTranslatedReplayTokenContext>& tokens);

enum class NativeGuestGuardedReplayFamily : uint32_t {
  kUnknown = 0,
  kVenuePs328,
  kVenue9E,
  kCrowdC6,
  kVenue14D,
  // A MAIN shader pair discovered at run time rather than reverse engineered
  // in advance. Its identity is the vertex/pixel shader hash pair carried on
  // each batch token and family summary, not this enumerator.
  kGenericMain,
};

struct NativeGuestGuardedReplayPacket {
  NativeGuestGuardedReplayFamily family =
      NativeGuestGuardedReplayFamily::kUnknown;
  uint64_t backend_frame_sequence = 0;
  std::vector<NativeGuestTranslatedReplayTokenContext> tokens;
  // Full-output copies derived from all three observed tile blocks. The
  // backend never synthesizes viewport/NDC/constants; currently the only
  // permitted mutation is widening a proven tile-one scissor to 1280x720.
  std::vector<NativeGuestTranslatedReplayTokenContext> normalized_tokens;
  uint32_t normalized_output_width = 0;
  uint32_t normalized_output_height = 0;
  bool scissor_only_normalization = false;
  bool full_output_normalization_proven = false;
  bool three_blocks_exact = false;
  bool valid = false;
};

using NativeGuestGuardedReplayPacketQuery = bool (*)(
    NativeGuestGuardedReplayFamily family, uint64_t backend_frame_sequence,
    NativeGuestGuardedReplayPacket& packet, void* user_data);
void SetNativeGuestGuardedReplayPacketQuery(
    NativeGuestGuardedReplayPacketQuery query, void* user_data);
bool TryGetCurrentFrameGuardedReplayPacket(
    NativeGuestGuardedReplayFamily family, uint64_t backend_frame_sequence,
    NativeGuestGuardedReplayPacket& packet);

struct NativeGuestGuardedReplayBatchToken {
  NativeGuestGuardedReplayFamily family =
      NativeGuestGuardedReplayFamily::kUnknown;
  uint32_t family_offset = 0;
  // Exact identity of a discovered family, where the enumerator alone is
  // ambiguous. Always populated, including for the four named families.
  uint64_t vertex_shader_hash = 0;
  uint64_t pixel_shader_hash = 0;
  NativeGuestTranslatedReplayTokenContext token;
};

struct NativeGuestGuardedReplayBatch {
  static constexpr uint32_t kMaximumTokens = 320;

  uint64_t backend_frame_sequence = 0;
  uint32_t ps328_token_count = 0;
  uint32_t venue_9e_token_count = 0;
  uint32_t crowd_c6_token_count = 0;
  uint32_t venue_14d_token_count = 0;
  uint32_t normalized_output_width = 0;
  uint32_t normalized_output_height = 0;
  std::vector<NativeGuestGuardedReplayBatchToken> ordered_tokens;
  bool scissor_only_normalization = false;
  bool full_output_normalization_proven = false;
  bool valid = false;
};

// Read-only join of the complete current-frame PS328, Venue9E, CrowdC6 and
// Venue14D guarded first-block packets. All three-block proofs must be complete
// and every token resource-stable. The result is globally ordered by the
// opaque token identity allocated at guest draw capture; gaps for unselected
// families are allowed, duplicates and reversals fail closed. Tokens in
// `ordered_tokens` are the backend-proven full-output copies, never the raw
// tile-one tokens.
bool TryGetCurrentFrameGuardedVenueReplayBatch(
    uint64_t backend_frame_sequence,
    NativeGuestGuardedReplayBatch& batch);

// Why a family contributed no tokens to the generic MAIN plan. A family may
// report several reasons at once; the mask exists so the uncovered remainder
// can be triaged from one log line instead of a new bespoke observer.
enum NativeGuestGuardedMainFamilyReject : uint32_t {
  kNativeGuestGuardedMainFamilyRejectNone = 0,
  // The three-tile block proof never reached the proven phase this frame.
  kNativeGuestGuardedMainFamilyRejectUnproven = 1u << 0,
  // Tile blocks differed in title-owned state, not merely in tile geometry.
  kNativeGuestGuardedMainFamilyRejectSemanticTileState = 1u << 1,
  // A guarded payload was missing, mutable, or outside the family budget.
  kNativeGuestGuardedMainFamilyRejectResourcesNotStable = 1u << 2,
  // The family's attachment signature differed from the plan's MAIN target.
  kNativeGuestGuardedMainFamilyRejectAttachmentMismatch = 1u << 3,
  // The family alone, or the plan with it, exceeded the token budget.
  kNativeGuestGuardedMainFamilyRejectTokenBudget = 1u << 4,
};

struct NativeGuestGuardedMainFamilySummary {
  NativeGuestGuardedReplayFamily family =
      NativeGuestGuardedReplayFamily::kUnknown;
  uint64_t vertex_shader_hash = 0;
  uint64_t pixel_shader_hash = 0;
  // Guarded first-block candidates observed this frame, before any proof.
  uint32_t observed_token_count = 0;
  // Tokens this family actually contributed to `ordered_tokens`.
  uint32_t replayed_token_count = 0;
  uint32_t phase = 0;
  uint32_t reject_mask = kNativeGuestGuardedMainFamilyRejectNone;
};

struct NativeGuestGuardedMainReplayBatch {
  // The live MAIN pass has recorded 310-515 logical draws across camera and
  // content variation, so the plan budget is well above the old 320-token
  // four-family ceiling.
  static constexpr uint32_t kMaximumTokens = 1024;
  static constexpr uint32_t kMaximumFamilies = 128;

  uint64_t backend_frame_sequence = 0;
  uint32_t normalized_output_width = 0;
  uint32_t normalized_output_height = 0;
  uint32_t discovered_family_count = 0;
  uint32_t proven_family_count = 0;
  // Every guarded first-block candidate observed this frame across all
  // families. `covered_token_count / observed_token_count` is the honest
  // generic-capture coverage ratio.
  uint32_t observed_token_count = 0;
  uint32_t covered_token_count = 0;
  std::vector<NativeGuestGuardedReplayBatchToken> ordered_tokens;
  std::vector<NativeGuestGuardedMainFamilySummary> families;
  bool scissor_only_normalization = false;
  bool full_output_normalization_proven = false;
  bool valid = false;
};

// Read-only join of every proven guarded family in the current frame, ordered
// globally by the opaque token identity allocated at guest draw capture.
//
// Unlike the four-family venue batch, an unproven family is excluded and
// reported rather than failing the whole query, because generic capture is
// expected to be partial while families are still being admitted. `valid`
// therefore means "this plan is internally consistent and replay-safe", not
// "the MAIN pass is completely covered". Coverage is a separate serving gate.
using NativeGuestGuardedMainReplayBatchQuery = bool (*)(
    uint64_t backend_frame_sequence,
    NativeGuestGuardedMainReplayBatch& batch, void* user_data);
void SetNativeGuestGuardedMainReplayBatchQuery(
    NativeGuestGuardedMainReplayBatchQuery query, void* user_data);
bool TryGetCurrentFrameGuardedMainReplayBatch(
    uint64_t backend_frame_sequence,
    NativeGuestGuardedMainReplayBatch& batch);

// Records exactly one current-frame token into a private target. This is a
// fail-closed diagnostic transaction: any validation failure records no replay
// draw and the function never falls back to, suppresses, or presents guest
// rendering.
NativeGuestTranslatedReplayResult TryReplayNativeGuestTranslatedDraw(
    const NativeGuestOutputRenderContext& output_context,
    const NativeGuestTranslatedReplayTokenContext& token,
    const NativeGuestTranslatedReplayTarget& target);
NativeGuestTranslatedReplayResult TryReplayNativeGuestTranslatedDrawBatch(
    const NativeGuestOutputRenderContext& output_context,
    const std::vector<NativeGuestTranslatedReplayTokenContext>& tokens,
    const NativeGuestTranslatedReplayTarget& target);
const char* NativeGuestTranslatedReplayResultName(
    NativeGuestTranslatedReplayResult result);

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
