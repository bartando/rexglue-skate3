#include <rex/graphics/native_guest_renderer.h>

#include <algorithm>
#include <atomic>
#include <ranges>
#include <string>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>

// Defined here (shared TU) rather than in a backend's command_processor.cpp
// so both D3D12 and Vulkan see the same definition.
REXCVAR_DEFINE_BOOL(native_render_suppress_emulated_draws, true, "GPU",
                    "While the registered native guest-output renderer is actively "
                    "replacing frames, skip emulated draw and resolve execution in the "
                    "command processor (PM4 parsing, fences, queries and memexport draws "
                    "still run). Menus/pause yield to the emulated path and execute "
                    "normally.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(native_render_suppress_mode, 2, "GPU",
                     "Which emulated passes to suppress while the native guest-output "
                     "renderer is active. 0 = framebuffer-sized passes only (surface "
                     "pitch >= 1280); the game's whole postfx chain then still executes "
                     "at 1152x640 x resolution scale EVERY frame and paces the pipeline "
                     "(~half the achievable frame rate). 1 = suppress everything (perf "
                     "probing; mid-gameplay lightmap page composition breaks). 2 = "
                     "suppress everything EXCEPT the memory-composition passes whose "
                     "outputs the native renderer samples from guest memory: lightmap "
                     "page composition (pitch 1024) and small composite surfaces (pitch "
                     "<= 512, CAS outfit pieces). Menus/pause/loading always render "
                     "fully (the native renderer yields there), so shop/outfit "
                     "composition is unaffected by any mode. 3 = portrait-window mode: "
                     "like 0 the sub-framebuffer RTT passes execute (one-shot frontend "
                     "portrait renders, census pitches 560-1200), but the 1152-wide "
                     "main scene + postfx band stays suppressed; that band is the "
                     "whole-pipeline cost at scaled resolutions and portraits never "
                     "need it.")
    .range(0, 3)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// A timing probe, not a rendering mode. Normal suppression requires the native
// renderer to be actively serving, which is precisely what cannot be arranged
// before the work it would pay for exists. This answers the question that gates
// that work: of the ~100 ms guest frame, how much is the framebuffer-sized MAIN
// pass and its resolves - draws, residency, texture upload and render-pass
// setup together - rather than the passes a MAIN takeover would not remove?
//
// Output is deliberately wrong while this is on. It must never be enabled in a
// content-correctness run, and it does not present, replace, or serve anything.
REXCVAR_DEFINE_BOOL(native_render_main_pass_suppression_benchmark, false, "GPU",
                    "Timing probe: skip framebuffer-sized (pitch >= 1280) emulated passes "
                    "and their resolves even when no native renderer is serving. Renders "
                    "incorrect output on purpose; measures the frame cost that a whole-MAIN "
                    "native takeover could remove.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(native_render_main_pass_suppression_benchmark_delay_seconds, 0, "GPU",
                     "Delay before the MAIN-pass timing probe engages, in seconds of host "
                     "uptime. Automated runs navigate menus by screenshot and cannot find "
                     "them once the probe blanks the output, so the probe waits until "
                     "gameplay has been reached. 0 engages immediately.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(native_render_suppress_exempt_depth_only, true, "GPU",
                    "Within the suppression-EXEMPT passes (see "
                    "native_render_suppress_mode), also skip draws with no pixel shader "
                    "(depth/stencil-only: shadow-map casters, z-prepasses). Their output "
                    "feeds only the suppressed scene passes; the native renderer builds "
                    "its own shadows, and on Vulkan this stream is the dominant "
                    "remaining emulated GPU cost (2-12 ms/frame at 3x, the bimodal-FPS "
                    "slow state). Disable if lightmap-page or CAS composition content "
                    "regresses (a composition pass depth/stencil-testing against its own "
                    "no-PS lay-down would need this off).")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {
namespace {

std::atomic<NativeGuestOutputRenderer> g_renderer{nullptr};
std::atomic<void*> g_renderer_user_data{nullptr};
// True while the registered renderer actually replaced the last presented
// frame (false when it yields: menus, early-outs, no renderer).
std::atomic<bool> g_native_output_active{false};
std::atomic<NativeGuestOutputPostProcessor> g_post_processor{nullptr};
std::atomic<void*> g_post_processor_user_data{nullptr};
std::atomic<bool> g_post_process_requested{false};
std::atomic<NativeGuestDrawMatcher> g_draw_matcher{nullptr};
std::atomic<NativeGuestDrawRenderer> g_draw_renderer{nullptr};
std::atomic<void*> g_draw_replacer_user_data{nullptr};
std::atomic<NativeGuestDrawEligibilityObserver> g_draw_eligibility_observer{
    nullptr};
std::atomic<void*> g_draw_eligibility_observer_user_data{nullptr};
std::atomic<NativeGuestDrawStateObserver> g_draw_state_observer{nullptr};
std::atomic<NativeGuestDrawStateFilter> g_draw_state_filter{nullptr};
std::atomic<void*> g_draw_state_observer_user_data{nullptr};
std::atomic<NativeGuestShaderArtifactObserver> g_shader_artifact_observer{
    nullptr};
std::atomic<NativeGuestShaderArtifactFilter> g_shader_artifact_filter{nullptr};
std::atomic<void*> g_shader_artifact_observer_user_data{nullptr};
std::atomic<NativeGuestTranslatedReplayTokenObserver> g_replay_token_observer{
    nullptr};
std::atomic<NativeGuestTranslatedReplayTokenFilter> g_replay_token_filter{
    nullptr};
std::atomic<void*> g_replay_token_observer_user_data{nullptr};
thread_local NativeGuestTranslatedReplayBackend g_translated_replay_backend =
    nullptr;
thread_local void* g_translated_replay_backend_user_data = nullptr;
thread_local NativeGuestTranslatedReplayBatchBackend
    g_translated_replay_batch_backend = nullptr;
thread_local void* g_translated_replay_batch_backend_user_data = nullptr;
thread_local NativeGuestGuardedPs328Tile1TokenQuery
    g_guarded_ps328_tile1_token_query = nullptr;
thread_local void* g_guarded_ps328_tile1_token_query_user_data = nullptr;
thread_local NativeGuestGuardedReplayPacketQuery
    g_guarded_replay_packet_query = nullptr;
thread_local void* g_guarded_replay_packet_query_user_data = nullptr;
thread_local NativeGuestGuardedMainReplayBatchQuery
    g_guarded_main_replay_batch_query = nullptr;
thread_local void* g_guarded_main_replay_batch_query_user_data = nullptr;

}  // namespace

void SetNativeGuestOutputRenderer(NativeGuestOutputRenderer renderer, void* user_data) {
  g_renderer_user_data.store(user_data, std::memory_order_release);
  g_renderer.store(renderer, std::memory_order_release);
  if (renderer == nullptr) {
    g_native_output_active.store(false, std::memory_order_relaxed);
  }
}

bool TryRenderNativeGuestOutput(const NativeGuestOutputRenderContext& context) {
  NativeGuestOutputRenderer renderer = g_renderer.load(std::memory_order_acquire);
  if (renderer == nullptr) {
    g_native_output_active.store(false, std::memory_order_relaxed);
    return false;
  }
  void* user_data = g_renderer_user_data.load(std::memory_order_acquire);
  const bool rendered = renderer(context, user_data);
  g_native_output_active.store(rendered, std::memory_order_relaxed);
  return rendered;
}

bool HasNativeGuestOutputRenderer() {
  return g_renderer.load(std::memory_order_acquire) != nullptr;
}

void SetNativeGuestOutputPostProcessor(NativeGuestOutputPostProcessor post_processor,
                                       void* user_data) {
  g_post_processor_user_data.store(user_data, std::memory_order_release);
  g_post_processor.store(post_processor, std::memory_order_release);
}

bool HasNativeGuestOutputPostProcessor() {
  return g_post_processor.load(std::memory_order_acquire) != nullptr;
}

void InvokeNativeGuestOutputPostProcessor(const NativeGuestOutputRenderContext& context) {
  NativeGuestOutputPostProcessor post_processor =
      g_post_processor.load(std::memory_order_acquire);
  if (post_processor == nullptr) {
    return;
  }
  post_processor(context, g_post_processor_user_data.load(std::memory_order_acquire));
}

void RequestNativeGuestOutputPostProcess(bool requested) {
  g_post_process_requested.store(requested, std::memory_order_release);
}

bool IsNativeGuestOutputPostProcessRequested() {
  return g_post_process_requested.load(std::memory_order_acquire);
}

void SetNativeGuestDrawReplacer(NativeGuestDrawMatcher matcher,
                                NativeGuestDrawRenderer renderer,
                                void* user_data) {
  g_draw_replacer_user_data.store(user_data, std::memory_order_release);
  g_draw_renderer.store(renderer, std::memory_order_release);
  g_draw_matcher.store(matcher, std::memory_order_release);
}

bool HasNativeGuestDrawReplacer() {
  return g_draw_matcher.load(std::memory_order_acquire) != nullptr &&
         g_draw_renderer.load(std::memory_order_acquire) != nullptr;
}

void SetNativeGuestDrawEligibilityObserver(
    NativeGuestDrawEligibilityObserver observer, void* user_data) {
  g_draw_eligibility_observer_user_data.store(user_data,
                                               std::memory_order_release);
  g_draw_eligibility_observer.store(observer, std::memory_order_release);
}

bool HasNativeGuestDrawEligibilityObserver() {
  return g_draw_eligibility_observer.load(std::memory_order_acquire) !=
         nullptr;
}

void ObserveNativeGuestDrawEligibility(
    const NativeGuestDrawEligibilityContext& context) {
  NativeGuestDrawEligibilityObserver observer =
      g_draw_eligibility_observer.load(std::memory_order_acquire);
  if (observer == nullptr) {
    return;
  }
  observer(context, g_draw_eligibility_observer_user_data.load(
                        std::memory_order_acquire));
}

void SetNativeGuestDrawStateObserver(NativeGuestDrawStateObserver observer,
                                     NativeGuestDrawStateFilter filter,
                                     void* user_data) {
  g_draw_state_observer_user_data.store(user_data,
                                        std::memory_order_release);
  g_draw_state_filter.store(filter, std::memory_order_release);
  g_draw_state_observer.store(observer, std::memory_order_release);
}

bool HasNativeGuestDrawStateObserver() {
  return g_draw_state_observer.load(std::memory_order_acquire) != nullptr;
}

bool ShouldObserveNativeGuestDrawState(uint64_t vertex_shader_hash,
                                       uint64_t pixel_shader_hash) {
  NativeGuestDrawStateFilter filter =
      g_draw_state_filter.load(std::memory_order_acquire);
  return filter == nullptr ||
         filter(vertex_shader_hash, pixel_shader_hash,
                g_draw_state_observer_user_data.load(
                    std::memory_order_acquire));
}

void ObserveNativeGuestDrawState(
    const NativeGuestDrawStateContext& context) {
  NativeGuestDrawStateObserver observer =
      g_draw_state_observer.load(std::memory_order_acquire);
  if (observer == nullptr) {
    return;
  }
  observer(context, g_draw_state_observer_user_data.load(
                        std::memory_order_acquire));
}

void SetNativeGuestShaderArtifactObserver(
    NativeGuestShaderArtifactObserver observer,
    NativeGuestShaderArtifactFilter filter, void* user_data) {
  g_shader_artifact_observer_user_data.store(user_data,
                                             std::memory_order_release);
  g_shader_artifact_filter.store(filter, std::memory_order_release);
  g_shader_artifact_observer.store(observer, std::memory_order_release);
}

bool HasNativeGuestShaderArtifactObserver() {
  return g_shader_artifact_observer.load(std::memory_order_acquire) != nullptr;
}

bool ShouldObserveNativeGuestShaderArtifact(uint64_t shader_hash,
                                            uint64_t modification,
                                            bool is_vertex_shader) {
  NativeGuestShaderArtifactFilter filter =
      g_shader_artifact_filter.load(std::memory_order_acquire);
  return filter == nullptr ||
         filter(shader_hash, modification, is_vertex_shader,
                g_shader_artifact_observer_user_data.load(
                    std::memory_order_acquire));
}

void ObserveNativeGuestShaderArtifact(
    const NativeGuestShaderArtifactContext& context) {
  NativeGuestShaderArtifactObserver observer =
      g_shader_artifact_observer.load(std::memory_order_acquire);
  if (observer == nullptr || !context.valid) {
    return;
  }
  observer(context, g_shader_artifact_observer_user_data.load(
                        std::memory_order_acquire));
}

void SetNativeGuestTranslatedReplayTokenObserver(
    NativeGuestTranslatedReplayTokenObserver observer,
    NativeGuestTranslatedReplayTokenFilter filter, void* user_data) {
  g_replay_token_observer_user_data.store(user_data,
                                          std::memory_order_release);
  g_replay_token_filter.store(filter, std::memory_order_release);
  g_replay_token_observer.store(observer, std::memory_order_release);
}

bool HasNativeGuestTranslatedReplayTokenObserver() {
  return g_replay_token_observer.load(std::memory_order_acquire) != nullptr;
}

bool ShouldCaptureNativeGuestTranslatedReplayToken(
    uint64_t vertex_shader_hash, uint64_t pixel_shader_hash) {
  NativeGuestTranslatedReplayTokenFilter filter =
      g_replay_token_filter.load(std::memory_order_acquire);
  return filter == nullptr ||
         filter(vertex_shader_hash, pixel_shader_hash,
                g_replay_token_observer_user_data.load(
                    std::memory_order_acquire));
}

void ObserveNativeGuestTranslatedReplayToken(
    const NativeGuestTranslatedReplayTokenContext& context) {
  NativeGuestTranslatedReplayTokenObserver observer =
      g_replay_token_observer.load(std::memory_order_acquire);
  if (observer == nullptr || !context.valid) {
    return;
  }
  observer(context, g_replay_token_observer_user_data.load(
                        std::memory_order_acquire));
}

void SetNativeGuestTranslatedReplayBackend(
    NativeGuestTranslatedReplayBackend backend, void* user_data) {
  g_translated_replay_backend_user_data = user_data;
  g_translated_replay_backend = backend;
}

void SetNativeGuestTranslatedReplayBatchBackend(
    NativeGuestTranslatedReplayBatchBackend backend, void* user_data) {
  g_translated_replay_batch_backend_user_data = user_data;
  g_translated_replay_batch_backend = backend;
}

void SetNativeGuestGuardedPs328Tile1TokenQuery(
    NativeGuestGuardedPs328Tile1TokenQuery query, void* user_data) {
  g_guarded_ps328_tile1_token_query_user_data = user_data;
  g_guarded_ps328_tile1_token_query = query;
}

bool TryGetCurrentFrameGuardedPs328Tile1ReplayTokens(
    uint64_t backend_frame_sequence,
    std::vector<NativeGuestTranslatedReplayTokenContext>& tokens) {
  tokens.clear();
  if (backend_frame_sequence == 0 ||
      g_guarded_ps328_tile1_token_query == nullptr) {
    return false;
  }
  if (!g_guarded_ps328_tile1_token_query(
          backend_frame_sequence, tokens,
          g_guarded_ps328_tile1_token_query_user_data)) {
    tokens.clear();
    return false;
  }
  return !tokens.empty();
}

void SetNativeGuestGuardedReplayPacketQuery(
    NativeGuestGuardedReplayPacketQuery query, void* user_data) {
  g_guarded_replay_packet_query_user_data = user_data;
  g_guarded_replay_packet_query = query;
}

bool TryGetCurrentFrameGuardedReplayPacket(
    NativeGuestGuardedReplayFamily family,
    uint64_t backend_frame_sequence,
    NativeGuestGuardedReplayPacket& packet) {
  packet = {};
  if (family == NativeGuestGuardedReplayFamily::kUnknown ||
      backend_frame_sequence == 0 ||
      g_guarded_replay_packet_query == nullptr ||
      !g_guarded_replay_packet_query(
          family, backend_frame_sequence, packet,
          g_guarded_replay_packet_query_user_data) ||
      !packet.valid || !packet.three_blocks_exact ||
      !packet.full_output_normalization_proven ||
      !packet.scissor_only_normalization ||
      packet.normalized_output_width != 1280 ||
      packet.normalized_output_height != 720 ||
      packet.family != family ||
      packet.backend_frame_sequence != backend_frame_sequence ||
      packet.tokens.empty() ||
      packet.normalized_tokens.size() != packet.tokens.size()) {
    packet = {};
    return false;
  }
  return true;
}

bool TryGetCurrentFrameGuardedVenueReplayBatch(
    uint64_t backend_frame_sequence,
    NativeGuestGuardedReplayBatch& batch) {
  constexpr uint64_t kPs328VertexShader =
      UINT64_C(0x0E9982BE6B1E99A1);
  constexpr uint64_t kPs328PixelShader =
      UINT64_C(0x328FA02B07C392DC);
  constexpr uint64_t kVenue9EVertexShader =
      UINT64_C(0x37F2AEC8A23E44E0);
  constexpr uint64_t kVenue9EPixelShader =
      UINT64_C(0x9E1AF02A96682354);
  constexpr uint64_t kCrowdC6VertexShader =
      UINT64_C(0xBD4B1DF972B828B7);
  constexpr uint64_t kCrowdC6PixelShader =
      UINT64_C(0xC6CEFDA3753CF2BA);
  constexpr uint64_t kVenue14DVertexShader40 =
      UINT64_C(0x4EAEC701E97DCDAD);
  constexpr uint64_t kVenue14DVertexShader48 =
      UINT64_C(0x08D6210341AD63F6);
  constexpr uint64_t kVenue14DPixelShader =
      UINT64_C(0x14D6B61CBC3D853C);
  constexpr size_t kExpectedCrowdC6LogicalDraws = 156;
  batch = {};
  if (backend_frame_sequence == 0) {
    return false;
  }

  NativeGuestGuardedReplayPacket ps328;
  NativeGuestGuardedReplayPacket venue_9e;
  NativeGuestGuardedReplayPacket crowd_c6;
  NativeGuestGuardedReplayPacket venue_14d;
  if (!TryGetCurrentFrameGuardedReplayPacket(
          NativeGuestGuardedReplayFamily::kVenuePs328,
          backend_frame_sequence, ps328) ||
      !TryGetCurrentFrameGuardedReplayPacket(
          NativeGuestGuardedReplayFamily::kVenue9E,
          backend_frame_sequence, venue_9e) ||
      !TryGetCurrentFrameGuardedReplayPacket(
          NativeGuestGuardedReplayFamily::kCrowdC6,
          backend_frame_sequence, crowd_c6) ||
      !TryGetCurrentFrameGuardedReplayPacket(
          NativeGuestGuardedReplayFamily::kVenue14D,
          backend_frame_sequence, venue_14d)) {
    return false;
  }
  if (crowd_c6.tokens.size() != kExpectedCrowdC6LogicalDraws ||
      ps328.tokens.size() + venue_9e.tokens.size() +
          crowd_c6.tokens.size() + venue_14d.tokens.size() >
      NativeGuestGuardedReplayBatch::kMaximumTokens) {
    return false;
  }

  batch.backend_frame_sequence = backend_frame_sequence;
  batch.ps328_token_count = uint32_t(ps328.tokens.size());
  batch.venue_9e_token_count = uint32_t(venue_9e.tokens.size());
  batch.crowd_c6_token_count = uint32_t(crowd_c6.tokens.size());
  batch.venue_14d_token_count = uint32_t(venue_14d.tokens.size());
  batch.normalized_output_width = 1280;
  batch.normalized_output_height = 720;
  batch.scissor_only_normalization = true;
  batch.full_output_normalization_proven = true;
  batch.ordered_tokens.reserve(
      ps328.tokens.size() + venue_9e.tokens.size() +
      crowd_c6.tokens.size() + venue_14d.tokens.size());
  const auto append_family =
      [&](NativeGuestGuardedReplayFamily family,
          uint64_t vertex_shader_hash,
          uint64_t alternate_vertex_shader_hash,
          uint64_t pixel_shader_hash,
          const NativeGuestGuardedReplayPacket& packet) {
        if (!packet.full_output_normalization_proven ||
            !packet.scissor_only_normalization ||
            packet.normalized_output_width != 1280 ||
            packet.normalized_output_height != 720 ||
            packet.normalized_tokens.size() != packet.tokens.size()) {
          return false;
        }
        uint64_t previous_token = 0;
        for (uint32_t offset = 0; offset < packet.tokens.size();
             ++offset) {
          const NativeGuestTranslatedReplayTokenContext& raw =
              packet.tokens[offset];
          const NativeGuestTranslatedReplayTokenContext& token =
              packet.normalized_tokens[offset];
          if (!token.valid || !token.constant_contents_valid ||
              !token.resource_contents_valid ||
              !token.resources_stable_for_deferred_replay ||
              token.backend_frame_sequence != backend_frame_sequence ||
              (token.vertex_shader_hash != vertex_shader_hash &&
               (alternate_vertex_shader_hash == 0 ||
                token.vertex_shader_hash !=
                    alternate_vertex_shader_hash)) ||
              token.pixel_shader_hash != pixel_shader_hash ||
              token.opaque_token != raw.opaque_token ||
              token.viewport != raw.viewport ||
              token.ndc_scale != raw.ndc_scale ||
              token.ndc_offset != raw.ndc_offset ||
              token.system_constants != raw.system_constants ||
              token.scissor_offset != std::array<int32_t, 2>{0, 0} ||
              token.scissor_extent !=
                  std::array<uint32_t, 2>{1280, 720} ||
              token.opaque_token == 0 ||
              token.opaque_token <= previous_token) {
            return false;
          }
          previous_token = token.opaque_token;
          batch.ordered_tokens.push_back({
              .family = family,
              .family_offset = offset,
              .token = token,
          });
        }
        return true;
      };
  if (!append_family(NativeGuestGuardedReplayFamily::kVenuePs328,
                     kPs328VertexShader, 0, kPs328PixelShader,
                     ps328) ||
      !append_family(NativeGuestGuardedReplayFamily::kVenue9E,
                     kVenue9EVertexShader, 0, kVenue9EPixelShader,
                     venue_9e) ||
      !append_family(NativeGuestGuardedReplayFamily::kCrowdC6,
                     kCrowdC6VertexShader, 0, kCrowdC6PixelShader,
                     crowd_c6) ||
      !append_family(NativeGuestGuardedReplayFamily::kVenue14D,
                     kVenue14DVertexShader40,
                     kVenue14DVertexShader48,
                     kVenue14DPixelShader, venue_14d)) {
    batch = {};
    return false;
  }

  std::ranges::sort(
      batch.ordered_tokens, {},
      [](const NativeGuestGuardedReplayBatchToken& entry) {
        return entry.token.opaque_token;
      });
  uint64_t previous_token = 0;
  const NativeGuestTranslatedReplayTokenContext& first =
      batch.ordered_tokens.front().token;
  for (const NativeGuestGuardedReplayBatchToken& entry :
       batch.ordered_tokens) {
    const NativeGuestTranslatedReplayTokenContext& token = entry.token;
    if (token.opaque_token <= previous_token ||
        token.color_attachment_count != first.color_attachment_count ||
        token.color_attachment_formats != first.color_attachment_formats ||
        token.depth_attachment_format != first.depth_attachment_format ||
        token.stencil_attachment_format != first.stencil_attachment_format ||
        token.sample_count != first.sample_count ||
        token.sample_mask != first.sample_mask ||
        token.dynamic_rendering != first.dynamic_rendering) {
      batch = {};
      return false;
    }
    previous_token = token.opaque_token;
  }
  const size_t aggregate_token_count =
      size_t(batch.ps328_token_count) +
      size_t(batch.venue_9e_token_count) +
      size_t(batch.crowd_c6_token_count) +
      size_t(batch.venue_14d_token_count);
  batch.valid = !batch.ordered_tokens.empty() &&
                aggregate_token_count == batch.ordered_tokens.size() &&
                aggregate_token_count <=
                    NativeGuestGuardedReplayBatch::kMaximumTokens &&
                batch.ps328_token_count != 0 &&
                batch.venue_9e_token_count != 0 &&
                batch.crowd_c6_token_count != 0 &&
                batch.venue_14d_token_count != 0 &&
                batch.full_output_normalization_proven;
  if (!batch.valid) {
    batch = {};
    return false;
  }
  return true;
}

void SetNativeGuestGuardedMainReplayBatchQuery(
    NativeGuestGuardedMainReplayBatchQuery query, void* user_data) {
  g_guarded_main_replay_batch_query_user_data = user_data;
  g_guarded_main_replay_batch_query = query;
}

bool TryGetCurrentFrameGuardedMainReplayBatch(
    uint64_t backend_frame_sequence,
    NativeGuestGuardedMainReplayBatch& batch) {
  batch = {};
  if (backend_frame_sequence == 0 ||
      g_guarded_main_replay_batch_query == nullptr ||
      !g_guarded_main_replay_batch_query(
          backend_frame_sequence, batch,
          g_guarded_main_replay_batch_query_user_data)) {
    batch = {};
    return false;
  }
  // The backend builds the plan; this is an independent re-check of the
  // invariants a caller is allowed to rely on, so a backend regression cannot
  // hand a title an unordered or unnormalized plan.
  if (!batch.full_output_normalization_proven ||
      !batch.scissor_only_normalization ||
      batch.backend_frame_sequence != backend_frame_sequence ||
      batch.normalized_output_width != 1280 ||
      batch.normalized_output_height != 720 ||
      // An empty plan is a legitimate result while families are still being
      // proven; `valid` distinguishes it from a replayable plan.
      batch.valid != !batch.ordered_tokens.empty() ||
      batch.ordered_tokens.size() !=
          size_t(batch.covered_token_count) ||
      batch.ordered_tokens.size() >
          NativeGuestGuardedMainReplayBatch::kMaximumTokens ||
      batch.families.size() >
          NativeGuestGuardedMainReplayBatch::kMaximumFamilies ||
      batch.families.size() != size_t(batch.discovered_family_count) ||
      batch.proven_family_count > batch.discovered_family_count ||
      batch.covered_token_count > batch.observed_token_count) {
    batch = {};
    return false;
  }
  if (batch.ordered_tokens.empty()) {
    return true;
  }
  uint64_t previous_token = 0;
  const NativeGuestTranslatedReplayTokenContext& first =
      batch.ordered_tokens.front().token;
  for (const NativeGuestGuardedReplayBatchToken& entry :
       batch.ordered_tokens) {
    const NativeGuestTranslatedReplayTokenContext& token = entry.token;
    if (!token.valid || !token.constant_contents_valid ||
        !token.resource_contents_valid ||
        !token.resources_stable_for_deferred_replay ||
        token.backend_frame_sequence != backend_frame_sequence ||
        token.opaque_token == 0 ||
        token.opaque_token <= previous_token ||
        entry.family == NativeGuestGuardedReplayFamily::kUnknown ||
        entry.vertex_shader_hash != token.vertex_shader_hash ||
        entry.pixel_shader_hash != token.pixel_shader_hash ||
        token.scissor_offset != std::array<int32_t, 2>{0, 0} ||
        token.scissor_extent != std::array<uint32_t, 2>{1280, 720} ||
        token.color_attachment_count != first.color_attachment_count ||
        token.color_attachment_formats != first.color_attachment_formats ||
        token.depth_attachment_format != first.depth_attachment_format ||
        token.stencil_attachment_format != first.stencil_attachment_format ||
        token.sample_count != first.sample_count ||
        token.sample_mask != first.sample_mask ||
        token.dynamic_rendering != first.dynamic_rendering) {
      batch = {};
      return false;
    }
    previous_token = token.opaque_token;
  }
  return true;
}

NativeGuestTranslatedReplayResult TryReplayNativeGuestTranslatedDraw(
    const NativeGuestOutputRenderContext& output_context,
    const NativeGuestTranslatedReplayTokenContext& token,
    const NativeGuestTranslatedReplayTarget& target) {
  if (g_translated_replay_backend == nullptr) {
    return NativeGuestTranslatedReplayResult::kUnsupportedBackend;
  }
  return g_translated_replay_backend(
      output_context, token, target, g_translated_replay_backend_user_data);
}

NativeGuestTranslatedReplayResult TryReplayNativeGuestTranslatedDrawBatch(
    const NativeGuestOutputRenderContext& output_context,
    const std::vector<NativeGuestTranslatedReplayTokenContext>& tokens,
    const NativeGuestTranslatedReplayTarget& target) {
  if (g_translated_replay_batch_backend == nullptr) {
    return NativeGuestTranslatedReplayResult::kUnsupportedBackend;
  }
  return g_translated_replay_batch_backend(
      output_context, tokens, target,
      g_translated_replay_batch_backend_user_data);
}

const char* NativeGuestTranslatedReplayResultName(
    NativeGuestTranslatedReplayResult result) {
  switch (result) {
    case NativeGuestTranslatedReplayResult::kSucceeded:
      return "succeeded";
    case NativeGuestTranslatedReplayResult::kUnsupportedBackend:
      return "unsupported_backend";
    case NativeGuestTranslatedReplayResult::kInvalidContext:
      return "invalid_context";
    case NativeGuestTranslatedReplayResult::kInvalidToken:
      return "invalid_token";
    case NativeGuestTranslatedReplayResult::kExpiredToken:
      return "expired_token";
    case NativeGuestTranslatedReplayResult::kResourcesNotStable:
      return "resources_not_stable";
    case NativeGuestTranslatedReplayResult::kIncompatibleTarget:
      return "incompatible_target";
    case NativeGuestTranslatedReplayResult::kIncompleteBindings:
      return "incomplete_bindings";
    case NativeGuestTranslatedReplayResult::kUnsupportedDraw:
      return "unsupported_draw";
    case NativeGuestTranslatedReplayResult::kRenderScopeUnavailable:
      return "render_scope_unavailable";
  }
  return "unknown";
}

bool MatchesNativeGuestDraw(const NativeGuestDrawContext& context) {
  NativeGuestDrawMatcher matcher =
      g_draw_matcher.load(std::memory_order_acquire);
  if (matcher == nullptr) {
    return false;
  }
  return matcher(
      context, g_draw_replacer_user_data.load(std::memory_order_acquire));
}

bool TryReplaceNativeGuestDraw(const NativeGuestDrawContext& context) {
  NativeGuestDrawRenderer renderer =
      g_draw_renderer.load(std::memory_order_acquire);
  if (renderer == nullptr) {
    return false;
  }
  return renderer(
      context, g_draw_replacer_user_data.load(std::memory_order_acquire));
}

bool IsNativeGuestOutputActive() {
  return g_native_output_active.load(std::memory_order_relaxed);
}

namespace {
std::atomic<double> g_wide_aspect{0.0};
}  // namespace

void SetNativeGuestOutputWideAspect(double aspect) {
  g_wide_aspect.store(aspect > 0.0 ? aspect : 0.0, std::memory_order_relaxed);
}

double GetNativeGuestOutputWideAspect() {
  return g_wide_aspect.load(std::memory_order_relaxed);
}

bool ApplyNativeGuestOutputWideAspect(uint32_t& guest_output_width, uint32_t guest_output_height,
                                      uint32_t& display_width, uint32_t& display_height) {
  const double aspect = g_wide_aspect.load(std::memory_order_relaxed);
  if (aspect <= 0.0 || !guest_output_width || !guest_output_height ||
      !HasNativeGuestOutputRenderer() ||
      !g_native_output_active.load(std::memory_order_relaxed)) {
    return false;
  }
  // Even width keeps every half-resolution derived target (bloom, SSAO,
  // menu snapshots) an exact divide.
  uint32_t wide_width =
      uint32_t(std::clamp(aspect * double(guest_output_height) + 0.5, 1.0, 16384.0)) & ~1u;
  if (wide_width <= guest_output_width) {
    return false;
  }
  guest_output_width = wide_width;
  // The display aspect is the actual pixel aspect: the presenter maps the
  // wide image edge-to-edge on a matching window.
  display_width = guest_output_width;
  display_height = guest_output_height;
  return true;
}

bool ShouldSuppressEmulatedDraws() {
  return REXCVAR_GET(native_render_suppress_emulated_draws) &&
         g_native_output_active.load(std::memory_order_relaxed);
}

bool ShouldSuppressExemptDepthOnlyDraws() {
  return REXCVAR_GET(native_render_suppress_exempt_depth_only);
}

bool ShouldSuppressMainPassForBenchmark(uint32_t surface_pitch) {
  if (!REXCVAR_GET(native_render_main_pass_suppression_benchmark) ||
      surface_pitch < 1280) {
    return false;
  }
  const int32_t delay_seconds =
      REXCVAR_GET(native_render_main_pass_suppression_benchmark_delay_seconds);
  if (delay_seconds > 0 &&
      rex::chrono::Clock::QueryHostUptimeMillis() <
          uint64_t(delay_seconds) * 1000u) {
    return false;
  }
  return true;
}

bool ShouldSuppressPassAtPitch(uint32_t surface_pitch) {
  switch (REXCVAR_GET(native_render_suppress_mode)) {
    case 0:
      return surface_pitch >= 1280;
    case 1:
      return true;
    case 3:
      // Portrait-window mode: the one-shot frontend portrait RTTs (Skate 3
      // census during the window: 1200/800/640/600/560 + small mips)
      // execute, while the 1152-wide main scene/postfx band, the
      // whole-pipeline cost at scaled resolutions, stays suppressed like
      // the framebuffer.
      return surface_pitch >= 1280 || surface_pitch == 1152;
    default:
      // Execute only the memory-composition passes the native renderer
      // samples: lightmap page composition (1024) and small composite
      // surfaces (<= 512, CAS outfit pieces). The <= 512 window also lets a
      // few tiny postfx pyramid mips through, negligible, and safer than
      // guessing which small surfaces matter.
      return !(surface_pitch == 1024 || surface_pitch <= 512);
  }
}

}  // namespace rex::graphics

namespace rex::graphics::nrhi {
namespace {

// Set once during app startup before any CreateShader (see native_rhi.h);
// lives in this shared TU so every backend reads the same value.
std::string g_shader_bytecode_cache_dir;

}  // namespace

void SetShaderBytecodeCacheDirectory(const char* path) {
  g_shader_bytecode_cache_dir = path != nullptr ? path : "";
}

const char* GetShaderBytecodeCacheDirectory() {
  return g_shader_bytecode_cache_dir.c_str();
}

}  // namespace rex::graphics::nrhi
