/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <rex/assert.h>
#include <rex/chrono/clock.h>
#include <rex/filesystem.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/packet_disassembler.h>
#include <rex/graphics/pipeline/texture/conversion.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/sampler_info.h>
#include <rex/graphics/trace_viewer.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>
#include <rex/platform.h>
#include <rex/string.h>
#include <rex/system.h>
#include <rex/system/kernel_state.h>
#include <rex/thread.h>
#include <rex/ui/imgui_drawer.h>
#include <rex/ui/immediate_drawer.h>
#include <rex/ui/presenter.h>
#include <rex/ui/ui_event.h>
#include <rex/ui/virtual_key.h>
#include <rex/ui/window.h>
#include <rex/ui/windowed_app_context.h>

#include <imgui.h>

REXCVAR_DEFINE_STRING(target_trace_file, "", "GPU", "Specifies the trace file to load");
REXCVAR_DEFINE_BOOL(trace_dump, false, "GPU", "Dump trace draw state to the log and exit");
REXCVAR_DEFINE_STRING(trace_dump_shader, "", "GPU",
                     "Shader hash to include detailed state for while dumping a trace");
REXCVAR_DEFINE_INT32(
    trace_dump_detail_start_command, -1, "GPU",
    "First command eligible for detailed shader state (-1 for the first matching draw)");
REXCVAR_DEFINE_INT32(
    trace_dump_detail_draw_count, 9, "GPU",
    "Maximum matching draws with detailed shader state (0 disables the limit)");
REXCVAR_DEFINE_INT32(trace_dump_end_command, -1, "GPU",
                    "Stop a textual trace dump after this command (-1 for the whole trace)");
REXCVAR_DEFINE_STRING(trace_dump_textures, "", "GPU",
                     "Directory for supported textures referenced by detailed trace draws");
REXCVAR_DEFINE_STRING(trace_dump_frame, "", "GPU",
                     "PPM output path after replaying the dumped trace");

namespace rex::graphics {

using namespace rex::graphics::xenos;

static const ImVec4 kColorError = ImVec4(255 / 255.0f, 0 / 255.0f, 0 / 255.0f, 255 / 255.0f);
static const ImVec4 kColorComment = ImVec4(42 / 255.0f, 179 / 255.0f, 0 / 255.0f, 255 / 255.0f);
static const ImVec4 kColorIgnored = ImVec4(100 / 255.0f, 100 / 255.0f, 100 / 255.0f, 255 / 255.0f);

static float Float16ToFloat(uint16_t value) {
  _Float16 half;
  std::memcpy(&half, &value, sizeof(half));
  return static_cast<float>(half);
}

static bool DumpTextureDDS(const TextureInfo& texture_info, const uint8_t* texture_data,
                           const std::filesystem::path& path) {
  const FormatInfo* format_info = texture_info.format_info();
  uint32_t fourcc;
  switch (texture_info.format) {
    case xenos::TextureFormat::k_DXT1:
      fourcc = UINT32_C(0x31545844);  // DXT1
      break;
    case xenos::TextureFormat::k_DXT2_3:
      fourcc = UINT32_C(0x33545844);  // DXT3
      break;
    case xenos::TextureFormat::k_DXT4_5:
      fourcc = UINT32_C(0x35545844);  // DXT5
      break;
    default:
      return false;
  }
  if (texture_info.dimension != xenos::DataDimension::k2DOrStacked ||
      texture_info.depth != 0) {
    return false;
  }

  const uint32_t width = texture_info.width + 1;
  const uint32_t height = texture_info.height + 1;
  const uint32_t width_blocks =
      (width + format_info->block_width - 1) / format_info->block_width;
  const uint32_t height_blocks =
      (height + format_info->block_height - 1) / format_info->block_height;
  const uint32_t bytes_per_block = format_info->bytes_per_block();
  std::vector<uint8_t> linear_data(size_t(width_blocks) * height_blocks * bytes_per_block);

  auto copy_block = [endianness = texture_info.endianness](
                        void* output, const void* input, size_t length) {
    texture_conversion::CopySwapBlock(endianness, output, input, length);
  };
  if (texture_info.is_tiled) {
    texture_conversion::UntileInfo untile_info{
        .offset_x = 0,
        .offset_y = 0,
        .width = width_blocks,
        .height = height_blocks,
        .input_pitch = texture_info.extent.block_pitch_h,
        .output_pitch = width_blocks,
        .input_format_info = format_info,
        .output_format_info = format_info,
        .copy_callback = copy_block,
    };
    texture_conversion::Untile(linear_data.data(), texture_data, &untile_info);
  } else {
    const size_t input_row_size = size_t(texture_info.extent.block_pitch_h) * bytes_per_block;
    const size_t output_row_size = size_t(width_blocks) * bytes_per_block;
    for (uint32_t y = 0; y < height_blocks; ++y) {
      copy_block(linear_data.data() + y * output_row_size,
                 texture_data + y * input_row_size, output_row_size);
    }
  }

  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) {
    return false;
  }

  // DDS_HEADER with a legacy DXT FourCC, including the four-byte magic.
  uint32_t dds[32] = {};
  dds[0] = UINT32_C(0x20534444);   // "DDS "
  dds[1] = 124;                    // DDS_HEADER size
  dds[2] = UINT32_C(0x000A1007);   // CAPS | HEIGHT | WIDTH | PIXELFORMAT | LINEARSIZE
  dds[3] = height;
  dds[4] = width;
  dds[5] = uint32_t(linear_data.size());
  dds[7] = 1;
  dds[19] = 32;                    // DDS_PIXELFORMAT size
  dds[20] = UINT32_C(0x00000004);  // DDPF_FOURCC
  dds[21] = fourcc;
  dds[27] = UINT32_C(0x00001000);  // DDSCAPS_TEXTURE

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  output.write(reinterpret_cast<const char*>(dds), sizeof(dds));
  output.write(reinterpret_cast<const char*>(linear_data.data()), linear_data.size());
  return output.good();
}

static bool DumpGuestOutputPPM(ui::Presenter* presenter,
                               const std::filesystem::path& path) {
  ui::RawImage image;
  if (!presenter || !presenter->CaptureGuestOutput(image)) {
    return false;
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  output << "P6\n" << image.width << " " << image.height << "\n255\n";
  for (uint32_t y = 0; y < image.height; ++y) {
    const uint8_t* row = image.data.data() + y * image.stride;
    for (uint32_t x = 0; x < image.width; ++x) {
      output.write(reinterpret_cast<const char*>(row + x * 4), 3);
    }
  }
  return output.good();
}

TraceViewer::TraceViewer(rex::ui::WindowedAppContext& app_context, const std::string_view name)
    : rex::ui::WindowedApp(app_context, name, "some.trace"), window_listener_(*this) {
  AddPositionalOption("target_trace_file");
}

TraceViewer::~TraceViewer() = default;

bool TraceViewer::OnInitialize() {
  std::string path = REXCVAR_GET(target_trace_file);

  if (path.empty()) {
    rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Warning, "No trace file specified");
    return false;
  }

  if (!Setup()) {
    rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error, "Unable to setup trace viewer");
    return false;
  }
  if (!Load(path)) {
    rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error,
                              "Unable to load trace file; not found?");
    return false;
  }
  return true;
}

bool TraceViewer::Setup() {
  enum : size_t {
    kZOrderImGui,
    kZOrderTraceViewerInput,
  };

  // Main display window.
  assert_true(app_context().IsInUIThread());
  window_ = rex::ui::Window::Create(app_context(), "xenia-gpu-trace-viewer", 1920, 1080);
  window_->AddListener(&window_listener_);
  window_->AddInputListener(&window_listener_, kZOrderTraceViewerInput);
  if (!window_->Open()) {
    REXGPU_ERROR("Failed to open the main window");
    return false;
  }

  emulator_ = std::make_unique<Runtime>("");
  emulator_->set_app_context(&app_context());
  emulator_->set_display_window(window_.get());
  RuntimeConfig runtime_config;
  runtime_config.graphics = CreateGraphicsSystem();
  X_STATUS result = emulator_->Setup(std::move(runtime_config));
  if (XFAILED(result)) {
    REXGPU_ERROR("Failed to setup emulator: {:08X}", result);
    return false;
  }
  memory_ = emulator_->memory();
  graphics_system_ = dynamic_cast<GraphicsSystem*>(emulator_->graphics_system());
  if (!graphics_system_) {
    REXGPU_ERROR("Runtime did not create a compatible graphics system");
    return false;
  }

  player_ = std::make_unique<TracePlayer>(graphics_system_);

  // Setup drawing to the window.
  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    REXGPU_ERROR("Failed to initialize the presenter");
    return false;
  }
  rex::ui::GraphicsProvider& graphics_provider = *graphics_system_->provider();
  immediate_drawer_ = graphics_provider.CreateImmediateDrawer();
  if (!immediate_drawer_) {
    REXGPU_ERROR("Failed to initialize the immediate drawer");
    return false;
  }
  immediate_drawer_->SetPresenter(presenter);
  imgui_drawer_ = std::make_unique<rex::ui::ImGuiDrawer>(window_.get(), kZOrderImGui);
  imgui_drawer_->SetPresenterAndImmediateDrawer(presenter, immediate_drawer_.get());
  trace_viewer_dialog_ =
      std::unique_ptr<TraceViewerDialog>(new TraceViewerDialog(imgui_drawer_.get(), *this));
  window_->SetPresenter(presenter);

  return true;
}

void TraceViewer::TraceViewerWindowListener::OnClosing(rex::ui::UIEvent& e) {
  trace_viewer_.app_context().QuitFromUIThread();
}

void TraceViewer::TraceViewerWindowListener::OnKeyDown(rex::ui::KeyEvent& e) {
  switch (e.virtual_key()) {
    case rex::ui::VirtualKey::kF5:
      trace_viewer_.graphics_system_->ClearCaches();
      break;
    default:
      return;
  }
  e.set_handled(true);
}

void TraceViewer::TraceViewerDialog::OnDraw(ImGuiIO& io) {
  trace_viewer_.DrawUI();
}

bool TraceViewer::Load(const std::string_view trace_file_path) {
  window_->SetTitle("Xenia GPU Trace Viewer: " + std::string(trace_file_path));

  if (!player_->Open(trace_file_path)) {
    REXGPU_ERROR("Could not load trace file");
    return false;
  }

  if (REXCVAR_GET(trace_dump)) {
    DumpTraceToLog();
    app_context().QuitFromUIThread();
  }

  return true;
}

void TraceViewer::DumpTraceToLog() {
  struct ShaderStats {
    uint32_t draw_count = 0;
    int first_frame = -1;
    int first_command = -1;
    size_t ucode_dwords = 0;
    size_t texture_bindings = 0;
  };

  uint64_t detail_shader_hash = 0;
  const std::string& detail_shader = REXCVAR_GET(trace_dump_shader);
  if (!detail_shader.empty()) {
    try {
      detail_shader_hash = std::stoull(detail_shader, nullptr, 16);
    } catch (...) {
      REXGPU_ERROR("Invalid trace_dump_shader hash: {}", detail_shader);
      return;
    }
  }

  std::map<uint64_t, ShaderStats> pixel_shader_stats;
  uint32_t total_draws = 0;
  uint32_t detailed_draws = 0;
  const int32_t detail_start_command =
      REXCVAR_GET(trace_dump_detail_start_command);
  const int32_t detail_draw_count =
      REXCVAR_GET(trace_dump_detail_draw_count);
  const int32_t end_command = REXCVAR_GET(trace_dump_end_command);
  const std::filesystem::path texture_dump_path = REXCVAR_GET(trace_dump_textures);
  std::set<std::pair<uint32_t, uint32_t>> dumped_textures;

  for (int frame_index = 0; frame_index < player_->frame_count(); ++frame_index) {
    if (player_->current_frame_index() != frame_index) {
      player_->SeekFrame(frame_index);
      player_->WaitOnPlayback();
      player_->SeekCommand(-1);
    }
    const TraceReader::Frame* frame = player_->current_frame();
    REXGPU_INFO("TRACE_FRAME frame={} commands={}", frame_index, frame->commands.size());

    for (int command_index = 0; command_index < int(frame->commands.size()); ++command_index) {
      if (end_command >= 0 && command_index > end_command) {
        break;
      }
      const TraceReader::Frame::Command& command = frame->commands[command_index];
      if (command.type != TraceReader::Frame::Command::Type::kDraw) {
        continue;
      }

      player_->SeekCommand(command_index);
      player_->WaitOnPlayback();
      ++total_draws;

      CommandProcessor* command_processor = graphics_system_->command_processor();
      Shader* vertex_shader = command_processor->active_vertex_shader();
      Shader* pixel_shader = command_processor->active_pixel_shader();
      const uint64_t vertex_shader_hash =
          vertex_shader ? vertex_shader->ucode_data_hash() : 0;
      const uint64_t pixel_shader_hash = pixel_shader ? pixel_shader->ucode_data_hash() : 0;

      uint32_t index_count = 0;
      uint32_t primitive_type = 0;
      uint32_t index_buffer_base = 0;
      uint32_t index_buffer_size = 0;
      bool indexed = false;
      const uint8_t* packet_head = command.head_ptr + sizeof(PacketStartCommand);
      const uint32_t packet = memory::load_and_swap<uint32_t>(packet_head);
      const uint32_t opcode = (packet >> 8) & 0x7F;
      if (opcode == PM4_DRAW_INDX) {
        const uint32_t draw_initiator =
            memory::load_and_swap<uint32_t>(packet_head + 8);
        index_count = draw_initiator >> 16;
        primitive_type = draw_initiator & 0x3F;
        indexed = ((draw_initiator >> 6) & 0x3) == 0;
        if (indexed) {
          index_buffer_base =
              memory::load_and_swap<uint32_t>(packet_head + 12);
          uint32_t index_buffer_elements =
              memory::load_and_swap<uint32_t>(packet_head + 16) & 0x00FFFFFF;
          const bool index_32bit = ((draw_initiator >> 11) & 1) != 0;
          index_buffer_size =
              index_buffer_elements * (index_32bit ? 4 : 2);
        }
      } else if (opcode == PM4_DRAW_INDX_2) {
        const uint32_t draw_initiator =
            memory::load_and_swap<uint32_t>(packet_head + 4);
        index_count = draw_initiator >> 16;
        primitive_type = draw_initiator & 0x3F;
      }

      ShaderStats& stats = pixel_shader_stats[pixel_shader_hash];
      ++stats.draw_count;
      if (stats.first_frame < 0) {
        stats.first_frame = frame_index;
        stats.first_command = command_index;
        stats.ucode_dwords = pixel_shader ? pixel_shader->ucode_dword_count() : 0;
        stats.texture_bindings = pixel_shader ? pixel_shader->texture_bindings().size() : 0;
      }

      REXGPU_INFO(
          "TRACE_DRAW frame={} command={} indices={} primitive={} indexed={} "
          "index_base={:08X} index_bytes={} vs={:016X} ps={:016X} "
          "ps_textures={}",
          frame_index, command_index, index_count, primitive_type, indexed,
          index_buffer_base, index_buffer_size, vertex_shader_hash,
          pixel_shader_hash,
          pixel_shader ? pixel_shader->texture_bindings().size() : 0);

      const bool before_detail_start =
          detail_start_command >= 0 && command_index < detail_start_command;
      const bool detail_limit_reached =
          detail_draw_count > 0 &&
          detailed_draws >= static_cast<uint32_t>(detail_draw_count);
      if (!pixel_shader || pixel_shader_hash != detail_shader_hash ||
          before_detail_start || detail_limit_reached) {
        continue;
      }
      ++detailed_draws;

      RegisterFile& regs = *graphics_system_->register_file();
      const reg::SQ_VS_CONST vs_constants = regs.Get<reg::SQ_VS_CONST>();
      const reg::SQ_PS_CONST ps_constants = regs.Get<reg::SQ_PS_CONST>();
      const reg::RB_COLOR_INFO color_info =
          regs.Get<reg::RB_COLOR_INFO>();
      const reg::RB_COLORCONTROL color_control =
          regs.Get<reg::RB_COLORCONTROL>();
      const reg::RB_DEPTHCONTROL depth_control =
          regs.Get<reg::RB_DEPTHCONTROL>();
      const reg::RB_BLENDCONTROL blend_control =
          regs.Get<reg::RB_BLENDCONTROL>();
      const reg::PA_SU_SC_MODE_CNTL mode_control =
          regs.Get<reg::PA_SU_SC_MODE_CNTL>();
      REXGPU_INFO(
          "TRACE_SHADER_DETAIL draw={} frame={} command={} ps={:016X} ucode_dwords={} "
          "constant_base={} constant_size={} texture_bindings={}",
          detailed_draws, frame_index, command_index, pixel_shader_hash,
          pixel_shader->ucode_dword_count(), ps_constants.base, ps_constants.size + 1,
          pixel_shader->texture_bindings().size());
      REXGPU_INFO(
          "TRACE_RASTER color_info={:08X} color_format={} exp_bias={} "
          "color_control={:08X} blend_control={:08X} "
          "depth_control={:08X} mode_control={:08X}",
          color_info.value, uint32_t(color_info.color_format),
          int32_t(color_info.color_exp_bias), color_control.value,
          blend_control.value, depth_control.value, mode_control.value);

      if (vertex_shader) {
        REXGPU_INFO(
            "TRACE_VERTEX_STATE vs={:016X} constant_base={} constant_size={} "
            "vertex_bindings={}",
            vertex_shader_hash, vs_constants.base, vs_constants.size + 1,
            vertex_shader->vertex_bindings().size());
        for (const Shader::VertexBinding& binding :
             vertex_shader->vertex_bindings()) {
          const xenos::xe_gpu_vertex_fetch_t fetch =
              regs.GetVertexFetch(binding.fetch_constant);
          const uint32_t byte_address = fetch.address << 2;
          const uint32_t byte_size = fetch.size << 2;
          const uint32_t sample_word_count =
              std::min(fetch.size, binding.stride_words * 2);
          const uint8_t* fetch_data =
              memory_->TranslatePhysical(byte_address);
          uint64_t sample_hash = UINT64_C(1469598103934665603);
          std::string sample_words;
          sample_words.reserve(sample_word_count * 9);
          for (uint32_t word = 0; word < sample_word_count; ++word) {
            const uint32_t value = xenos::GpuSwap(
                memory::load<uint32_t>(fetch_data + word * sizeof(uint32_t)),
                fetch.endian);
            sample_hash =
                (sample_hash ^ value) * UINT64_C(1099511628211);
            if (!sample_words.empty()) {
              sample_words.push_back(',');
            }
            char formatted_word[9];
            std::snprintf(formatted_word, sizeof(formatted_word), "%08X",
                          value);
            sample_words.append(formatted_word);
          }
          uint32_t first_nonzero_vertex = UINT32_MAX;
          std::string first_nonzero_words;
          const uint32_t stride_words = binding.stride_words;
          if (stride_words != 0) {
            const uint32_t vertex_count = fetch.size / stride_words;
            for (uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
              const uint8_t* vertex_data =
                  fetch_data + size_t(vertex) * stride_words *
                                   sizeof(uint32_t);
              bool nonzero = false;
              for (uint32_t word = 0; word < stride_words; ++word) {
                nonzero |= xenos::GpuSwap(
                               memory::load<uint32_t>(
                                   vertex_data + word * sizeof(uint32_t)),
                               fetch.endian) != 0;
              }
              if (!nonzero) {
                continue;
              }
              first_nonzero_vertex = vertex;
              first_nonzero_words.reserve(stride_words * 9);
              for (uint32_t word = 0; word < stride_words; ++word) {
                if (!first_nonzero_words.empty()) {
                  first_nonzero_words.push_back(',');
                }
                const uint32_t value = xenos::GpuSwap(
                    memory::load<uint32_t>(
                        vertex_data + word * sizeof(uint32_t)),
                    fetch.endian);
                char formatted_word[9];
                std::snprintf(formatted_word, sizeof(formatted_word), "%08X",
                              value);
                first_nonzero_words.append(formatted_word);
              }
              break;
            }
          }
          REXGPU_INFO(
              "TRACE_VERTEX_BUFFER vf={} binding={} address={:08X} size={} "
              "endian={} stride_words={} attributes={} sample_hash={:016X} "
              "sample_words={} first_nonzero_vertex={} "
              "first_nonzero_words={}",
              binding.fetch_constant, binding.binding_index, byte_address,
              byte_size, uint32_t(fetch.endian), binding.stride_words,
              binding.attributes.size(), sample_hash, sample_words,
              first_nonzero_vertex, first_nonzero_words);
          for (size_t attribute_index = 0;
               attribute_index < binding.attributes.size();
               ++attribute_index) {
            const ParsedVertexFetchInstruction& instruction =
                binding.attributes[attribute_index].fetch_instr;
            const auto& attributes = instruction.attributes;
            const InstructionResult& result = instruction.result;
            REXGPU_INFO(
                "TRACE_VERTEX_ATTRIBUTE vf={} binding={} attribute={} "
                "format={} offset={} stride_words={} exp_adjust={} "
                "signed={} integer={} mini={} target={} target_index={} "
                "write_mask={:X} swizzle={}{}{}{}",
                binding.fetch_constant, binding.binding_index,
                attribute_index, uint32_t(attributes.data_format),
                attributes.offset, attributes.stride,
                attributes.exp_adjust, attributes.is_signed,
                attributes.is_integer, instruction.is_mini_fetch,
                uint32_t(result.storage_target), result.storage_index,
                result.original_write_mask,
                GetCharForSwizzle(result.components[0]),
                GetCharForSwizzle(result.components[1]),
                GetCharForSwizzle(result.components[2]),
                GetCharForSwizzle(result.components[3]));
          }
        }

        const Shader::ConstantRegisterMap& constants =
            vertex_shader->constant_register_map();
        for (uint32_t constant = 0; constant < 256; ++constant) {
          if (!(constants.float_bitmap[constant / 64] &
                (UINT64_C(1) << (constant % 64)))) {
            continue;
          }
          const uint32_t register_index =
              XE_GPU_REG_SHADER_CONSTANT_000_X +
              4 * (vs_constants.base + constant);
          REXGPU_INFO(
              "TRACE_VERTEX_CONSTANT c{}={},{},{},{}", constant,
              regs.Get<float>(register_index + 0),
              regs.Get<float>(register_index + 1),
              regs.Get<float>(register_index + 2),
              regs.Get<float>(register_index + 3));
        }
      }

      std::set<uint32_t> dumped_texture_slots;
      for (const Shader::TextureBinding& binding : pixel_shader->texture_bindings()) {
        const uint32_t slot = binding.fetch_constant;
        if (!dumped_texture_slots.insert(slot).second) {
          continue;
        }
        const xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(slot);
        TextureInfo texture_info;
        SamplerInfo sampler_info{};
        const bool valid_type =
            fetch.type == xenos::FetchConstantType::kTexture ||
            (REXCVAR_GET(gpu_allow_invalid_fetch_constants) &&
             fetch.type == xenos::FetchConstantType::kInvalidTexture);
        const bool valid_info = valid_type && TextureInfo::Prepare(fetch, &texture_info);
        const bool valid_sampler =
            valid_type &&
            SamplerInfo::Prepare(fetch, binding.fetch_instr, &sampler_info);

        const uint32_t* raw_fetch =
            &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * slot];
        if (!valid_info) {
          REXGPU_INFO(
              "TRACE_TEXTURE slot={} binding={} invalid type={} raw={:08X},{:08X},{:08X},"
              "{:08X},{:08X},{:08X}",
              slot, binding.binding_index, uint32_t(fetch.type), raw_fetch[0], raw_fetch[1],
              raw_fetch[2], raw_fetch[3], raw_fetch[4], raw_fetch[5]);
          continue;
        }

        const size_t sample_size = texture_info.memory.base_size;
        const uint8_t* texture_data =
            memory_->TranslatePhysical(texture_info.memory.base_address);
        uint64_t sample_hash = UINT64_C(1469598103934665603);
        size_t zero_bytes = 0;
        for (size_t i = 0; i < sample_size; ++i) {
          const uint8_t value = texture_data[i];
          zero_bytes += value == 0;
          sample_hash = (sample_hash ^ value) * UINT64_C(1099511628211);
        }
        REXGPU_INFO(
            "TRACE_TEXTURE slot={} binding={} address={:08X} size={} format={} dimensions={}x{}x{} "
            "tiled={} mips={}-{} mip_address={:08X} mip_size={} endian={} swizzle={:03X} "
            "sample_bytes={} zero_bytes={} sample_hash={:016X} "
            "raw={:08X},{:08X},{:08X},{:08X},{:08X},{:08X}",
            slot, binding.binding_index, texture_info.memory.base_address,
            texture_info.memory.base_size, texture_info.format_info()->name, texture_info.width + 1,
            texture_info.height + 1, texture_info.depth + 1, texture_info.is_tiled,
            texture_info.mip_min_level, texture_info.mip_max_level,
            texture_info.memory.mip_address, texture_info.memory.mip_size,
            uint32_t(texture_info.endianness), fetch.swizzle, sample_size, zero_bytes,
            sample_hash, raw_fetch[0], raw_fetch[1], raw_fetch[2], raw_fetch[3], raw_fetch[4],
            raw_fetch[5]);
        if (valid_sampler) {
          REXGPU_INFO(
              "TRACE_SAMPLER slot={} binding={} min={} mag={} mip={} "
              "clamp={}/{}/{} aniso={} border={} lod_bias={} mips={}-{}",
              slot, binding.binding_index,
              uint32_t(sampler_info.min_filter),
              uint32_t(sampler_info.mag_filter),
              uint32_t(sampler_info.mip_filter),
              uint32_t(sampler_info.clamp_u),
              uint32_t(sampler_info.clamp_v),
              uint32_t(sampler_info.clamp_w),
              uint32_t(sampler_info.aniso_filter),
              uint32_t(sampler_info.border_color), sampler_info.lod_bias,
              sampler_info.mip_min_level, sampler_info.mip_max_level);
        } else {
          REXGPU_INFO("TRACE_SAMPLER slot={} binding={} invalid", slot,
                      binding.binding_index);
        }

        if (!texture_dump_path.empty() &&
            dumped_textures.emplace(texture_info.memory.base_address,
                                    uint32_t(texture_info.format))
                .second) {
          char texture_filename[128];
          std::snprintf(texture_filename, sizeof(texture_filename),
                        "command_%04d_slot_%02u_addr_%08X.dds", command_index, slot,
                        texture_info.memory.base_address);
          const std::filesystem::path texture_path =
              texture_dump_path / texture_filename;
          if (DumpTextureDDS(texture_info, texture_data, texture_path)) {
            REXGPU_INFO("TRACE_TEXTURE_DUMP command={} slot={} path={}", command_index,
                        slot, texture_path.string());
          } else {
            REXGPU_WARN(
                "TRACE_TEXTURE_DUMP unsupported or failed command={} slot={} format={}",
                command_index, slot, texture_info.format_info()->name);
          }
        }
      }

      const Shader::ConstantRegisterMap& constants = pixel_shader->constant_register_map();
      for (uint32_t constant = 0; constant < 256; ++constant) {
        if (!(constants.float_bitmap[constant / 64] &
              (UINT64_C(1) << (constant % 64)))) {
          continue;
        }
        const uint32_t register_index =
            XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * (ps_constants.base + constant);
        REXGPU_INFO("TRACE_CONSTANT c{}={},{},{},{}", constant,
                    regs.Get<float>(register_index + 0), regs.Get<float>(register_index + 1),
                    regs.Get<float>(register_index + 2), regs.Get<float>(register_index + 3));
      }
    }
  }

  std::vector<std::pair<uint64_t, ShaderStats>> sorted_stats(pixel_shader_stats.begin(),
                                                             pixel_shader_stats.end());
  std::sort(sorted_stats.begin(), sorted_stats.end(), [](const auto& left, const auto& right) {
    return left.second.draw_count > right.second.draw_count;
  });
  REXGPU_INFO("TRACE_SUMMARY frames={} draws={} pixel_shaders={}", player_->frame_count(),
              total_draws, sorted_stats.size());
  for (const auto& [hash, stats] : sorted_stats) {
    REXGPU_INFO(
        "TRACE_SHADER ps={:016X} draws={} first={}:{} ucode_dwords={} texture_bindings={}", hash,
        stats.draw_count, stats.first_frame, stats.first_command, stats.ucode_dwords,
        stats.texture_bindings);
  }

  const std::filesystem::path frame_dump_path = REXCVAR_GET(trace_dump_frame);
  if (!frame_dump_path.empty()) {
    const TraceReader::Frame* frame = player_->current_frame();
    player_->SeekCommand(-1);
    player_->SeekCommand(int(frame->commands.size()) - 1);
    player_->WaitOnPlayback();
    if (DumpGuestOutputPPM(graphics_system_->presenter(), frame_dump_path)) {
      REXGPU_INFO("TRACE_FRAME_DUMP path={}", frame_dump_path.string());
    } else {
      REXGPU_ERROR("TRACE_FRAME_DUMP failed path={}", frame_dump_path.string());
    }
  }
}

void TraceViewer::DrawMultilineString(const std::string_view str) {
  size_t i = 0;
  bool done = false;
  while (!done && i < str.size()) {
    size_t next_i = str.find('\n', i);
    if (next_i == std::string::npos) {
      done = true;
      next_i = str.size() - 1;
    }
    auto line = str.substr(i, next_i - i);
    ImGui::Text("%s", std::string(line).c_str());
    i = next_i + 1;
  }
}

void TraceViewer::DrawUI() {
  // ImGui::ShowDemoWindow();

  DrawControllerUI();
  DrawCommandListUI();
  DrawStateUI();
  DrawPacketDisassemblerUI();
}

void TraceViewer::DrawControllerUI() {
  ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(340, 60));
  ImGui::SetNextWindowBgAlpha(kWindowBgAlpha);
  if (!ImGui::Begin("Controller", nullptr)) {
    ImGui::End();
    return;
  }

  int target_frame = player_->current_frame_index();
  if (ImGui::Button("|<<")) {
    target_frame = 0;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Reset to first frame");
  }
  ImGui::SameLine();
  ImGui::PushButtonRepeat(true);
  if (ImGui::Button(">>", ImVec2(0, 0))) {
    if (target_frame + 1 < player_->frame_count()) {
      ++target_frame;
    }
  }
  ImGui::PopButtonRepeat();
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Next frame (hold for continuous)");
  }
  ImGui::SameLine();
  if (ImGui::Button(">>|")) {
    target_frame = player_->frame_count() - 1;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Skip to last frame");
  }
  if (player_->is_playing_trace()) {
    // Don't allow the user to change the frame index just yet...
    // TODO: Find a way to disable the slider below.
    target_frame = player_->current_frame_index();
  }

  ImGui::SameLine();
  ImGui::SliderInt("##", &target_frame, 0, player_->frame_count() - 1);
  if (target_frame != player_->current_frame_index() && !player_->is_playing_trace()) {
    player_->SeekFrame(target_frame);
  }
  ImGui::End();
}

void TraceViewer::DrawPacketDisassemblerUI() {
  ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowPos(ImVec2(float(window_->GetActualLogicalWidth()) - 500 - 5, 5),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(500, 300));
  ImGui::SetNextWindowBgAlpha(kWindowBgAlpha);
  if (!ImGui::Begin("Packet Disassembler", nullptr)) {
    ImGui::End();
    return;
  }
  if (!player_->current_frame() || player_->current_command_index() == -1) {
    ImGui::Text("No frame/command selected");
    ImGui::End();
    return;
  }

  auto frame = player_->current_frame();
  const auto& command = frame->commands[player_->current_command_index()];
  const uint8_t* start_ptr = command.start_ptr;
  const uint8_t* end_ptr = command.end_ptr;

  ImGui::Text("Frame #%d, command %d", player_->current_frame_index(),
              player_->current_command_index());
  ImGui::Separator();
  ImGui::BeginChild("packet_disassembler_list");
  const PacketStartCommand* pending_packet = nullptr;
  auto trace_ptr = start_ptr;
  while (trace_ptr < end_ptr) {
    auto type = static_cast<TraceCommandType>(memory::load<uint32_t>(trace_ptr));
    switch (type) {
      case TraceCommandType::kPrimaryBufferStart: {
        auto cmd = reinterpret_cast<const PrimaryBufferStartCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->count * 4;
        ImGui::BulletText("PrimaryBufferStart");
        break;
      }
      case TraceCommandType::kPrimaryBufferEnd: {
        auto cmd = reinterpret_cast<const PrimaryBufferEndCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        ImGui::BulletText("PrimaryBufferEnd");
        break;
      }
      case TraceCommandType::kIndirectBufferStart: {
        auto cmd = reinterpret_cast<const IndirectBufferStartCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->count * 4;
        ImGui::BulletText("IndirectBufferStart");
        break;
      }
      case TraceCommandType::kIndirectBufferEnd: {
        auto cmd = reinterpret_cast<const IndirectBufferEndCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        ImGui::BulletText("IndirectBufferEnd");
        break;
      }
      case TraceCommandType::kPacketStart: {
        auto cmd = reinterpret_cast<const PacketStartCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->count * 4;
        pending_packet = cmd;
        break;
      }
      case TraceCommandType::kPacketEnd: {
        auto cmd = reinterpret_cast<const PacketEndCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        if (pending_packet) {
          PacketInfo packet_info = {0};
          if (PacketDisassembler::DisasmPacket(
                  reinterpret_cast<const uint8_t*>(pending_packet) + sizeof(PacketStartCommand),
                  &packet_info)) {
            if (packet_info.predicated) {
              ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
            }
            ImGui::BulletText("%s", packet_info.type_info->name);
            ImGui::TreePush((const char*)0);
            for (auto action : packet_info.actions) {
              switch (action.type) {
                case PacketAction::Type::kRegisterWrite: {
                  auto register_info =
                      rex::graphics::RegisterFile::GetRegisterInfo(action.register_write.index);
                  ImGui::Columns(2);
                  ImGui::Text("%.4X %s", action.register_write.index,
                              register_info ? register_info->name : "???");
                  ImGui::NextColumn();
                  if (!register_info || register_info->type == RegisterInfo::Type::kDword) {
                    ImGui::Text("%.8X", action.register_write.value);
                  } else {
                    ImGui::Text("%8f",
                                rex::memory::Reinterpret<float>(action.register_write.value));
                  }
                  ImGui::Columns(1);
                  break;
                }
                case PacketAction::Type::kSetBinMask: {
                  ImGui::Text("%.16" PRIX64, action.set_bin_mask.value);
                  break;
                }
                case PacketAction::Type::kSetBinSelect: {
                  ImGui::Text("%.16" PRIX64, action.set_bin_select.value);
                  break;
                }
              }
            }
            ImGui::TreePop();
            if (packet_info.predicated) {
              ImGui::PopStyleColor();
            }
          } else {
            ImGui::BulletText("<invalid packet>");
          }
          pending_packet = nullptr;
        }
        break;
      }
      case TraceCommandType::kMemoryRead: {
        auto cmd = reinterpret_cast<const MemoryCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        // ImGui::BulletText("MemoryRead");
        break;
      }
      case TraceCommandType::kMemoryWrite: {
        auto cmd = reinterpret_cast<const MemoryCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        // ImGui::BulletText("MemoryWrite");
        break;
      }
      case TraceCommandType::kEdramSnapshot: {
        auto cmd = reinterpret_cast<const EdramSnapshotCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        // ImGui::BulletText("EdramSnapshot");
        break;
      }
      case TraceCommandType::kEvent: {
        auto cmd = reinterpret_cast<const EventCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        switch (cmd->event_type) {
          case EventCommand::Type::kSwap: {
            ImGui::BulletText("<swap>");
            break;
          }
        }
        break;
      }
      case TraceCommandType::kRegisters: {
        auto cmd = reinterpret_cast<const RegistersCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        // ImGui::BulletText("Registers");
        break;
      }
      case TraceCommandType::kGammaRamp: {
        auto cmd = reinterpret_cast<const GammaRampCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        // ImGui::BulletText("GammaRamp");
        break;
      }
    }
  }
  ImGui::EndChild();
  ImGui::End();
}

int TraceViewer::RecursiveDrawCommandBufferUI(const TraceReader::Frame* frame,
                                              TraceReader::CommandBuffer* buffer) {
  int selected_id = -1;
  int column_width = int(ImGui::GetContentRegionMax().x);

  for (size_t i = 0; i < buffer->commands.size(); i++) {
    switch (buffer->commands[i].type) {
      case TraceReader::CommandBuffer::Command::Type::kBuffer: {
        auto subtree = buffer->commands[i].command_subtree.get();
        if (!subtree->commands.size()) {
          continue;
        }

        ImGui::PushID(int(i));
        if (ImGui::TreeNode((void*)0, "Indirect Buffer %" PRIu64, i)) {
          ImGui::Indent();
          auto id = RecursiveDrawCommandBufferUI(frame, buffer->commands[i].command_subtree.get());
          ImGui::Unindent();
          ImGui::TreePop();

          if (id != -1) {
            selected_id = id;
          }
        }
        ImGui::PopID();
      } break;

      case TraceReader::CommandBuffer::Command::Type::kCommand: {
        uint32_t command_id = buffer->commands[i].command_id;

        const auto& command = frame->commands[command_id];
        bool is_selected = command_id == player_->current_command_index();
        const char* label;
        switch (command.type) {
          case TraceReader::Frame::Command::Type::kDraw:
            label = "Draw";
            break;
          case TraceReader::Frame::Command::Type::kSwap:
            label = "Swap";
            break;
        }

        ImGui::PushID(command_id);
        if (ImGui::Selectable(label, &is_selected)) {
          selected_id = command_id;
        }
        ImGui::SameLine(column_width - 60.0f);
        ImGui::Text("%d", command_id);
        ImGui::PopID();
        // if (did_seek && target_command == i) {
        //   ImGui::SetScrollPosHere();
        // }
      } break;
    }
  }

  return selected_id;
}

void TraceViewer::DrawCommandListUI() {
  ImGui::SetNextWindowPos(ImVec2(5, 70), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(200, 640));
  ImGui::SetNextWindowBgAlpha(kWindowBgAlpha);
  if (!ImGui::Begin("Command List", nullptr)) {
    ImGui::End();
    return;
  }

  static const TracePlayer::Frame* previous_frame = nullptr;
  auto frame = player_->current_frame();
  if (!frame) {
    ImGui::End();
    return;
  }
  bool did_seek = false;
  if (previous_frame != frame) {
    did_seek = true;
    previous_frame = frame;
  }
  int command_count = int(frame->commands.size());
  int target_command = player_->current_command_index();
  int column_width = int(ImGui::GetContentRegionMax().x);
  ImGui::Text("Frame #%d", player_->current_frame_index());
  ImGui::Separator();
  if (ImGui::Button("reset")) {
    target_command = -1;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Reset to before any frame commands");
  }
  ImGui::SameLine();
  ImGui::PushButtonRepeat(true);
  if (ImGui::Button("prev", ImVec2(0, 0))) {
    if (target_command >= 0) {
      --target_command;
    }
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Move to the previous command (hold)");
  }
  ImGui::SameLine();
  if (ImGui::Button("next", ImVec2(0, 0))) {
    if (target_command < command_count - 1) {
      ++target_command;
    }
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Move to the next command (hold)");
  }
  ImGui::PopButtonRepeat();
  ImGui::SameLine();
  if (ImGui::Button("end")) {
    target_command = command_count - 1;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Move to the last command");
  }
  if (player_->is_playing_trace()) {
    // Don't allow the user to change the command index just yet...
    // TODO: Find a way to disable the slider below.
    target_command = player_->current_command_index();
  }

  ImGui::PushItemWidth(float(column_width - 15));
  ImGui::SliderInt("##", &target_command, -1, command_count - 1);
  ImGui::PopItemWidth();

  if (target_command != player_->current_command_index() && !player_->is_playing_trace()) {
    did_seek = true;
    player_->SeekCommand(target_command);
  }
  ImGui::Separator();
  ImGui::BeginChild("command_list");
  ImGui::PushID(-1);
  bool is_selected = player_->current_command_index() == -1;
  if (ImGui::Selectable("<start>", &is_selected)) {
    player_->SeekCommand(-1);
  }
  ImGui::PopID();
  if (did_seek && target_command == -1) {
    ImGui::SetScrollHereY(0.5f);
  }

  auto id = RecursiveDrawCommandBufferUI(frame, frame->command_tree.get());
  if (id != -1 && id != player_->current_command_index() && !player_->is_playing_trace()) {
    player_->SeekCommand(id);
  }

  ImGui::EndChild();
  ImGui::End();
}

TraceViewer::ShaderDisplayType TraceViewer::DrawShaderTypeUI() {
  static ShaderDisplayType shader_display_type = ShaderDisplayType::kUcode;
  ImGui::RadioButton("ucode", reinterpret_cast<int*>(&shader_display_type),
                     static_cast<int>(ShaderDisplayType::kUcode));
  ImGui::SameLine();
  ImGui::RadioButton("translated", reinterpret_cast<int*>(&shader_display_type),
                     static_cast<int>(ShaderDisplayType::kTranslated));
  ImGui::SameLine();
  ImGui::RadioButton("disasm", reinterpret_cast<int*>(&shader_display_type),
                     static_cast<int>(ShaderDisplayType::kHostDisasm));
  return shader_display_type;
}

void TraceViewer::DrawShaderUI(Shader* shader, ShaderDisplayType display_type) {
  // Must be prepared for advanced display modes.
  // FIXME(Triang3l): This should display the actual translation used in the
  // draw, but it may depend on multiple backend-related factors, including
  // drawing multiple times with multiple modifications, even depending on
  // values obtained during translation of other modifications (for instance,
  // a memexporting shader can be executed both as a vertex shader (to draw the
  // points) and as a compute shader (to actually export) if the host doesn't
  // support writes from vertex shaders.
  const Shader::Translation* translation = nullptr;
  if (display_type != ShaderDisplayType::kUcode) {
    for (const auto& translation_pair : shader->translations()) {
      if (translation_pair.second->is_valid()) {
        translation = translation_pair.second;
      }
    }
    if (!translation) {
      ImGui::TextColored(kColorError, "ERROR: shader error during parsing/translation");
      return;
    }
  }

  switch (display_type) {
    case ShaderDisplayType::kUcode: {
      DrawMultilineString(shader->ucode_disassembly());
      break;
    }
    case ShaderDisplayType::kTranslated: {
      const auto& str = translation->GetTranslatedBinaryString();
      size_t i = 0;
      bool done = false;
      while (!done && i < str.size()) {
        size_t next_i = str.find('\n', i);
        if (next_i == std::string::npos) {
          done = true;
          next_i = str.size() - 1;
        }
        auto line = str.substr(i, next_i - i);
        if (line.find("//") != std::string::npos) {
          ImGui::TextColored(kColorComment, "%s", line.c_str());
        } else {
          ImGui::Text("%s", line.c_str());
        }
        i = next_i + 1;
      }
      break;
    }
    case ShaderDisplayType::kHostDisasm: {
      DrawMultilineString(translation->host_disassembly());
      break;
    }
  }
}

// glBlendEquationSeparatei(i, blend_op, blend_op_alpha);
// glBlendFuncSeparatei(i, src_blend, dest_blend, src_blend_alpha,
//  dest_blend_alpha);
void TraceViewer::DrawBlendMode(uint32_t src_blend, uint32_t dest_blend, uint32_t blend_op) {
  static const char* kBlendNames[] = {
      /*  0 */ "ZERO",
      /*  1 */ "ONE",
      /*  2 */ "UNK2",  // ?
      /*  3 */ "UNK3",  // ?
      /*  4 */ "SRC_COLOR",
      /*  5 */ "ONE_MINUS_SRC_COLOR",
      /*  6 */ "SRC_ALPHA",
      /*  7 */ "ONE_MINUS_SRC_ALPHA",
      /*  8 */ "DST_COLOR",
      /*  9 */ "ONE_MINUS_DST_COLOR",
      /* 10 */ "DST_ALPHA",
      /* 11 */ "ONE_MINUS_DST_ALPHA",
      /* 12 */ "CONSTANT_COLOR",
      /* 13 */ "ONE_MINUS_CONSTANT_COLOR",
      /* 14 */ "CONSTANT_ALPHA",
      /* 15 */ "ONE_MINUS_CONSTANT_ALPHA",
      /* 16 */ "SRC_ALPHA_SATURATE",
  };
  const char* src_str = kBlendNames[src_blend];
  const char* dest_str = kBlendNames[dest_blend];
  const char* op_template;
  switch (blend_op) {
    case 0:  // add
      op_template = "%s + %s";
      break;
    case 1:  // subtract
      op_template = "%s - %s";
      break;
    case 2:  // min
      op_template = "min(%s, %s)";
      break;
    case 3:  // max
      op_template = "max(%s, %s)";
      break;
    case 4:  // reverse subtract
      op_template = "-(%s) + %s";
      break;
    default:
      op_template = "%s ? %s";
      break;
  }
  ImGui::Text(op_template, src_str, dest_str);
}

void TraceViewer::DrawTextureInfo(const Shader::TextureBinding& texture_binding) {
  auto& regs = *graphics_system_->register_file();

  xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(texture_binding.fetch_constant);
  if (fetch.type != xenos::FetchConstantType::kTexture &&
      (!REXCVAR_GET(gpu_allow_invalid_fetch_constants) ||
       fetch.type != xenos::FetchConstantType::kInvalidTexture)) {
    DrawFailedTextureInfo(texture_binding, "Invalid fetch type");
    return;
  }
  TextureInfo texture_info;
  if (!TextureInfo::Prepare(fetch, &texture_info)) {
    DrawFailedTextureInfo(texture_binding, "Unable to parse texture fetcher info");
    return;
  }
  SamplerInfo sampler_info;
  if (!SamplerInfo::Prepare(fetch, texture_binding.fetch_instr, &sampler_info)) {
    DrawFailedTextureInfo(texture_binding, "Unable to parse sampler info");
    return;
  }
  auto texture = GetTextureEntry(texture_info, sampler_info);

  ImGui::Columns(2);
  if (texture) {
    ImVec2 button_size(256, 256);
    if (ImGui::ImageButton("#texture_info_image", ImTextureID(texture), button_size, ImVec2(0, 0),
                           ImVec2(1, 1))) {
      // show viewer
    }
  } else {
    DrawFailedTextureInfo(texture_binding, "Failed to demand texture");
  }
  ImGui::NextColumn();
  ImGui::Text("Fetch Slot: %u", texture_binding.fetch_constant);
  ImGui::Text("Guest Address: %.8X", texture_info.memory.base_address);
  ImGui::Text("Format: %s", texture_info.format_info()->name);
  switch (texture_info.dimension) {
    case xenos::DataDimension::k1D:
      ImGui::Text("1D: %dpx", texture_info.width + 1);
      break;
    case xenos::DataDimension::k2DOrStacked:
      ImGui::Text("2D: %dx%dpx", texture_info.width + 1, texture_info.height + 1);
      break;
    case xenos::DataDimension::k3D:
      ImGui::Text("3D: %dx%dx%dpx", texture_info.width + 1, texture_info.height + 1,
                  texture_info.depth + 1);
      break;
    case xenos::DataDimension::kCube:
      ImGui::Text("Cube: ?");
      break;
  }
  static const char* kSwizzleMap[] = {"R", "G", "B", "A", "0", "1"};
  ImGui::Text("Swizzle: %s%s%s%s", kSwizzleMap[(fetch.swizzle >> 0) & 0x7],
              kSwizzleMap[(fetch.swizzle >> 3) & 0x7], kSwizzleMap[(fetch.swizzle >> 6) & 0x7],
              kSwizzleMap[(fetch.swizzle >> 9) & 0x7]);

  ImGui::Columns(1);
}

void TraceViewer::DrawFailedTextureInfo(const Shader::TextureBinding& texture_binding,
                                        const char* message) {
  // TODO(benvanik): better error info/etc.
  ImGui::TextColored(kColorError, "ERROR: %s", message);
}

void TraceViewer::DrawVertexFetcher(Shader* shader, const Shader::VertexBinding& vertex_binding,
                                    const xe_gpu_vertex_fetch_t& fetch) {
  const uint8_t* addr = memory_->TranslatePhysical(fetch.address << 2);
  uint32_t vertex_count = fetch.size / vertex_binding.stride_words;
  int column_count = 0;
  for (const auto& attrib : vertex_binding.attributes) {
    switch (attrib.fetch_instr.attributes.data_format) {
      case xenos::VertexFormat::k_32:
      case xenos::VertexFormat::k_32_FLOAT:
        ++column_count;
        break;
      case xenos::VertexFormat::k_16_16:
      case xenos::VertexFormat::k_16_16_FLOAT:
      case xenos::VertexFormat::k_32_32:
      case xenos::VertexFormat::k_32_32_FLOAT:
        column_count += 2;
        break;
      case xenos::VertexFormat::k_10_11_11:
      case xenos::VertexFormat::k_11_11_10:
      case xenos::VertexFormat::k_32_32_32_FLOAT:
        column_count += 3;
        break;
      case xenos::VertexFormat::k_8_8_8_8:
        ++column_count;
        break;
      case xenos::VertexFormat::k_2_10_10_10:
      case xenos::VertexFormat::k_16_16_16_16:
      case xenos::VertexFormat::k_32_32_32_32:
      case xenos::VertexFormat::k_16_16_16_16_FLOAT:
      case xenos::VertexFormat::k_32_32_32_32_FLOAT:
        column_count += 4;
        break;
      case xenos::VertexFormat::kUndefined:
        assert_unhandled_case(attrib.fetch_instr.attributes.data_format);
        break;
    }
  }
  ImGui::BeginChild("#indices", ImVec2(0, 300));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 0));
  int display_start = 0;
  int display_end = vertex_count;
  ImGui::Dummy(ImVec2(0, (display_start)*ImGui::GetTextLineHeight()));
  ImGui::Columns(column_count);
  if (display_start <= 1) {
    for (size_t el_index = 0; el_index < vertex_binding.attributes.size(); ++el_index) {
      const auto& attrib = vertex_binding.attributes[el_index];
      switch (attrib.fetch_instr.attributes.data_format) {
        case xenos::VertexFormat::k_32:
        case xenos::VertexFormat::k_32_FLOAT:
          ImGui::Text("e%" PRId64 ".x", el_index);
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_16_16:
        case xenos::VertexFormat::k_16_16_FLOAT:
        case xenos::VertexFormat::k_32_32:
        case xenos::VertexFormat::k_32_32_FLOAT:
          ImGui::Text("e%" PRId64 ".x", el_index);
          ImGui::NextColumn();
          ImGui::Text("e%" PRId64 ".y", el_index);
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_10_11_11:
        case xenos::VertexFormat::k_11_11_10:
        case xenos::VertexFormat::k_32_32_32_FLOAT:
          ImGui::Text("e%" PRId64 ".x", el_index);
          ImGui::NextColumn();
          ImGui::Text("e%" PRId64 ".y", el_index);
          ImGui::NextColumn();
          ImGui::Text("e%" PRId64 ".z", el_index);
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_8_8_8_8:
          ImGui::Text("e%" PRId64 ".xyzw", el_index);
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_2_10_10_10:
        case xenos::VertexFormat::k_16_16_16_16:
        case xenos::VertexFormat::k_32_32_32_32:
        case xenos::VertexFormat::k_16_16_16_16_FLOAT:
        case xenos::VertexFormat::k_32_32_32_32_FLOAT:
          ImGui::Text("e%" PRId64 ".x", el_index);
          ImGui::NextColumn();
          ImGui::Text("e%" PRId64 ".y", el_index);
          ImGui::NextColumn();
          ImGui::Text("e%" PRId64 ".z", el_index);
          ImGui::NextColumn();
          ImGui::Text("e%" PRId64 ".w", el_index);
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::kUndefined:
          assert_unhandled_case(attrib.fetch_instr.attributes.data_format);
          break;
      }
    }
    ImGui::Separator();
  }
  for (int i = display_start; i < display_end; ++i) {
    const uint8_t* vstart = addr + i * vertex_binding.stride_words * 4;
    for (const auto& attrib : vertex_binding.attributes) {
#define LOADEL(type, wo)                                                                \
  GpuSwap(memory::load<type>(vstart + (attrib.fetch_instr.attributes.offset + wo) * 4), \
          fetch.endian)
      switch (attrib.fetch_instr.attributes.data_format) {
        case xenos::VertexFormat::k_32:
          ImGui::Text("%.8X", LOADEL(uint32_t, 0));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_32_FLOAT:
          ImGui::Text("%.3f", LOADEL(float, 0));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_16_16: {
          auto e0 = LOADEL(uint32_t, 0);
          ImGui::Text("%.4X", (e0 >> 16) & 0xFFFF);
          ImGui::NextColumn();
          ImGui::Text("%.4X", (e0 >> 0) & 0xFFFF);
          ImGui::NextColumn();
        } break;
        case xenos::VertexFormat::k_16_16_FLOAT: {
          auto e0 = LOADEL(uint32_t, 0);
          ImGui::Text("%.2f", Float16ToFloat((e0 >> 16) & 0xFFFF));
          ImGui::NextColumn();
          ImGui::Text("%.2f", Float16ToFloat((e0 >> 0) & 0xFFFF));
          ImGui::NextColumn();
        } break;
        case xenos::VertexFormat::k_32_32:
          ImGui::Text("%.8X", LOADEL(uint32_t, 0));
          ImGui::NextColumn();
          ImGui::Text("%.8X", LOADEL(uint32_t, 1));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_32_32_FLOAT:
          ImGui::Text("%.3f", LOADEL(float, 0));
          ImGui::NextColumn();
          ImGui::Text("%.3f", LOADEL(float, 1));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_10_11_11:
        case xenos::VertexFormat::k_11_11_10:
          ImGui::Text("??");
          ImGui::NextColumn();
          ImGui::Text("??");
          ImGui::NextColumn();
          ImGui::Text("??");
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_32_32_32_FLOAT:
          ImGui::Text("%.3f", LOADEL(float, 0));
          ImGui::NextColumn();
          ImGui::Text("%.3f", LOADEL(float, 1));
          ImGui::NextColumn();
          ImGui::Text("%.3f", LOADEL(float, 2));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_8_8_8_8:
          ImGui::Text("%.8X", LOADEL(uint32_t, 0));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_2_10_10_10: {
          auto e0 = LOADEL(uint32_t, 0);
          ImGui::Text("??");
          ImGui::NextColumn();
          ImGui::Text("??");
          ImGui::NextColumn();
          ImGui::Text("??");
          ImGui::NextColumn();
          ImGui::Text("??");
          ImGui::NextColumn();
        } break;
        case xenos::VertexFormat::k_16_16_16_16: {
          auto e0 = LOADEL(uint32_t, 0);
          auto e1 = LOADEL(uint32_t, 1);
          ImGui::Text("%.4X", (e0 >> 16) & 0xFFFF);
          ImGui::NextColumn();
          ImGui::Text("%.4X", (e0 >> 0) & 0xFFFF);
          ImGui::NextColumn();
          ImGui::Text("%.4X", (e1 >> 16) & 0xFFFF);
          ImGui::NextColumn();
          ImGui::Text("%.4X", (e1 >> 0) & 0xFFFF);
          ImGui::NextColumn();
        } break;
        case xenos::VertexFormat::k_32_32_32_32:
          ImGui::Text("%.8X", LOADEL(uint32_t, 0));
          ImGui::NextColumn();
          ImGui::Text("%.8X", LOADEL(uint32_t, 1));
          ImGui::NextColumn();
          ImGui::Text("%.8X", LOADEL(uint32_t, 2));
          ImGui::NextColumn();
          ImGui::Text("%.8X", LOADEL(uint32_t, 3));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::k_16_16_16_16_FLOAT: {
          auto e0 = LOADEL(uint32_t, 0);
          auto e1 = LOADEL(uint32_t, 1);
          ImGui::Text("%.2f", Float16ToFloat((e0 >> 16) & 0xFFFF));
          ImGui::NextColumn();
          ImGui::Text("%.2f", Float16ToFloat((e0 >> 0) & 0xFFFF));
          ImGui::NextColumn();
          ImGui::Text("%.2f", Float16ToFloat((e1 >> 16) & 0xFFFF));
          ImGui::NextColumn();
          ImGui::Text("%.2f", Float16ToFloat((e1 >> 0) & 0xFFFF));
          ImGui::NextColumn();
        } break;
        case xenos::VertexFormat::k_32_32_32_32_FLOAT:
          ImGui::Text("%.3f", LOADEL(float, 0));
          ImGui::NextColumn();
          ImGui::Text("%.3f", LOADEL(float, 1));
          ImGui::NextColumn();
          ImGui::Text("%.3f", LOADEL(float, 2));
          ImGui::NextColumn();
          ImGui::Text("%.3f", LOADEL(float, 3));
          ImGui::NextColumn();
          break;
        case xenos::VertexFormat::kUndefined:
          assert_unhandled_case(attrib.fetch_instr.attributes.data_format);
          break;
      }
    }
  }
  ImGui::Columns(1);
  ImGui::Dummy(ImVec2(0, (vertex_count - display_end) * ImGui::GetTextLineHeight()));
  ImGui::PopStyleVar();
  ImGui::EndChild();
}

static const char* kCompareFuncNames[] = {
    "<false>", "<", "==", "<=", ">", "!=", ">=", "<true>",
};
static const char* kStencilFuncNames[] = {
    "Keep",
    "Zero",
    "Replace",
    "Increment and Wrap",
    "Decrement and Wrap",
    "Invert",
    "Increment and Clamp",
    "Decrement and Clamp",
};
static const char* kIndexFormatNames[] = {
    "uint16",
    "uint32",
};
static const char* kEndiannessNames[] = {
    "unspecified endianness",
    "8-in-16",
    "8-in-32",
    "16-in-32",
};
static const char* kColorFormatNames[] = {
    /* 0  */ "k_8_8_8_8",
    /* 1  */ "k_8_8_8_8_GAMMA",
    /* 2  */ "k_2_10_10_10",
    /* 3  */ "k_2_10_10_10_FLOAT",
    /* 4  */ "k_16_16",
    /* 5  */ "k_16_16_16_16",
    /* 6  */ "k_16_16_FLOAT",
    /* 7  */ "k_16_16_16_16_FLOAT",
    /* 8  */ "unknown(8)",
    /* 9  */ "unknown(9)",
    /* 10 */ "k_2_10_10_10_AS_10_10_10_10",
    /* 11 */ "unknown(11)",
    /* 12 */ "k_2_10_10_10_FLOAT_AS_16_16_16_16",
    /* 13 */ "unknown(13)",
    /* 14 */ "k_32_FLOAT",
    /* 15 */ "k_32_32_FLOAT",
};
static const char* kDepthFormatNames[] = {
    "kD24S8",
    "kD24FS8",
};

void ProgressBar(float frac, float width, float height = 0,
                 const ImVec4& color = ImVec4(0, 1, 0, 1),
                 const ImVec4& border_color = ImVec4(0, 1, 0, 1)) {
  if (height == 0) {
    height = ImGui::GetTextLineHeightWithSpacing();
  }
  frac = rex::saturate(frac);

  const auto fontAtlas = ImGui::GetIO().Fonts;

  auto pos = ImGui::GetCursorScreenPos();
  auto col = ImGui::ColorConvertFloat4ToU32(color);
  auto border_col = ImGui::ColorConvertFloat4ToU32(border_color);

  if (frac > 0) {
    // Progress bar
    ImGui::GetWindowDrawList()->AddRectFilled(pos, ImVec2(pos.x + width * frac, pos.y + height),
                                              col);
  }
  if (border_color.w > 0.f) {
    // Border
    ImGui::GetWindowDrawList()->AddRect(pos, ImVec2(pos.x + width, pos.y + height), border_col);
  }

  ImGui::Dummy(ImVec2(width, height));
}

void ZoomedImage(ImTextureID tex, ImVec2 rel_pos, ImVec2 tex_size, float focus_size,
                 ImVec2 image_size = ImVec2(128, 128)) {
  ImVec2 focus;
  focus.x = rel_pos.x - (focus_size * 0.5f);
  focus.y = rel_pos.y - (focus_size * 0.5f);

  ImVec2 uv0 = ImVec2(focus.x / tex_size.x, focus.y / tex_size.y);
  ImVec2 uv1 = ImVec2((focus.x + focus_size) / tex_size.x, (focus.y + focus_size) / tex_size.y);
  ImGui::Image(tex, image_size, uv0, uv1);
}

void TraceViewer::DrawStateUI() {
  auto command_processor = graphics_system_->command_processor();
  auto& regs = *graphics_system_->register_file();

  ImGui::SetNextWindowPos(ImVec2(float(window_->GetActualLogicalWidth()) - 500 - 5, 30),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(500, 680));
  ImGui::SetNextWindowBgAlpha(kWindowBgAlpha);
  if (!ImGui::Begin("State", nullptr)) {
    ImGui::End();
    return;
  }

  if (!player_->current_frame() || player_->current_command_index() == -1) {
    ImGui::Text("No frame/command selected");
    ImGui::End();
    return;
  }

  auto frame = player_->current_frame();
  const auto& command = frame->commands[player_->current_command_index()];
  auto packet_head = command.head_ptr + sizeof(PacketStartCommand);
  uint32_t packet = memory::load_and_swap<uint32_t>(packet_head);
  uint32_t packet_type = packet >> 30;
  assert_true(packet_type == 0x03);
  uint32_t opcode = (packet >> 8) & 0x7F;
  struct {
    xenos::PrimitiveType prim_type;
    bool is_auto_index;
    uint32_t index_count;
    uint32_t index_buffer_ptr;
    uint32_t index_buffer_size;
    xenos::Endian index_endianness;
    xenos::IndexFormat index_format;
  } draw_info;
  std::memset(&draw_info, 0, sizeof(draw_info));
  switch (opcode) {
    case PM4_DRAW_INDX: {
      uint32_t dword0 = memory::load_and_swap<uint32_t>(packet_head + 4);
      uint32_t dword1 = memory::load_and_swap<uint32_t>(packet_head + 8);
      draw_info.index_count = dword1 >> 16;
      draw_info.prim_type = static_cast<xenos::PrimitiveType>(dword1 & 0x3F);
      uint32_t src_sel = (dword1 >> 6) & 0x3;
      if (src_sel == 0x0) {
        // Indexed draw.
        draw_info.is_auto_index = false;
        draw_info.index_buffer_ptr = memory::load_and_swap<uint32_t>(packet_head + 12);
        uint32_t index_size = memory::load_and_swap<uint32_t>(packet_head + 16);
        draw_info.index_endianness = static_cast<xenos::Endian>(index_size >> 30);
        index_size &= 0x00FFFFFF;
        bool index_32bit = (dword1 >> 11) & 0x1;
        draw_info.index_format =
            index_32bit ? xenos::IndexFormat::kInt32 : xenos::IndexFormat::kInt16;
        draw_info.index_buffer_size = index_size * (index_32bit ? 4 : 2);
      } else if (src_sel == 0x2) {
        // Auto draw.
        draw_info.is_auto_index = true;
      } else {
        // Unknown source select.
        assert_always();
      }
      break;
    }
    case PM4_DRAW_INDX_2: {
      uint32_t dword0 = memory::load_and_swap<uint32_t>(packet_head + 4);
      uint32_t src_sel = (dword0 >> 6) & 0x3;
      assert_true(src_sel == 0x2);  // 'SrcSel=AutoIndex'
      draw_info.prim_type = static_cast<xenos::PrimitiveType>(dword0 & 0x3F);
      draw_info.is_auto_index = true;
      draw_info.index_count = dword0 >> 16;
      break;
    }
  }

  if (player_->is_playing_trace()) {
    ImGui::Text("Playing trace...");
    float width = ImGui::GetWindowWidth() - 20.f;

    ProgressBar(float(player_->playback_percent()) / 10000.f, width);
    ImGui::End();
    return;
  }

  auto enable_mode = static_cast<EdramMode>(regs[XE_GPU_REG_RB_MODECONTROL] & 0x7);

  const char* mode_name = "Unknown";
  switch (enable_mode) {
    case EdramMode::kNoOperation:
      ImGui::Text("Ignored Command %d", player_->current_command_index());
      break;
    case EdramMode::kColorDepth:
    case EdramMode::kDepthOnly: {
      static const char* kPrimNames[] = {
          "<none>",         "point list",   "line list",      "line strip",
          "triangle list",  "triangle fan", "triangle strip", "unknown 0x7",
          "rectangle list", "unknown 0x9",  "unknown 0xA",    "unknown 0xB",
          "line loop",      "quad list",    "quad strip",     "unknown 0xF",
      };
      ImGui::Text("%s Command %d: %s, %d indices",
                  enable_mode == EdramMode::kColorDepth ? "Color-Depth" : "Depth-only",
                  player_->current_command_index(), kPrimNames[int(draw_info.prim_type)],
                  draw_info.index_count);
      break;
    }
    case EdramMode::kCopy: {
      uint32_t copy_dest_base = regs[XE_GPU_REG_RB_COPY_DEST_BASE];
      ImGui::Text("Copy Command %d (to %.8X)", player_->current_command_index(), copy_dest_base);
      break;
    }
  }

  ImGui::Columns(2);
  ImGui::BulletText("Viewport State:");
  if (true) {
    ImGui::TreePush((const void*)0);
    uint32_t pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
    if ((pa_su_sc_mode_cntl >> 16) & 1) {
      uint32_t window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
      int16_t window_offset_x = window_offset & 0x7FFF;
      int16_t window_offset_y = (window_offset >> 16) & 0x7FFF;
      if (window_offset_x & 0x4000) {
        window_offset_x |= 0x8000;
      }
      if (window_offset_y & 0x4000) {
        window_offset_y |= 0x8000;
      }
      ImGui::BulletText("Window Offset: %d, %d", window_offset_x, window_offset_y);
    } else {
      ImGui::BulletText("Window Offset: disabled");
    }
    uint32_t window_scissor_tl = regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL];
    uint32_t window_scissor_br = regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR];
    ImGui::BulletText("Window Scissor: %d,%d to %d,%d (%d x %d)", window_scissor_tl & 0x7FFF,
                      (window_scissor_tl >> 16) & 0x7FFF, window_scissor_br & 0x7FFF,
                      (window_scissor_br >> 16) & 0x7FFF,
                      (window_scissor_br & 0x7FFF) - (window_scissor_tl & 0x7FFF),
                      ((window_scissor_br >> 16) & 0x7FFF) - ((window_scissor_tl >> 16) & 0x7FFF));
    uint32_t surface_info = regs[XE_GPU_REG_RB_SURFACE_INFO];
    uint32_t surface_hiz = (surface_info >> 18) & 0x3FFF;
    uint32_t surface_pitch = surface_info & 0x3FFF;
    auto surface_msaa = (surface_info >> 16) & 0x3;
    static const char* kMsaaNames[] = {
        "1X",
        "2X",
        "4X",
    };
    ImGui::BulletText("Surface Pitch: %d", surface_pitch);
    ImGui::BulletText("Surface HI-Z Pitch: %d", surface_hiz);
    ImGui::BulletText("Surface MSAA: %s", kMsaaNames[surface_msaa]);
    uint32_t vte_control = regs[XE_GPU_REG_PA_CL_VTE_CNTL];
    bool vport_xscale_enable = (vte_control & (1 << 0)) > 0;
    bool vport_xoffset_enable = (vte_control & (1 << 1)) > 0;
    bool vport_yscale_enable = (vte_control & (1 << 2)) > 0;
    bool vport_yoffset_enable = (vte_control & (1 << 3)) > 0;
    bool vport_zscale_enable = (vte_control & (1 << 4)) > 0;
    bool vport_zoffset_enable = (vte_control & (1 << 5)) > 0;
    assert_true(vport_xscale_enable == vport_yscale_enable == vport_zscale_enable ==
                vport_xoffset_enable == vport_yoffset_enable == vport_zoffset_enable);
    if (!vport_xscale_enable) {
      ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
    }
    ImGui::BulletText(
        "Viewport Offset: %f, %f, %f",
        vport_xoffset_enable ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XOFFSET) : 0.0f,
        vport_yoffset_enable ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET) : 0.0f,
        vport_zoffset_enable ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_ZOFFSET) : 0.0f);
    ImGui::BulletText("Viewport Scale: %f, %f, %f",
                      vport_xscale_enable ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XSCALE) : 1.0f,
                      vport_yscale_enable ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE) : 1.0f,
                      vport_zscale_enable ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_ZSCALE) : 1.0f);
    if (!vport_xscale_enable) {
      ImGui::PopStyleColor();
    }

    ImGui::BulletText("Vertex Format: %s, %s, %s, %s", ((vte_control >> 8) & 0x1) ? "x/w0" : "x",
                      ((vte_control >> 8) & 0x1) ? "y/w0" : "y",
                      ((vte_control >> 9) & 0x1) ? "z/w0" : "z",
                      ((vte_control >> 10) & 0x1) ? "w0" : "1/w0");
    uint32_t clip_control = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
    bool clip_enabled = ((clip_control >> 17) & 0x1) == 0;
    bool dx_clip = ((clip_control >> 20) & 0x1) == 0x1;
    ImGui::BulletText("Clip Enabled: %s, DX Clip: %s", clip_enabled ? "true" : "false",
                      dx_clip ? "true" : "false");
    ImGui::TreePop();
  }
  ImGui::NextColumn();
  ImGui::BulletText("Rasterizer State:");
  if (true) {
    ImGui::TreePush((const void*)0);
    uint32_t pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
    uint32_t pa_sc_screen_scissor_tl = regs[XE_GPU_REG_PA_SC_SCREEN_SCISSOR_TL];
    uint32_t pa_sc_screen_scissor_br = regs[XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR];
    if (pa_sc_screen_scissor_tl != 0 && pa_sc_screen_scissor_br != 0x20002000) {
      int32_t screen_scissor_x = pa_sc_screen_scissor_tl & 0x7FFF;
      int32_t screen_scissor_y = (pa_sc_screen_scissor_tl >> 16) & 0x7FFF;
      int32_t screen_scissor_w = (pa_sc_screen_scissor_br & 0x7FFF) - screen_scissor_x;
      int32_t screen_scissor_h = ((pa_sc_screen_scissor_br >> 16) & 0x7FFF) - screen_scissor_y;
      ImGui::BulletText("Scissor: %d,%d to %d,%d (%d x %d)", screen_scissor_x, screen_scissor_y,
                        screen_scissor_x + screen_scissor_w, screen_scissor_y + screen_scissor_h,
                        screen_scissor_w, screen_scissor_h);
    } else {
      ImGui::BulletText("Scissor: disabled");
    }
    switch (pa_su_sc_mode_cntl & 0x3) {
      case 0:
        ImGui::BulletText("Culling: disabled");
        break;
      case 1:
        ImGui::BulletText("Culling: front-face");
        break;
      case 2:
        ImGui::BulletText("Culling: back-face");
        break;
    }
    if (pa_su_sc_mode_cntl & 0x4) {
      ImGui::BulletText("Front-face: clockwise");
    } else {
      ImGui::BulletText("Front-face: counter-clockwise");
    }
    static const char* kFillModeNames[3] = {
        "point",
        "line",
        "fill",
    };
    bool poly_mode = ((pa_su_sc_mode_cntl >> 3) & 0x3) != 0;
    if (poly_mode) {
      uint32_t front_poly_mode = (pa_su_sc_mode_cntl >> 5) & 0x7;
      uint32_t back_poly_mode = (pa_su_sc_mode_cntl >> 8) & 0x7;
      // GL only supports both matching.
      assert_true(front_poly_mode == back_poly_mode);
      ImGui::BulletText("Polygon Mode: %s", kFillModeNames[front_poly_mode]);
    } else {
      ImGui::BulletText("Polygon Mode: fill");
    }
    if (pa_su_sc_mode_cntl & (1 << 19)) {
      ImGui::BulletText("Provoking Vertex: last");
    } else {
      ImGui::BulletText("Provoking Vertex: first");
    }
    ImGui::TreePop();
  }
  ImGui::Columns(1);

  auto rb_surface_info = regs[XE_GPU_REG_RB_SURFACE_INFO];
  uint32_t surface_pitch = rb_surface_info & 0x3FFF;
  auto surface_msaa = static_cast<xenos::MsaaSamples>((rb_surface_info >> 16) & 0x3);

  if (ImGui::CollapsingHeader("Color Targets")) {
    if (enable_mode != EdramMode::kDepthOnly) {
      // Alpha testing -- ALPHAREF, ALPHAFUNC, ALPHATESTENABLE
      // if(ALPHATESTENABLE && frag_out.a [<=/ALPHAFUNC] ALPHAREF) discard;
      uint32_t color_control = regs[XE_GPU_REG_RB_COLORCONTROL];
      if ((color_control & 0x8) != 0) {
        ImGui::BulletText("Alpha Test: %s %.2f", kCompareFuncNames[color_control & 0x7],
                          regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF));
      } else {
        ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
        ImGui::BulletText("Alpha Test: disabled");
        ImGui::PopStyleColor();
      }

      auto blend_color = ImVec4(
          regs.Get<float>(XE_GPU_REG_RB_BLEND_RED), regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
          regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE), regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA));
      ImGui::BulletText("Blend Color: (%.2f,%.2f,%.2f,%.2f)", blend_color.x, blend_color.y,
                        blend_color.z, blend_color.w);
      ImGui::SameLine();
      // TODO small_height (was true) parameter was removed
      ImGui::ColorButton(nullptr, blend_color);

      uint32_t rb_color_mask = regs[XE_GPU_REG_RB_COLOR_MASK];
      uint32_t color_info[4] = {
          regs[XE_GPU_REG_RB_COLOR_INFO],
          regs[XE_GPU_REG_RB_COLOR1_INFO],
          regs[XE_GPU_REG_RB_COLOR2_INFO],
          regs[XE_GPU_REG_RB_COLOR3_INFO],
      };
      uint32_t rb_blendcontrol[4] = {
          regs[XE_GPU_REG_RB_BLENDCONTROL0],
          regs[XE_GPU_REG_RB_BLENDCONTROL1],
          regs[XE_GPU_REG_RB_BLENDCONTROL2],
          regs[XE_GPU_REG_RB_BLENDCONTROL3],
      };
      ImGui::Columns(2);
      for (int i = 0; i < rex::countof(color_info); ++i) {
        uint32_t blend_control = rb_blendcontrol[i];
        // A2XX_RB_BLEND_CONTROL_COLOR_SRCBLEND
        auto src_blend = (blend_control & 0x0000001F) >> 0;
        // A2XX_RB_BLEND_CONTROL_COLOR_DESTBLEND
        auto dest_blend = (blend_control & 0x00001F00) >> 8;
        // A2XX_RB_BLEND_CONTROL_COLOR_COMB_FCN
        auto blend_op = (blend_control & 0x000000E0) >> 5;
        // A2XX_RB_BLEND_CONTROL_ALPHA_SRCBLEND
        auto src_blend_alpha = (blend_control & 0x001F0000) >> 16;
        // A2XX_RB_BLEND_CONTROL_ALPHA_DESTBLEND
        auto dest_blend_alpha = (blend_control & 0x1F000000) >> 24;
        // A2XX_RB_BLEND_CONTROL_ALPHA_COMB_FCN
        auto blend_op_alpha = (blend_control & 0x00E00000) >> 21;
        // A2XX_RB_COLORCONTROL_BLEND_DISABLE ?? Can't find this!
        // Just guess based on actions.
        bool blend_enable =
            !((src_blend == 1) && (dest_blend == 0) && (blend_op == 0) && (src_blend_alpha == 1) &&
              (dest_blend_alpha == 0) && (blend_op_alpha == 0));
        if (blend_enable) {
          if (src_blend == src_blend_alpha && dest_blend == dest_blend_alpha &&
              blend_op == blend_op_alpha) {
            ImGui::BulletText("Blend %d: ", i);
            ImGui::SameLine();
            DrawBlendMode(src_blend, dest_blend, blend_op);
          } else {
            ImGui::BulletText("Blend %d:", i);
            ImGui::BulletText("  Color: ");
            ImGui::SameLine();
            DrawBlendMode(src_blend, dest_blend, blend_op);
            ImGui::BulletText("  Alpha: ");
            ImGui::SameLine();
            DrawBlendMode(src_blend_alpha, dest_blend_alpha, blend_op_alpha);
          }
        } else {
          ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
          ImGui::BulletText("Blend %d: disabled", i);
          ImGui::PopStyleColor();
        }
        ImGui::NextColumn();
        uint32_t write_mask = (rb_color_mask >> (i * 4)) & 0xF;
        if (write_mask) {
          ImGui::BulletText(
              "Write Mask %d: %s, %s, %s, %s", i, !!(write_mask & 0x1) ? "true" : "false",
              !!(write_mask & 0x2) ? "true" : "false", !!(write_mask & 0x4) ? "true" : "false",
              !!(write_mask & 0x8) ? "true" : "false");
        } else {
          ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
          ImGui::BulletText("Write Mask %d: disabled", i);
          ImGui::PopStyleColor();
        }
        ImGui::NextColumn();
      }
      ImGui::Columns(1);

      ImGui::Columns(4);
      for (int i = 0; i < rex::countof(color_info); ++i) {
        uint32_t write_mask = (rb_color_mask >> (i * 4)) & 0xF;
        uint32_t color_base = color_info[i] & 0xFFF;
        auto color_format =
            static_cast<xenos::ColorRenderTargetFormat>((color_info[i] >> 16) & 0xF);
        ImVec2 button_pos = ImGui::GetCursorScreenPos();
        ImVec2 button_size(256, 256);
        ImTextureID tex = 0;
        ImGui::PushID(i);
        if (write_mask) {
          auto color_target =
              GetColorRenderTarget(surface_pitch, surface_msaa, color_base, color_format);
          tex = ImTextureID(color_target);
          if (ImGui::ImageButton("#color_image", tex, button_size, ImVec2(0, 0), ImVec2(1, 1))) {
            // show viewer
          }
        } else {
          ImGui::ImageButton("#color_image", ImTextureID(0), button_size, ImVec2(0, 0),
                             ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(0, 0, 0, 0));
        }
        ImGui::PopID();
        if (ImGui::IsItemHovered()) {
          ImGui::BeginTooltip();
          ImGui::Text("Color Target %d (%s), base %.4X, pitch %d, format %s", i,
                      write_mask ? "enabled" : "disabled", color_base, surface_pitch,
                      kColorFormatNames[uint32_t(color_format)]);

          if (tex) {
            ImVec2 rel_pos;
            rel_pos.x = ImGui::GetMousePos().x - button_pos.x;
            rel_pos.y = ImGui::GetMousePos().y - button_pos.y;
            ZoomedImage(tex, rel_pos, button_size, 32.f, ImVec2(256, 256));
          }

          ImGui::EndTooltip();
        }
        ImGui::NextColumn();
      }
      ImGui::Columns(1);
    } else {
      ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
      ImGui::Text("Depth-only mode, no color targets");
      ImGui::PopStyleColor();
    }
  }

  if (ImGui::CollapsingHeader("Depth/Stencil Target")) {
    auto rb_depthcontrol = regs[XE_GPU_REG_RB_DEPTHCONTROL];
    auto rb_stencilrefmask = regs[XE_GPU_REG_RB_STENCILREFMASK];
    auto rb_depth_info = regs[XE_GPU_REG_RB_DEPTH_INFO];
    bool uses_depth = (rb_depthcontrol & 0x00000002) || (rb_depthcontrol & 0x00000004);
    uint32_t stencil_ref = (rb_stencilrefmask & 0xFF);
    uint32_t stencil_read_mask = (rb_stencilrefmask >> 8) & 0xFF;
    uint32_t stencil_write_mask = (rb_stencilrefmask >> 16) & 0xFF;
    bool uses_stencil = (rb_depthcontrol & 0x00000001) || (stencil_write_mask != 0);

    ImGui::Columns(2);

    if (rb_depthcontrol & 0x00000002) {
      ImGui::BulletText("Depth Test: enabled");
    } else {
      ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
      ImGui::BulletText("Depth Test: disabled");
    }
    ImGui::BulletText("Depth Func: %s", kCompareFuncNames[(rb_depthcontrol & 0x00000070) >> 4]);
    if (!(rb_depthcontrol & 0x00000002)) {
      ImGui::PopStyleColor();
    }
    if (rb_depthcontrol & 0x00000004) {
      ImGui::BulletText("Depth Write: enabled");
    } else {
      ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
      ImGui::BulletText("Depth Write: disabled");
      ImGui::PopStyleColor();
    }

    if (rb_depthcontrol & 0x00000001) {
      ImGui::BulletText("Stencil Test: enabled");
      ImGui::BulletText("Stencil ref: 0x%.2X", stencil_ref);
      ImGui::BulletText("Stencil read / write masks: 0x%.2X / 0x%.2X", stencil_read_mask,
                        stencil_write_mask);
      ImGui::BulletText("Front State:");
      ImGui::Indent();
      ImGui::BulletText("Compare Op: %s", kCompareFuncNames[(rb_depthcontrol >> 8) & 0x7]);
      ImGui::BulletText("Fail Op: %s", kStencilFuncNames[(rb_depthcontrol >> 11) & 0x7]);
      ImGui::BulletText("Pass Op: %s", kStencilFuncNames[(rb_depthcontrol >> 14) & 0x7]);
      ImGui::BulletText("Depth Fail Op: %s", kStencilFuncNames[(rb_depthcontrol >> 17) & 0x7]);
      ImGui::Unindent();

      // BACKFACE_ENABLE
      if (!(rb_depthcontrol & 0x80)) {
        ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
        ImGui::BulletText("Back State (same as front)");
        ImGui::PopStyleColor();
      } else {
        ImGui::BulletText("Back State:");
        ImGui::Indent();
        ImGui::BulletText("Compare Op: %s", kCompareFuncNames[(rb_depthcontrol >> 20) & 0x7]);
        ImGui::BulletText("Fail Op: %s", kStencilFuncNames[(rb_depthcontrol >> 23) & 0x7]);
        ImGui::BulletText("Pass Op: %s", kStencilFuncNames[(rb_depthcontrol >> 26) & 0x7]);
        ImGui::BulletText("Depth Fail Op: %s", kStencilFuncNames[(rb_depthcontrol >> 29) & 0x7]);
        ImGui::Unindent();
      }
    } else {
      ImGui::PushStyleColor(ImGuiCol_Text, kColorIgnored);
      ImGui::BulletText("Stencil Test: disabled");
      ImGui::PopStyleColor();
    }

    ImGui::NextColumn();

    if (uses_depth || uses_stencil) {
      uint32_t depth_base = rb_depth_info & 0xFFF;
      auto depth_format = static_cast<xenos::DepthRenderTargetFormat>((rb_depth_info >> 16) & 0x1);
      auto depth_target =
          GetDepthRenderTarget(surface_pitch, surface_msaa, depth_base, depth_format);

      auto button_pos = ImGui::GetCursorScreenPos();
      ImVec2 button_size(256, 256);
      ImGui::ImageButton("#depth_stencil_image", ImTextureID(depth_target), button_size,
                         ImVec2(0, 0), ImVec2(1, 1));
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();

        ImGui::Text("Depth Target: base %.4X, pitch %d, format %s", depth_base, surface_pitch,
                    kDepthFormatNames[uint32_t(depth_format)]);

        ImVec2 rel_pos;
        rel_pos.x = ImGui::GetMousePos().x - button_pos.x;
        rel_pos.y = ImGui::GetMousePos().y - button_pos.y;
        ZoomedImage(ImTextureID(depth_target), rel_pos, button_size, 32.f, ImVec2(256, 256));

        ImGui::EndTooltip();
      }
    } else {
      ImGui::Text("No depth target");
    }

    ImGui::Columns(1);
  }

  if (ImGui::CollapsingHeader("Vertex Shader")) {
    ShaderDisplayType shader_display_type = DrawShaderTypeUI();
    ImGui::BeginChild("#vertex_shader_text", ImVec2(0, 400));
    auto shader = command_processor->active_vertex_shader();
    if (shader) {
      DrawShaderUI(shader, shader_display_type);
    } else {
      ImGui::TextColored(kColorError, "ERROR: no vertex shader set");
    }
    ImGui::EndChild();
  }
  if (ImGui::CollapsingHeader("Vertex Shader Output") && QueryVSOutputElementSize()) {
    auto size = QueryVSOutputSize();
    auto el_size = QueryVSOutputElementSize();
    if (size > 0) {
      std::vector<float> vertices;
      vertices.resize(size / 4);
      QueryVSOutput(vertices.data(), size);

      ImGui::Text("%" PRIu64 " output vertices", vertices.size() / 4);
      ImGui::SameLine();
      static bool normalize = false;
      ImGui::Checkbox("Normalize", &normalize);

      ImGui::BeginChild("#vsvertices", ImVec2(0, 300));

      int display_start = 0;
      int display_end = int(vertices.size() / 4);
      ImGui::Dummy(ImVec2(0, (display_start)*ImGui::GetTextLineHeight()));

      ImGui::Columns(int(el_size), "#vsvertices", true);
      for (size_t i = display_start; i < display_end; i++) {
        size_t start_vtx = i * el_size;
        float verts[4] = {vertices[start_vtx], vertices[start_vtx + 1], vertices[start_vtx + 2],
                          vertices[start_vtx + 3]};
        assert_true(el_size <= rex::countof(verts));
        if (normalize) {
          for (int j = 0; j < el_size; j++) {
            verts[j] /= verts[3];
          }
        }

        for (int j = 0; j < el_size; j++) {
          ImGui::Text("%.3f", verts[j]);
          ImGui::NextColumn();
        }
      }
      ImGui::Columns(1);

      ImGui::Dummy(ImVec2(0, ((vertices.size() / 4) - display_end) * ImGui::GetTextLineHeight()));
      ImGui::EndChild();
    } else {
      ImGui::Text("No vertex shader output");
    }
  }
  if (ImGui::CollapsingHeader("Pixel Shader")) {
    ShaderDisplayType shader_display_type = DrawShaderTypeUI();
    ImGui::BeginChild("#pixel_shader_text", ImVec2(0, 400));
    auto shader = command_processor->active_pixel_shader();
    if (shader) {
      DrawShaderUI(shader, shader_display_type);
    } else {
      ImGui::TextColored(kColorError, "ERROR: no pixel shader set");
    }
    ImGui::EndChild();
  }
  if (ImGui::CollapsingHeader("Index Buffer")) {
    if (draw_info.is_auto_index) {
      ImGui::Text("%d indices, auto-indexed", draw_info.index_count);
    } else {
      ImGui::Text("%d indices from buffer %.8X (%db), %s, %s", draw_info.index_count,
                  draw_info.index_buffer_ptr, draw_info.index_buffer_size,
                  kIndexFormatNames[int(draw_info.index_format)],
                  kEndiannessNames[int(draw_info.index_endianness)]);
      uint32_t pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
      if (pa_su_sc_mode_cntl & (1 << 21)) {
        uint32_t reset_index = regs[XE_GPU_REG_VGT_MULTI_PRIM_IB_RESET_INDX];
        if (draw_info.index_format == xenos::IndexFormat::kInt16) {
          ImGui::Text("Reset Index: %.4X", reset_index & 0xFFFF);
        } else {
          ImGui::Text("Reset Index: %.8X", reset_index);
        }
      } else {
        ImGui::Text("Reset Index: disabled");
      }
      ImGui::BeginChild("#indices", ImVec2(0, 300));
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
      int display_start = 0;
      int display_end = 1 + draw_info.index_count;
      ImGui::Dummy(ImVec2(0, (display_start)*ImGui::GetTextLineHeight()));
      ImGui::Columns(2, "#indices", true);
      ImGui::SetColumnOffset(1, 60);
      if (display_start <= 1) {
        ImGui::Text("Ordinal");
        ImGui::NextColumn();
        ImGui::Text(" Value");
        ImGui::NextColumn();
        ImGui::Separator();
      }
      uint32_t element_size = draw_info.index_format == xenos::IndexFormat::kInt32 ? 4 : 2;
      const uint8_t* data_ptr =
          memory_->TranslatePhysical(draw_info.index_buffer_ptr + (display_start * element_size));
      for (int i = display_start; i < display_end; ++i, data_ptr += element_size) {
        if (i < 10) {
          ImGui::Text("     %d", i);
        } else if (i < 100) {
          ImGui::Text("    %d", i);
        } else if (i < 1000) {
          ImGui::Text("   %d", i);
        } else {
          ImGui::Text("  %d", i);
        }
        ImGui::NextColumn();
        uint32_t value =
            element_size == 4
                ? GpuSwap(memory::load<uint32_t>(data_ptr), draw_info.index_endianness)
                : GpuSwap(memory::load<uint16_t>(data_ptr), draw_info.index_endianness);
        ImGui::Text(" %d", value);
        ImGui::NextColumn();
      }
      ImGui::Columns(1);
      ImGui::Dummy(ImVec2(0, (draw_info.index_count - display_end) * ImGui::GetTextLineHeight()));
      ImGui::PopStyleVar();
      ImGui::EndChild();
    }
  }
  if (ImGui::CollapsingHeader("Vertex Buffers")) {
    auto shader = command_processor->active_vertex_shader();
    if (shader) {
      for (const auto& vertex_binding : shader->vertex_bindings()) {
        xe_gpu_vertex_fetch_t fetch = regs.GetVertexFetch(vertex_binding.fetch_constant);
        assert_true(fetch.endian == xenos::Endian::k8in32);
        char tree_root_id[32];
        sprintf(tree_root_id, "#vertices_root_%d", vertex_binding.fetch_constant);
        if (ImGui::TreeNode(tree_root_id, "vf%d: 0x%.8X (%db), %s", vertex_binding.fetch_constant,
                            fetch.address << 2, fetch.size * 4,
                            kEndiannessNames[int(fetch.endian)])) {
          ImGui::BeginChild("#vertices", ImVec2(0, 300));
          DrawVertexFetcher(shader, vertex_binding, fetch);
          ImGui::EndChild();
          ImGui::TreePop();
        }
      }
    } else {
      ImGui::TextColored(kColorError, "ERROR: no vertex shader set");
    }
  }
  if (ImGui::CollapsingHeader("Vertex Textures")) {
    auto shader = command_processor->active_vertex_shader();
    if (shader) {
      const auto& texture_bindings = shader->texture_bindings();
      if (!texture_bindings.empty()) {
        for (const auto& texture_binding : texture_bindings) {
          DrawTextureInfo(texture_binding);
        }
      } else {
        ImGui::Text("No vertex shader samplers");
      }
    } else {
      ImGui::TextColored(kColorError, "ERROR: no vertex shader set");
    }
  }
  if (ImGui::CollapsingHeader("Pixel Textures")) {
    auto shader = command_processor->active_pixel_shader();
    if (shader) {
      const auto& texture_bindings = shader->texture_bindings();
      if (!texture_bindings.empty()) {
        for (const auto& texture_binding : texture_bindings) {
          DrawTextureInfo(texture_binding);
        }
      } else {
        ImGui::Text("No pixel shader samplers");
      }
    } else {
      ImGui::TextColored(kColorError, "ERROR: no pixel shader set");
    }
  }
  if (ImGui::CollapsingHeader("Fetch Constants (raw)")) {
    ImGui::Columns(2);
    for (int i = XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0; i <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5;
         ++i) {
      ImGui::Text("f%02d_%d", (i - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6,
                  (i - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) % 6);
      ImGui::NextColumn();
      ImGui::Text("%.8X", regs[i]);
      ImGui::NextColumn();
    }
    ImGui::Columns(1);
  }
  if (ImGui::CollapsingHeader("ALU Constants")) {
    ImGui::Columns(2);
    for (int i = XE_GPU_REG_SHADER_CONSTANT_000_X; i <= XE_GPU_REG_SHADER_CONSTANT_511_X; i += 4) {
      ImGui::Text("c%d", (i - XE_GPU_REG_SHADER_CONSTANT_000_X) / 4);
      ImGui::NextColumn();
      ImGui::Text("%f, %f, %f, %f", regs.Get<float>(i + 0), regs.Get<float>(i + 1),
                  regs.Get<float>(i + 2), regs.Get<float>(i + 3));
      ImGui::NextColumn();
    }
    ImGui::Columns(1);
  }
  if (ImGui::CollapsingHeader("Bool Constants")) {
    ImGui::Columns(2);
    for (int i = XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031;
         i <= XE_GPU_REG_SHADER_CONSTANT_BOOL_224_255; ++i) {
      ImGui::Text("b%03d-%03d", (i - XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031) * 32,
                  (i - XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031) * 32 + 31);
      ImGui::NextColumn();
      ImGui::Text("%.8X", regs[i]);
      ImGui::NextColumn();
    }
    ImGui::Columns(1);
  }
  if (ImGui::CollapsingHeader("Loop Constants")) {
    ImGui::Columns(2);
    for (int i = XE_GPU_REG_SHADER_CONSTANT_LOOP_00; i <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31; ++i) {
      ImGui::Text("l%d", i - XE_GPU_REG_SHADER_CONSTANT_LOOP_00);
      ImGui::NextColumn();
      ImGui::Text("%.8X", regs[i]);
      ImGui::NextColumn();
    }
    ImGui::Columns(1);
  }
  ImGui::End();
}

}  // namespace rex::graphics
