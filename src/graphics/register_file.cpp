/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <cstring>

#include <rex/graphics/register_file.h>
#include <rex/math.h>

#include <vector>

namespace rex::graphics {

RegisterFile::RegisterFile() {
  std::memset(values, 0, sizeof(values));
}

const RegisterInfo* RegisterFile::GetRegisterInfo(uint32_t index) {
  switch (index) {
#define XE_GPU_REGISTER(index, type, name) \
  case index: {                            \
    static const RegisterInfo reg_info = { \
        RegisterInfo::Type::type,          \
        #name,                             \
    };                                     \
    return &reg_info;                      \
  }
#include <rex/graphics/register_table.inc>
#undef XE_GPU_REGISTER
    default:
      return nullptr;
  }
}

bool RegisterFile::IsKnownRegister(uint32_t index) {
  static const std::vector<bool> known = [] {
    std::vector<bool> table(kRegisterCount);
    for (uint32_t i = 0; i < kRegisterCount; ++i) {
      table[i] = GetRegisterInfo(i) != nullptr;
    }
    return table;
  }();
  return index < kRegisterCount && known[index];
}

}  // namespace rex::graphics
