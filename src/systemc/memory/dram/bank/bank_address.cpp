#include "systemc/memory/dram/bank/bank_address.h"

namespace flexnpu_sim {

BankDecoder::BankDecoder(uint32_t banks, uint32_t line_bytes)
    : banks_(banks ? banks : 1),
      line_bytes_(line_bytes ? line_bytes : 1) {}

BankAddress BankDecoder::decode(uint64_t address) const {
    const uint64_t line = address / line_bytes_;   // global row index
    BankAddress a;
    a.bank = static_cast<uint32_t>(line % banks_); // interleave rows across banks
    a.row  = line / banks_;
    return a;
}

}  // namespace flexnpu_sim
