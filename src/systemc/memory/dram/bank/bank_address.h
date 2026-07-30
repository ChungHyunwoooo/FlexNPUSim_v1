/**
 * @file bank_address.h
 * @brief Address -> (bank, row) decode for the `bank` tier.
 *
 * A "line" is a row-buffer's worth of bytes (a DRAM row). Global rows are
 * interleaved across banks so consecutive lines land in different banks.
 */

#pragma once

#include <cstdint>

namespace flexnpu_sim {

struct BankAddress {
    uint32_t bank;
    uint64_t row;
};

class BankDecoder {
public:
    BankDecoder(uint32_t banks, uint32_t line_bytes);
    BankAddress decode(uint64_t address) const;

private:
    uint32_t banks_;
    uint32_t line_bytes_;
};

}  // namespace flexnpu_sim
