#ifndef FLEXNPU_TRANSPORT_AXI_MASTER_AXI_TICKET_POOL_H
#define FLEXNPU_TRANSPORT_AXI_MASTER_AXI_TICKET_POOL_H

// Fixed-capacity slot allocator used by the R2 signal-level master.
// Each outstanding transaction occupies one slot (ticket) until the caller
// explicitly releases it. When the pool is full, acquire() blocks
// cooperatively on sc_core::wait(freed_) until release() frees a slot.

#include <systemc.h>

#include <vector>

namespace flexnpu_sim::transport::axi {

class TicketPool {
public:
    explicit TicketPool(unsigned capacity)
        : used_(capacity == 0 ? 1 : capacity, false),
          in_use_(0),
          capacity_(capacity == 0 ? 1 : capacity) {}

    int acquire() {
        while (in_use_ >= capacity_) {
            sc_core::wait(freed_);
        }
        for (unsigned i = 0; i < capacity_; ++i) {
            if (!used_[i]) {
                used_[i] = true;
                ++in_use_;
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void release(int ticket) {
        if (ticket < 0 || static_cast<unsigned>(ticket) >= capacity_) return;
        if (!used_[ticket]) return;
        used_[ticket] = false;
        --in_use_;
        freed_.notify(sc_core::SC_ZERO_TIME);
    }

    unsigned capacity() const { return capacity_; }
    unsigned in_use()   const { return in_use_;   }
    bool     full()     const { return in_use_ >= capacity_; }

private:
    std::vector<bool> used_;
    unsigned          in_use_;
    unsigned          capacity_;
    sc_core::sc_event freed_;
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_MASTER_AXI_TICKET_POOL_H
