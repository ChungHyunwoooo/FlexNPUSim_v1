#ifndef FLEXNPU_TRANSPORT_AXI_BUS_AXI_ARBITER_H
#define FLEXNPU_TRANSPORT_AXI_BUS_AXI_ARBITER_H

// Arbiter for AXI interconnect grant decisions (R4).
//
// Pure C++ — the arbiter is invoked from a bus SC_METHOD / SC_THREAD each
// cycle, but the policy itself is testable without SystemC. Five policies
// are supported:
//
//   RoundRobin     — equal-share rotation through requesters.
//   WeightedRR     — per-master weight; credits drain during consecutive
//                    grants, replenish on rotation.
//   FixedPriority  — per-master priority; lower value wins. Starves
//                    lower-priority masters under sustained contention.
//   PriorityRR     — priority tier first, then round-robin within the
//                    highest-priority tier that has requesters.
//   QoSAware       — AXI QOS field (runtime per-transaction) is the first
//                    tiebreaker; round-robin within the same QOS level.
//
// Per-master runtime configuration (qos / weight / priority / default_qos)
// is stored in MasterArbitCfg. Per-transaction QOS is injected via
// set_qos(), which QoSAware reads each grant call.

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace flexnpu_sim::transport::axi {

enum class ArbitPolicy : uint8_t {
    RoundRobin     = 0,
    WeightedRR     = 1,
    FixedPriority  = 2,
    PriorityRR     = 3,
    QoSAware       = 4,
};

struct MasterArbitCfg {
    unsigned qos         = 0;  ///< static QoS value (overwritable at runtime).
    unsigned weight      = 1;  ///< WeightedRR credits per rotation (>=1).
    unsigned priority    = 0;  ///< FixedPriority / PriorityRR tier; lower wins.
    unsigned default_qos = 0;  ///< QoSAware fallback when caller does not set_qos.
};

class Arbiter {
public:
    Arbiter(ArbitPolicy policy,
            const std::vector<MasterArbitCfg>& cfgs)
        : policy_(policy), cfgs_(cfgs), rr_cursor_(0),
          live_qos_(cfgs.size()), credits_(cfgs.size()) {
        if (cfgs_.empty()) {
            throw std::invalid_argument("Arbiter: at least one master required");
        }
        reset_credits();
        for (std::size_t i = 0; i < cfgs_.size(); ++i) {
            live_qos_[i] = cfgs_[i].default_qos;
        }
    }

    /// Updates the live QoS for master @p idx. QoSAware uses this on the
    /// next grant() call; other policies ignore it.
    void set_qos(unsigned idx, unsigned qos) {
        if (idx < live_qos_.size()) live_qos_[idx] = qos;
    }

    unsigned num_masters() const { return static_cast<unsigned>(cfgs_.size()); }
    ArbitPolicy policy() const { return policy_; }

    /// Returns the granted master index, or -1 if no requesters.
    /// @p requesters.size() must equal num_masters().
    int grant(const std::vector<bool>& requesters) {
        if (requesters.size() != cfgs_.size()) {
            throw std::invalid_argument("Arbiter::grant: requesters size mismatch");
        }
        switch (policy_) {
            case ArbitPolicy::RoundRobin:    return grant_round_robin(requesters);
            case ArbitPolicy::WeightedRR:    return grant_weighted_rr(requesters);
            case ArbitPolicy::FixedPriority: return grant_fixed_priority(requesters);
            case ArbitPolicy::PriorityRR:    return grant_priority_rr(requesters);
            case ArbitPolicy::QoSAware:      return grant_qos_aware(requesters);
        }
        return -1;
    }

private:
    void reset_credits() {
        for (std::size_t i = 0; i < cfgs_.size(); ++i) {
            credits_[i] = cfgs_[i].weight == 0 ? 1u : cfgs_[i].weight;
        }
    }

    int grant_round_robin(const std::vector<bool>& req) {
        const unsigned n = static_cast<unsigned>(cfgs_.size());
        for (unsigned i = 0; i < n; ++i) {
            const unsigned idx = (rr_cursor_ + i) % n;
            if (req[idx]) {
                rr_cursor_ = (idx + 1u) % n;
                return static_cast<int>(idx);
            }
        }
        return -1;
    }

    // WeightedRR: each master holds `weight` credits. On grant, one credit
    // is consumed. When the currently-pointed master has credits and is
    // requesting, it keeps winning; otherwise the cursor advances. When
    // all credits drain, replenish to starting weights and resume.
    int grant_weighted_rr(const std::vector<bool>& req) {
        const unsigned n = static_cast<unsigned>(cfgs_.size());
        for (unsigned round = 0; round < 2u; ++round) {
            // Sticky pass: try current cursor first with its remaining credits.
            for (unsigned i = 0; i < n; ++i) {
                const unsigned idx = (rr_cursor_ + i) % n;
                if (req[idx] && credits_[idx] > 0) {
                    --credits_[idx];
                    if (credits_[idx] == 0) {
                        rr_cursor_ = (idx + 1u) % n;
                    }
                    return static_cast<int>(idx);
                }
            }
            // All reachable masters exhausted their credits this epoch — replenish.
            reset_credits();
        }
        return -1;
    }

    int grant_fixed_priority(const std::vector<bool>& req) {
        const unsigned n = static_cast<unsigned>(cfgs_.size());
        int best = -1;
        unsigned best_pri = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (!req[i]) continue;
            const unsigned p = cfgs_[i].priority;
            if (best < 0 || p < best_pri) {
                best = static_cast<int>(i);
                best_pri = p;
            }
        }
        return best;
    }

    int grant_priority_rr(const std::vector<bool>& req) {
        const unsigned n = static_cast<unsigned>(cfgs_.size());
        unsigned best_pri = 0;
        bool have_req = false;
        for (unsigned i = 0; i < n; ++i) {
            if (!req[i]) continue;
            const unsigned p = cfgs_[i].priority;
            if (!have_req || p < best_pri) { best_pri = p; have_req = true; }
        }
        if (!have_req) return -1;
        for (unsigned i = 0; i < n; ++i) {
            const unsigned idx = (rr_cursor_ + i) % n;
            if (req[idx] && cfgs_[idx].priority == best_pri) {
                rr_cursor_ = (idx + 1u) % n;
                return static_cast<int>(idx);
            }
        }
        return -1;
    }

    int grant_qos_aware(const std::vector<bool>& req) {
        const unsigned n = static_cast<unsigned>(cfgs_.size());
        unsigned best_qos = 0;
        bool have_req = false;
        for (unsigned i = 0; i < n; ++i) {
            if (!req[i]) continue;
            const unsigned q = live_qos_[i];
            if (!have_req || q > best_qos) { best_qos = q; have_req = true; }
        }
        if (!have_req) return -1;
        for (unsigned i = 0; i < n; ++i) {
            const unsigned idx = (rr_cursor_ + i) % n;
            if (req[idx] && live_qos_[idx] == best_qos) {
                rr_cursor_ = (idx + 1u) % n;
                return static_cast<int>(idx);
            }
        }
        return -1;
    }

    ArbitPolicy                 policy_;
    std::vector<MasterArbitCfg> cfgs_;
    unsigned                    rr_cursor_;
    std::vector<unsigned>       live_qos_;
    std::vector<unsigned>       credits_;
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_BUS_AXI_ARBITER_H
