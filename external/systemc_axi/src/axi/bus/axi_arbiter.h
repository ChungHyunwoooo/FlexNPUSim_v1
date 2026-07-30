#ifndef AXI_ARBITER_H
#define AXI_ARBITER_H

#include <map>

/**
 * AXI Bus Arbitration Algorithms
 *
 * Supports multiple arbitration strategies:
 * - ROUND_ROBIN: Fair allocation, rotates priority
 * - LEAST_RECENTLY_USED: Prioritizes masters that haven't transacted recently
 * - PRIORITY_BASED: Fixed priority ordering
 */

enum class ArbiterPolicy {
    ROUND_ROBIN,
    LEAST_RECENTLY_USED,
    PRIORITY_BASED
};

/**
 * Base Arbiter Interface
 */
template<typename CFG>
class AXIArbiterBase {
public:
    using id_t = typename CFG::id_t;
    using addr_t = typename CFG::addr_t;

    struct MasterTransactionCount {
        unsigned int read = 0;
        unsigned int write = 0;
    };

    virtual ~AXIArbiterBase() = default;

    /**
     * Arbitrate conflicts between masters accessing the same slave
     * Returns true if master_a should have priority over master_b
     */
    virtual bool arbitrate(unsigned int master_a, unsigned int master_b,
                          const MasterTransactionCount* trans_counter) = 0;

    /**
     * Update arbitration state after a transaction completes
     */
    virtual void update_on_transaction(unsigned int master_id, bool is_read) {}
};

/**
 * Round-Robin Arbiter
 * Cycles through masters fairly
 */
template<typename CFG>
class RoundRobinArbiter : public AXIArbiterBase<CFG> {
public:
    using MasterTransactionCount = typename AXIArbiterBase<CFG>::MasterTransactionCount;

    bool arbitrate(unsigned int master_a, unsigned int master_b,
                  const MasterTransactionCount* trans_counter) override {
        // Use transaction count as a simple round-robin metric
        // Master with fewer transactions gets priority
        unsigned int total_a = trans_counter[master_a].read + trans_counter[master_a].write;
        unsigned int total_b = trans_counter[master_b].read + trans_counter[master_b].write;

        // If transaction counts differ, prefer master with fewer transactions
        if (total_a != total_b) {
            return total_a < total_b;
        }
        // Tie-breaker: lower ID wins
        return master_a < master_b;
    }
};

/**
 * Least Recently Used Arbiter
 * Prioritizes masters that haven't transacted recently
 */
template<typename CFG>
class LRUArbiter : public AXIArbiterBase<CFG> {
private:
    std::map<unsigned int, uint64_t> last_transaction_time;
    uint64_t current_cycle = 1;  // Start at 1 so default 0 means "never used"

public:
    using MasterTransactionCount = typename AXIArbiterBase<CFG>::MasterTransactionCount;

    bool arbitrate(unsigned int master_a, unsigned int master_b,
                  const MasterTransactionCount* trans_counter) override {
        uint64_t time_a = last_transaction_time[master_a];
        uint64_t time_b = last_transaction_time[master_b];

        // Master with older last transaction time gets priority
        if (time_a != time_b) {
            return time_a < time_b;
        }
        // Tie-breaker: lower ID wins
        return master_a < master_b;
    }

    void update_on_transaction(unsigned int master_id, bool is_read) override {
        last_transaction_time[master_id] = current_cycle++;
    }
};

/**
 * Priority-Based Arbiter
 * Lower master ID = higher priority
 */
template<typename CFG>
class PriorityArbiter : public AXIArbiterBase<CFG> {
public:
    using MasterTransactionCount = typename AXIArbiterBase<CFG>::MasterTransactionCount;

    bool arbitrate(unsigned int master_a, unsigned int master_b,
                  const MasterTransactionCount* trans_counter) override {
        // Lower ID = higher priority
        return master_a < master_b;
    }
};

/**
 * String to ArbiterPolicy conversion
 */
inline ArbiterPolicy string_to_arbiter_policy(const std::string& policy_str) {
    std::string lower = policy_str;
    // Convert to lowercase
    for (char& c : lower) {
        c = std::tolower(c);
    }

    if (lower == "round_robin" || lower == "roundrobin") {
        return ArbiterPolicy::ROUND_ROBIN;
    } else if (lower == "lru" || lower == "least_recently_used") {
        return ArbiterPolicy::LEAST_RECENTLY_USED;
    } else if (lower == "priority" || lower == "priority_based") {
        return ArbiterPolicy::PRIORITY_BASED;
    } else {
        // Default to round robin
        return ArbiterPolicy::ROUND_ROBIN;
    }
}

/**
 * Arbiter Factory
 */
template<typename CFG>
class AXIArbiterFactory {
public:
    static AXIArbiterBase<CFG>* create(ArbiterPolicy policy) {
        switch (policy) {
            case ArbiterPolicy::ROUND_ROBIN:
                return new RoundRobinArbiter<CFG>();
            case ArbiterPolicy::LEAST_RECENTLY_USED:
                return new LRUArbiter<CFG>();
            case ArbiterPolicy::PRIORITY_BASED:
                return new PriorityArbiter<CFG>();
            default:
                return new RoundRobinArbiter<CFG>();
        }
    }

    // Create from string
    static AXIArbiterBase<CFG>* create_from_string(const std::string& policy_str) {
        return create(string_to_arbiter_policy(policy_str));
    }
};

#endif // AXI_ARBITER_H
