/**
 * @file test_axi_arbiter.cpp
 * @brief Contract tests for the AXI bus arbiter (pure C++, no SystemC).
 *
 * Contracts (from axi_arbiter.h's documented policy semantics):
 *  - RoundRobin: equal-share rotation, idle masters skipped, -1 when idle
 *  - WeightedRR: sticky credits — weight w wins w consecutive grants;
 *    replenish when the reachable credits drain (w0:w1 grant ratio holds)
 *  - FixedPriority: lowest priority value always wins (documented starvation)
 *  - PriorityRR: round-robin restricted to the best-priority requesting tier
 *  - QoSAware: highest live QoS wins; set_qos() changes it per transaction;
 *    round-robin among equals
 *  - errors: empty master list and size-mismatched request vectors throw
 */

#include "systemc/bus/axi/bus/axi_arbiter.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace flexnpu_sim::transport::axi;

static std::vector<MasterArbitCfg> masters(std::initializer_list<MasterArbitCfg> l) {
    return {l};
}

static void test_round_robin() {
    Arbiter a(ArbitPolicy::RoundRobin, masters({{}, {}, {}}));
    const std::vector<bool> all{true, true, true};
    assert(a.grant(all) == 0);
    assert(a.grant(all) == 1);
    assert(a.grant(all) == 2);
    assert(a.grant(all) == 0);                    // wraps
    assert(a.grant({false, false, true}) == 2);   // idle masters skipped
    assert(a.grant({false, false, false}) == -1); // nobody asks
}

static void test_weighted_rr() {
    // weight 2:1 -> grant pattern 0,0,1 repeating (sticky credits).
    Arbiter a(ArbitPolicy::WeightedRR,
              masters({{.weight = 2}, {.weight = 1}}));
    const std::vector<bool> all{true, true};
    int g0 = 0, g1 = 0;
    for (int i = 0; i < 9; ++i) (a.grant(all) == 0 ? g0 : g1)++;
    assert(g0 == 6 && g1 == 3);                   // exact 2:1 share
    // First three grants are the canonical 0,0,1.
    Arbiter b(ArbitPolicy::WeightedRR,
              masters({{.weight = 2}, {.weight = 1}}));
    assert(b.grant(all) == 0 && b.grant(all) == 0 && b.grant(all) == 1);
}

static void test_fixed_priority() {
    // lower value wins; sustained contention starves the loser (documented).
    Arbiter a(ArbitPolicy::FixedPriority,
              masters({{.priority = 2}, {.priority = 0}, {.priority = 1}}));
    const std::vector<bool> all{true, true, true};
    for (int i = 0; i < 4; ++i) assert(a.grant(all) == 1);
    assert(a.grant({true, false, true}) == 2);    // next tier when 1 idle
    assert(a.grant({true, false, false}) == 0);
}

static void test_priority_rr() {
    // masters 0,1 in tier 0; master 2 in tier 1. RR inside the best tier.
    Arbiter a(ArbitPolicy::PriorityRR,
              masters({{.priority = 0}, {.priority = 0}, {.priority = 1}}));
    const std::vector<bool> all{true, true, true};
    assert(a.grant(all) == 0);
    assert(a.grant(all) == 1);
    assert(a.grant(all) == 0);                    // tier-0 rotation, 2 excluded
    assert(a.grant({false, false, true}) == 2);   // lower tier only when idle
}

static void test_qos_aware() {
    Arbiter a(ArbitPolicy::QoSAware,
              masters({{.default_qos = 0}, {.default_qos = 3}, {.default_qos = 3}}));
    const std::vector<bool> all{true, true, true};
    assert(a.grant(all) == 1);                    // highest QoS tier wins...
    assert(a.grant(all) == 2);                    // ...round-robin among equals
    a.set_qos(0, 7);                              // per-transaction escalation
    assert(a.grant(all) == 0);
    a.set_qos(0, 0);
    assert(a.grant(all) == 1 || a.grant(all) == 2);
}

static void test_errors() {
    bool threw = false;
    try {
        Arbiter a(ArbitPolicy::RoundRobin, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    Arbiter a(ArbitPolicy::RoundRobin, masters({{}, {}}));
    threw = false;
    try {
        (void)a.grant({true});                    // size mismatch
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_round_robin();
    test_weighted_rr();
    test_fixed_priority();
    test_priority_rr();
    test_qos_aware();
    test_errors();
    std::cout << "test_axi_arbiter: all contracts hold\n";
    return 0;
}
