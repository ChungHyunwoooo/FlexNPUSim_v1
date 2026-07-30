/**
 * @file npu_controller_timeline.cpp
 * @brief Timeline tracing + state-transition bookkeeping
 *
 * Split from npu_controller.cpp (behaviour-neutral); see
 * npu_controller.h for the class and npu_controller_internal.h for
 * the shared free helpers.
 */

#include "systemc/npu/npu_controller.h"
#include "systemc/npu/npu_controller_internal.h"
#include "system/packet_execution_policy.h"
#include "common/debug_log.h"
#include "compiler/dnn_image/tile_search.h"
#include "model/function/layer_function_runner.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "model/latency/latency_model_chain.h"
#include "systemc/memory/model/peak_bw_layer_model.h"

namespace flexnpu_sim {

static std::string short_process_name(const char* raw_name) {
    if (!raw_name) return "unknown";
    std::string n(raw_name);
    const size_t p = n.rfind('.');
    if (p != std::string::npos && p + 1 < n.size()) {
        return n.substr(p + 1);
    }
    return n;
}

static std::string infer_wait_subject_from_detail(const char* detail) {
    std::string d = detail ? detail : "";
    if (d.rfind("read_", 0) == 0) return "read_thread";
    if (d.rfind("compute_", 0) == 0) return "compute_thread";
    if (d.rfind("write_", 0) == 0) return "write_thread";
    return "wait_unknown";
}
uint64_t NpuController::now_ns() const {
    return static_cast<uint64_t>(sc_time_stamp().to_seconds() * 1e9);
}

const char* NpuController::state_to_string(NpuState s) const {
    switch (s) {
        case NpuState::Idle: return "Idle";
        case NpuState::LoadLayerDesc: return "LoadLayerDesc";
        case NpuState::DmaReadInput: return "DmaReadInput";
        case NpuState::DmaReadKernel: return "DmaReadKernel";
        case NpuState::DmaAndCompute: return "DmaAndCompute";
        case NpuState::Compute: return "Compute";
        case NpuState::DmaWriteOutput: return "DmaWriteOutput";
        case NpuState::LayerComplete: return "LayerComplete";
        case NpuState::AllDone: return "AllDone";
        default: return "Unknown";
    }
}

void NpuController::add_timeline_event(const std::string& type,
                                       uint32_t layer,
                                       uint64_t start_ns,
                                       uint64_t end_ns,
                                       uint64_t bytes,
                                       const std::string& detail,
                                       uint32_t lm_input_i,
                                       uint32_t lm_input_w,
                                       uint32_t lm_output,
                                       const std::string& subject) {
    if (!timeline_enabled_) return;
    if (start_ns < timeline_run_start_ns_) start_ns = timeline_run_start_ns_;
    if (end_ns < timeline_run_start_ns_) end_ns = timeline_run_start_ns_;
    if (end_ns <= start_ns) end_ns = start_ns + 1;
    std::string event_subject = subject;
    if (event_subject.empty()) {
        event_subject = current_subject();
    }
    timeline_events_.push_back(TimelineEvent{
        type, detail, event_subject, layer, start_ns, end_ns, bytes, lm_input_i, lm_input_w, lm_output
    });
}

std::string NpuController::current_subject() const {
    const auto proc = sc_get_current_process_handle();
    if (!proc.valid()) return "unknown";
    return short_process_name(proc.name());
}

void NpuController::flush_state_timeline() {
    if (!timeline_enabled_ || !timeline_state_active_) return;
    add_timeline_event("state",
                       timeline_state_layer_,
                       timeline_state_start_ns_,
                       now_ns(),
                       0,
                       state_to_string(timeline_state_),
                       0,
                       0,
                       0,
                       "fsm");
    timeline_state_active_ = false;
}

void NpuController::set_state(NpuState s) {
    if (state == s) return;
    FLEXNPU_LOG(Info, state, "%d -> %d (layer=%u sim_ns=%lu)",
                (int)state, (int)s, current_layer_idx_, now_ns());
    flush_state_timeline();
    state = s;
    if (timeline_enabled_) {
        timeline_state_active_ = true;
        timeline_state_ = s;
        timeline_state_layer_ = current_layer_idx_;
        timeline_state_start_ns_ = now_ns();
    }
}

void NpuController::wait_with_timeline(sc_event& ev,
                                       uint32_t layer,
                                       const char* detail) {
    if (!timeline_enabled_) {
        wait(ev);
        return;
    }
    const uint64_t t0 = now_ns();
    wait(ev);
    add_timeline_event("wait",
                       layer,
                       t0,
                       now_ns(),
                       0,
                       detail ? detail : "",
                       0,
                       0,
                       0,
                       infer_wait_subject_from_detail(detail));
}

void NpuController::dump_timeline_trace() const {
    if (!timeline_enabled_ || timeline_output_path_.empty()) return;

    std::ofstream ofs(timeline_output_path_, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "WARN: failed to open timeline output: "
                  << timeline_output_path_ << "\n";
        return;
    }
    ofs << "type,detail,subject,layer,start_ns,end_ns,duration_ns,bytes,lm_input_i,lm_input_w,lm_output\n";
    for (const auto& ev : timeline_events_) {
        uint64_t dur = (ev.end_ns >= ev.start_ns) ? (ev.end_ns - ev.start_ns) : 0;
        ofs << ev.type << ","
            << ev.detail << ","
            << ev.subject << ","
            << ev.layer << ","
            << ev.start_ns << ","
            << ev.end_ns << ","
            << dur << ","
            << ev.bytes << ","
            << ev.lm_input_i << ","
            << ev.lm_input_w << ","
            << ev.lm_output << "\n";
    }
}


}  // namespace flexnpu_sim
