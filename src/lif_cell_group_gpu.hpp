#pragma once
#include <cell_group.hpp>
#include <backends/gpu/managed_ptr.hpp>
#include <backends/gpu/stack.hpp>
#include <event_queue.hpp>
#include <lif_cell_description.hpp>
#include <memory/memory.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <threading/timer.hpp>
#include <util/unique_any.hpp>
#include <vector>

namespace arb {
class lif_cell_group_gpu: public cell_group {
public:
    using value_type = double;

    lif_cell_group_gpu() = default;

    // Constructor containing gid of first cell in a group and a container of all cells.
    lif_cell_group_gpu(std::vector<cell_gid_type> gids, const recipe& rec);

    virtual cell_kind get_cell_kind() const override;
    virtual void reset() override;
    virtual void set_binning_policy(binning_kind policy, time_type bin_interval) override;
    virtual void advance(epoch epoch, time_type dt, const event_lane_subrange& events) override;

    virtual const std::vector<spike>& spikes() const override;
    virtual void clear_spikes() override;

    // Sampler association methods below should be thread-safe, as they might be invoked
    // from a sampler call back called from a different cell group running on a different thread.
    virtual void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) override;
    virtual void remove_sampler(sampler_association_handle) override;
    virtual void remove_all_samplers() override;

    template <typename T>
    using managed_vector = std::vector<T, memory::managed_allocator<T> >;
    using stack_type = gpu::stack<gpu::threshold_crossing>;

private:
    // Stack for collecting spikes.
    gpu::managed_ptr<stack_type> spike_stack;

    // LIF parameters.
    managed_vector<double> tau_m_;
    managed_vector<double> V_th_;
    managed_vector<double> C_m_;
    managed_vector<double> E_L_;
    managed_vector<double> V_m_;
    managed_vector<double> V_reset_;
    managed_vector<double> t_ref_;
    std::vector<cell_gid_type> gids_;

    // Cells that belong to this group.
    std::vector<lif_cell_description> cells_;

    // Spikes that are generated (not necessarily sorted).
    std::vector<spike> spikes_;

    // Time when the cell was last updated.
    managed_vector<time_type> last_time_updated_;
};
} // namespace arb
