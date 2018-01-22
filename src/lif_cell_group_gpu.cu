#include <lif_cell_group_gpu.hpp>
#include <backends/gpu/kernels/stack.hpp>

using namespace arb;

// Constructor containing gid of first cell in a group and a container of all cells.
lif_cell_group_gpu::lif_cell_group_gpu(std::vector<cell_gid_type> gids, const recipe& rec):
gids_(std::move(gids))
{
    // Default to no binning of events.
    set_binning_policy(binning_kind::none, 0);

    // reserve
    cells_.reserve(gids_.size());
    // cell description variables
    tau_m_.reserve(gids_.size());
    V_th_.reserve(gids_.size());
    C_m_.reserve(gids_.size());
    E_L_.reserve(gids_.size());
    V_m_.reserve(gids_.size());
    V_reset_.reserve(gids_.size());
    t_ref_.reserve(gids_.size());

    // resize
    last_time_updated_.resize(gids_.size());

    for (auto lid: util::make_span(0, gids_.size())) {
        cells_.push_back(util::any_cast<lif_cell_description>(rec.get_cell_description(gids_[lid])));
        auto cell = cells_[lid];

        tau_m_.push_back(cell.tau_m);
        V_th_.push_back(cell.V_th);
        C_m_.push_back(cell.C_m);
        E_L_.push_back(cell.E_L);
        V_m_.push_back(cell.V_m);
        V_reset_.push_back(cell.V_reset);
        t_ref_.push_back(cell.t_ref);
    }

    // A buffer collecting the spikes produced by all cells in this group
    // during the invokation of advance(...) method
    // We assume that no neuron will spike more than 10 times
    // during one min_delay period.
    // The refractory period should prevent the neuron from spiking
    // more than this threshold.
    spike_stack = gpu::make_managed_ptr<stack_type>(gids.size() * 10);
}

cell_kind lif_cell_group_gpu::get_cell_kind() const {
    return cell_kind::lif_neuron;
}

const std::vector<spike>& lif_cell_group_gpu::spikes() const {
    return spikes_;
}

void lif_cell_group_gpu::clear_spikes() {
    spikes_.clear();
}

// TODO: implement sampler
void lif_cell_group_gpu::add_sampler(sampler_association_handle h, cell_member_predicate probe_ids,
                                    schedule sched, sampler_function fn, sampling_policy policy) {}
void lif_cell_group_gpu::remove_sampler(sampler_association_handle h) {}
void lif_cell_group_gpu::remove_all_samplers() {}

// TODO: implement binner_
void lif_cell_group_gpu::set_binning_policy(binning_kind policy, time_type bin_interval) {
}

void lif_cell_group_gpu::reset() {
    spikes_.clear();
    last_time_updated_.clear();
}

__global__
void advance_kernel (time_type tfinal,
                    unsigned num_cells,
                    double* tau_m,
                    double* V_th,
                    double* C_m,
                    double* E_L,
                    double* V_m,
                    double* V_reset,
                    double* t_ref,
                    time_type* last_time_updated,
                    pse_vector* event_lanes,
                    lif_cell_group_gpu::stack_type* spike_stack)
{
    int lid = threadIdx.x + blockIdx.x * blockDim.x;
    if (lid >= num_cells) return;

    pse_vector event_lane = event_lanes[lid];

    // Current time of last update.
    time_type t = last_time_updated[lid];
    unsigned i = 0;

    // If a neuron was in the refractory period,
    // ignore any new events that happened before t,
    // including poisson events as well.
    for (auto ev : event_lane) {
        if (ev.time >= t) break;
        ++i;
    }

    // Integrate until tfinal using the exact solution of membrane voltage differential equation.
    for (; i < event_lane.size(); i++) {
        auto ev = event_lane[i];
        if (ev.time >= tfinal) break;

        auto weight = ev.weight;
        auto time = ev.time;

        // If a neuron is in refractory period, ignore this event.
        if (time < t) continue;

        // if there are events that happened at the same time as this event, process them as well
        while (i + 1 < event_lane.size() && event_lane[i+1].time <= time) {
            weight += event_lane[i+1].weight;
            ++i;
        }

        // Let the membrane potential decay.
        V_m[lid] *= exp(-(time - t) / tau_m[lid]);
        // Add jump due to spike.
        V_m[lid] += weight/C_m[lid];
        t = time;
        // If crossing threshold occurred
        if (V_m[lid] >= V_th[lid]) {
            gpu::threshold_crossing spike;
            spike.index = lid;
            spike.time = t;
            gpu::push_back<postsynaptic_spike_event>(spike_stack->storage(), spike);

            // Advance last_time_updated.
            t += t_ref[lid];

            // Reset the voltage to the resting potential.
            V_m[lid] = E_L[lid];
        }
        // This is the last time a cell was updated.
        last_time_updated[lid] = t;
    }
}

void lif_cell_group_gpu::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE("lif");
    if (event_lanes.size() <= 0) return;

    unsigned block_dim = 128;
    unsigned grid_dim = (cells_.size() - 1) / block_dim + 1;

    advance_kernel<<<grid_dim, block_dim>>>(ep.tfinal,
                                            cells_.size(),
                                            tau_m_.data(),
                                            V_th_.data(),
                                            C_m_.data(),
                                            E_L_.data(),
                                            V_m_.data(),
                                            V_reset_.data(),
                                            t_ref_.data(),
                                            last_time_updated_.data(),
                                            event_lanes.data(),
                                            spike_stack.get());
    cudaDeviceSynchronize();

    for (unsigned i = 0; i < spike_stack->size(); ++i) {
        gpu::threshold_crossing crossing = (*spike_stack)[i];
        spikes_.push_back(spike({gids_[crossing.index], 0}, crossing.time));
    }

    spike_stack->clear();
    PL();
}
