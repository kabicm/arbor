#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

#include <algorithms.hpp>
#include <cell.hpp>
#include <common_types.hpp>
#include <event_queue.hpp>
#include <spike.hpp>
#include <spike_source.hpp>
#include <util/debug.hpp>
#include <util/partition.hpp>
#include <util/range.hpp>

#include <profiling/profiler.hpp>

namespace nest {
namespace mc {

template <typename LoweredCell>
class cell_group {
public:
    using iarray = cell_gid_type;
    using lowered_cell_type = LoweredCell;
    using value_type = typename lowered_cell_type::value_type;
    using size_type  = typename lowered_cell_type::value_type;
    using spike_detector_type = spike_detector<lowered_cell_type>;
    using source_id_type = cell_member_type;

    using time_type = float;
    using sample_type = double;
    using sampler_function = std::function<util::optional<time_type>(time_type, sample_type)>;

    struct spike_source_type {
        source_id_type source_id;
        spike_detector_type source;
    };

    cell_group() = default;

    template <typename Cells>
    cell_group(cell_gid_type first_gid, const Cells& cells):
        gid_base_{first_gid}
    {
        // Create lookup structure for probe and target ids.
        build_handle_partitions(cells);
        std::size_t n_probes = probe_handle_divisions_.back();
        std::size_t n_targets = target_handle_divisions_.back();
        std::size_t n_detectors =
            algorithms::sum(util::transform_view(cells, [](const cell& c) { return c.detectors().size(); }));

        // Allocate space to store handles / probes info.
        detector_handles_.resize(n_detectors);
        target_handles_.resize(n_targets);

        sample_cache_.resize(n_probes);  // nr of handles
        sampler_handler_dt_.resize(n_probes);
        probe_handles_.resize(n_probes);
        //samplers_ size is not known before hand, needs to be dynamic
 

        cell_.initialize(cells, detector_handles_, target_handles_, probe_handles_);

        // initialize the sample_cache_
        for (auto& item : sample_cache_) {
            // Insert the first sample (is always on t=0.0
            item.push_back(std::make_pair( 0.0, 0.0));  // For now push zero: We need to 
            //poll the underlaying lowered cell for the default start value
        }
        // We cannot set the dt: they sample rates are only known after adding
        // of the samplers


        // Create spike detectors and associate them with globally unique source ids.
        cell_gid_type source_gid = gid_base_;
        unsigned i = 0;
        for (const auto& cell: cells) {
            cell_lid_type source_lid = 0u;
            for (auto& d: cell.detectors()) {
                cell_member_type source_id{source_gid, source_lid++};

                spike_sources_.push_back({
                    source_id, spike_detector_type(cell_, detector_handles_[i++],  d.threshold, 0.f)
                });
            }
            ++source_gid;
        }
    }

    void reset() {
        clear_spikes();
        clear_events();
        reset_samplers();
        cell_.reset();
        for (auto& spike_source: spike_sources_) {
            spike_source.source.reset(cell_, 0.f);
        }
    }

    time_type min_step(time_type dt) {
        return 0.1*dt;
    }

    void advance(time_type tfinal, time_type dt) {
        
        // Advance the cell state
        while (cell_.time()<tfinal) {
            // take any pending samples
            time_type cell_time = cell_.time();
            PE("sampling");
            // We have a queue of sample events
            // they keep track of when we want to have a value
            // We want to have a value when:
            //  - When the time exceeds the sample time.
            //   event is a sample time.
            // Move out of the loop
            while (auto m = sample_events_.pop_if_before(cell_time)) {
                auto& s = samplers_[m->sampler_index];
                EXPECTS((bool)s.sampler);
                auto next = // the next time to sample (can depend on whatever) can be never or a time
                            //Sampler is the cpu side: writes to file or does some online calculation: Its user space
                            // we can have multiple samplers per probe
                    s.sampler(cell_.time(), // now store the value with the time stamp
                        cell_.probe( // Gives the number ( this function thus has a send and receive on the gpu)
                            s.handle) // Rich pointer to what we want to sample
                    );  // This talks to the cell (on GPU)

                if (next) { // If we want to sample more, push the sample event
                    m->time = std::max(*next, cell_time);
                    sample_events_.push(*m);
                }
            }
            PL();

            // look for events in the next time step
            time_type tstep = cell_.time()+dt;
            tstep = std::min(tstep, tfinal);
            auto next = events_.pop_if_before(tstep);

            // apply events that are due within the smallest allowed time step.
            while (next && (next->time-cell_.time()) < min_step(dt)) {
                auto handle = get_target_handle(next->target);
                cell_.deliver_event(handle, next->weight);
                next = events_.pop_if_before(tstep);
            }

            // integrate cell state
            time_type tnext = next ? next->time: tstep;
            cell_.advance(tnext - cell_.time());

            if (!cell_.is_physical_solution()) {
                std::cerr << "warning: solution out of bounds for cell "
                          << gid_base_ << " at t " << cell_.time() << " ms\n";
            }

            PE("events");
            // check for new spikes
            for (auto& s : spike_sources_) {
                if (auto spike = s.source.test(cell_, cell_.time())) {
                    spikes_.push_back({s.source_id, spike.get()});
                }
            }

            // apply events
            if (next) {
                auto handle = get_target_handle(next->target);
                cell_.deliver_event(handle, next->weight);
            }
            PL();
        }

        // Get the sampler handler data from the gpu

        // process the data
        //while (auto m = sample_events_.pop_if_before(tfinal)) {  //
        //    auto& s = samplers_[m->sampler_index];
        //    expects((bool)s.sampler);
        //    // find the sample in cache which is the closest by



        //    auto next = // the next time to sample (can depend on whatever) can be never or a time
        //                //sampler is the cpu side: writes to file or does some online calculation: its user space
        //                // we can have multiple samplers per probe
        //        s.sampler(cell_.time(), // now store the value with the time stamp
        //            cell_.probe( // gives the number ( this function thus has a send and receive on the gpu)
        //                s.handle) // rich pointer to what we want to sample
        //        );  // this talks to the cell (on gpu)

        //    if (next) { // if we want to sample more, push the sample event
        //        m->time = std::max(*next, cell_time);
        //        sample_events_.push(*m);
        //    }
        //}



        // Process the data to the samplers

        // //PE("sampling");
        // // Pseudo code
        // // 1 Process all sample events for the previous time step

        // // 2 Set all sampling dt, that are started in the next time step

        // auto start_time_it = sampler_start_times_.begin();
        // auto start_time_end = sampler_start_times_.end();
        // auto samplers_it = samplers_.begin();
        // auto samplers_end = samplers_.end();

        //while (start_time_it != start_time_end &&
        //       samplers_it   != samplers_end)  // Should be the same size
        // {

        // }



    }

    template <typename R>
    void enqueue_events(const R& events) {
        for (auto e : events) {
            events_.push(e);
        }
    }

    const std::vector<spike<source_id_type, time_type>>&
    spikes() const { return spikes_; }

    const std::vector<spike_source_type>&
    spike_sources() const {
        return spike_sources_;
    }

    void clear_spikes() {
        spikes_.clear();
    }

    void clear_events() {
        events_.clear();
    }

    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) {
        auto handle = get_probe_handle(probe_id);

        auto sampler_index = uint32_t(samplers_.size());
        samplers_.push_back({ handle, s });
        sampler_start_times_.push_back(start_time);

        // If we have start time of 0.0 we need to start sampling from the start
        float sim_start_t = 0.0;
        if (start_time == sim_start_t)
        {
            // We need to know the handle idx, it can be shared between samplers
            auto handle_idx = handle_partition_lookup(probe_handle_divisions_, probe_id);
            // Get the next time we must sample from the sampler_function

            auto next = s(sim_start_t, 
                sample_cache_[handle_idx].front().second);

            if (next) 
            {   // if we want to sample more, push the sample event
                sample_events_.push({ sampler_index, *next } );

                // But also store the delta because we need to tell the cell
                // to start measuring
                auto dt = *next - sim_start_t;
                // if this dt is smaller then what we have in the current dt vector
                if (dt < sampler_handler_dt_[handle_idx])
                {
                    sampler_handler_dt_[handle_idx] = dt;
                }
            }
        }
        else
        {
           sample_events_.push({ sampler_index, start_time });
        }

    }
    void remove_samplers() {
        sample_events_.clear();
        samplers_.clear();
        sampler_start_times_.clear();
    }

    void reset_samplers() {
        // clear all pending sample events and reset to start at time 0
        sample_events_.clear();
        for(uint32_t i=0u; i<samplers_.size(); ++i) {
            sample_events_.push({i, sampler_start_times_[i]});
        }
        // TODOW
        // push dt to the gpu
    }

    value_type probe(cell_member_type probe_id) const {
        return cell_.probe(get_probe_handle(probe_id));
    }

private:
    /// gid of first cell in group
    cell_gid_type gid_base_;

    /// the lowered cell state (e.g. FVM) of the cell
    lowered_cell_type cell_;

    /// spike detectors attached to the cell
    std::vector<spike_source_type> spike_sources_;

    /// spikes that are generated
    std::vector<spike<source_id_type, time_type>> spikes_;

    /// pending events to be delivered
    event_queue<postsynaptic_spike_event<time_type>> events_;

    /// pending samples to be taken
    event_queue<sample_event<time_type>> sample_events_;
    std::vector<time_type> sampler_start_times_;

    // For each handler the current sampling dt, we use only a single dt
    // for each handler (with possible multiple samplers)
    std::vector<time_type> sampler_handler_dt_;

    // Per sampler handler cache of samples, will be filled per
    // time step with data
    std::vector<std::vector<std::pair<time_type, sample_type>>> sample_cache_;


    /// the global id of the first target (e.g. a synapse) in this group
    iarray first_target_gid_;

    /// handles for accessing lowered cell
    using detector_handle = typename lowered_cell_type::detector_handle;
    std::vector<detector_handle> detector_handles_;

    using target_handle = typename lowered_cell_type::target_handle;
    std::vector<target_handle> target_handles_;

    using probe_handle = typename lowered_cell_type::probe_handle;
    std::vector<probe_handle> probe_handles_;

    struct sampler_entry {
        typename lowered_cell_type::probe_handle handle;
        sampler_function sampler;
    };

    /// collection of samplers to be run against probes in this group
    std::vector<sampler_entry> samplers_;

    /// lookup table for probe ids -> local probe handle indices
    std::vector<std::size_t> probe_handle_divisions_;

    /// lookup table for target ids -> local target handle indices
    std::vector<std::size_t> target_handle_divisions_;

    /// build handle index lookup tables
    template <typename Cells>
    void build_handle_partitions(const Cells& cells) {
        auto probe_counts = util::transform_view(cells, [](const cell& c) { return c.probes().size(); });
        auto target_counts = util::transform_view(cells, [](const cell& c) { return c.synapses().size(); });

        make_partition(probe_handle_divisions_, probe_counts);
        make_partition(target_handle_divisions_, target_counts);
    }

    /// use handle partition to get index from id
    template <typename Divisions>
    std::size_t handle_partition_lookup(const Divisions& divisions, cell_member_type id) const {
        // NB: without any assertion checking, this would just be:
        // return divisions[id.gid-gid_base_]+id.index;

        EXPECTS(id.gid>=gid_base_);

        auto handle_partition = util::partition_view(divisions);
        EXPECTS(id.gid-gid_base_<handle_partition.size());

        auto ival = handle_partition[id.gid-gid_base_];
        std::size_t i = ival.first + id.index;
        EXPECTS(i<ival.second);

        return i;
    }

    /// get probe handle from probe id
    probe_handle get_probe_handle(cell_member_type probe_id) const {
        return probe_handles_[handle_partition_lookup(probe_handle_divisions_, probe_id)];
    }

    /// get target handle from target id
    target_handle get_target_handle(cell_member_type target_id) const {
        return target_handles_[handle_partition_lookup(target_handle_divisions_, target_id)];
    }
};

} // namespace mc
} // namespace nest
