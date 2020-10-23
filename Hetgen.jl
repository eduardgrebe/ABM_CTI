# Copyright (C) 2020 Eduard Grebe and Nathan Geffen.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.  This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.  You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

module Hetgen

    export default_parameters, set_parameters!, set_parameters, 
            run_one_sim!, run_simulations, write_csv_header, Jiggle

    using Random
    using Distributions
    import Base: rand, append!
    using IterTools
    using DataStructures
    using Base.Threads

    const ISOLATED_BEFORE = -11
    const INFECTED_BEFORE = -1
    const NONE = -2

    const POS_SUSCEPTIBLE = 0.001
    const POS_EXPOSED = 0.5
    const POS_INFECTIOUS_A = 0.9
    const POS_INFECTIOUS_S = 0.999
    const POS_INFECTIOUS_H = 0.999
    const POS_INFECTIOUS_I = 0.999
    const POS_RECOVERED = 0.3
    const POS_DEAD = 0.99

    const TEST_SUSCEPTIBLE = 0.0
    const TEST_EXPOSED = 0.0
    const TEST_INFECTIOUS_A = 0.0
    const TEST_INFECTIOUS_S = 0.3
    const TEST_INFECTIOUS_H = 0.0
    const TEST_INFECTIOUS_I = 0.0
    const TEST_RECOVERED = 0.0
    const TEST_DEAD = 0.0

    IntNull = Union{Int64,Nothing}
    FloatNull = Union{Float64,Nothing}

    @enum TestResult NEGATIVE = 0 POSITIVE = 1

    @enum Health begin
        SUSCEPTIBLE = 0
        EXPOSED = 1
        INFECTIOUS_A = 2
        INFECTIOUS_S = 3
        INFECTIOUS_H = 4
        INFECTIOUS_I = 5
        RECOVERED = 6
        DEAD = 7
    end

    function rand_0_1(rng::Random.AbstractRNG, n::IntNull)
        d = Distributions.Uniform(0.0, 1.0)
        r = n === nothing ? rand(rng, d) : rand(rng, d, n)
        return r
    end

    function rand_0_1(rng)
        rand_0_1(rng, nothing)
    end

    function rand_0_1(n::Integer)
        rand_0_1(Random.GLOBAL_RNG, n)
    end

    function rand_0_1()
        rand_0_1(Random.GLOBAL_RNG, nothing)
    end

    mutable struct Infection
        id::IntNull
        iteration::IntNull
    end

    HealthChangeIters = OrderedDict{Health,AbstractVector{Int64}}

    function HealthChangeItersEmpty()
        return OrderedDict{Health,AbstractVector{Int64}}([
            SUSCEPTIBLE  => Vector{Int64}(),
            EXPOSED      => Vector{Int64}(),
            INFECTIOUS_A => Vector{Int64}(),
            INFECTIOUS_S => Vector{Int64}(),
            INFECTIOUS_H => Vector{Int64}(),
            INFECTIOUS_I => Vector{Int64}(),
            RECOVERED    => Vector{Int64}(),
            DEAD         => Vector{Int64}()
        ])
    end

    function append!(s::HealthChangeIters, h::Health, i::Integer)
        append!(s[h], i)
    end

    function HealthProbs(x::AbstractVector{<:AbstractFloat})
        if length(x) == 4
            return OrderedDict{Health,Float64}([
                SUSCEPTIBLE  => 0.0,
                EXPOSED      => 0.0,
                INFECTIOUS_A => x[1], 
                INFECTIOUS_S => x[2], 
                INFECTIOUS_H => x[3], 
                INFECTIOUS_I => x[4],
                RECOVERED    => 0.0,
                DEAD         => 0.0
                ])
        elseif length(x) == 8
            return OrderedDict{Health,Float64}([
                SUSCEPTIBLE  => x[1],
                EXPOSED      => x[2],
                INFECTIOUS_A => x[3], 
                INFECTIOUS_S => x[4], 
                INFECTIOUS_H => x[5], 
                INFECTIOUS_I => x[6],
                RECOVERED    => x[7],
                DEAD         => x[8]
                ])
        else
            error("Only 4 or 8 probabilities accepted")
        end
    end

    # Performance advantage to making immutable?
    # Currently the default value is l even when a range is specified
    # the string representation is the range, irrespective of whether the value
    # has been set to a random number or 
    mutable struct Jiggle{T<:Union{Integer,AbstractFloat}}
        l::T
        u::Union{T,Nothing}
        r::String # string representation
        v::Union{T,Nothing} # current value ; here I diverge from Nathan's approach
        function Jiggle(l)
            new{typeof(l)}(l, nothing, string(l), l) # default value = l!
        end
        function Jiggle(l,u)
            if u === nothing
                Jiggle(l)
            else
                new{typeof(l)}(
                    l,
                    u, 
                    "[" * string(l) * ":" * string(u) * "]", 
                    l # default value = l!
                    )
            end
        end
    end

    # Random draw methods for Type Jiggle, optional rng, optional n
    function rand(rng::Random.AbstractRNG, s::Jiggle{<:Integer}, n::IntNull)
        if s.u === nothing || s.u == s.l
            r = n === nothing ? s.l : repeat([s.l], n)
        elseif s.u < s.l
            throw(DomainError(s.u, "must be ≥ " * string(s.l)))
        else
            d = s.l:s.u
            r = n === nothing ? rand(rng, d) : rand(rng, d, n)
        end
        return r
    end

    function rand(rng::Random.AbstractRNG, s::Jiggle{<:AbstractFloat}, n::IntNull)
        if s.u === nothing || s.u == s.l
            r = n === nothing ? s.l : repeat([s.l], n)
        elseif s.u < s.l
            throw(DomainError(s.u, "must be ≥ " * string(s.l)))
        else
            d = Distributions.Uniform(s.l, s.u)
            r = n === nothing ? rand(rng, d) : rand(rng, d, n)
        end
        return r
    end

    function rand(s::Jiggle)
        rand(Random.GLOBAL_RNG, s)
    end

    function rand(s::Jiggle, n::Integer)
        rand(Random.GLOBAL_RNG, s, n)
    end

    function rand(rng::Random.AbstractRNG, s::Jiggle)
        rand(rng, s, nothing)
    end

    function set!(rng::Random.AbstractRNG, s::Jiggle)
        s.v = rand(rng, s)
    end

    function set!(s::Jiggle)
        s.v = rand(s)
    end

    function get!(from::Jiggle, to::Jiggle)
        to.v = from.v
    end

    mutable struct Parameters
        first_id::Int64
        scenario::Int64
        jiggle::Int64
        run::Int64
        id::Int64
        num_jiggles::Int64
        num_runs_per_jiggle::Int64
        seed::IntNull
        threads::Int64
        num_iterations::Int64
        num_agents::Int64
        stats_frequency::Int64
        report_frequency::Int64
        initial_infections::Int64
        verbose::Bool
        health::AbstractVector{Float64}
        risk_infection::Float64
        risk_infecting::OrderedDict{Health,Float64}
        risk_positive::OrderedDict{Health,Float64}
        prob_test::OrderedDict{Health,Float64}
        min_isolation::Float64
        max_isolation::Float64
        min_before_isolate::Int64
        trace_effective::Float64
        min_before_trace::Int64
        recover_before_hospital::Float64
        recover_before_icu::Float64
        infectious_h_risk::Float64
        recover_before_death::Float64
        infectious_i_risk::Float64
        exposed_risk::Jiggle
        asymptomatic::Jiggle
        infectious_a_risk::Jiggle
        k_assort::Jiggle
        k_unassort::Jiggle
        k_tracing::Jiggle
        pos_test_susceptible::Jiggle
        pos_test_exposed::Jiggle
        pos_test_infectious_a::Jiggle
        pos_test_infectious_s::Jiggle
        pos_test_infectious_h::Jiggle
        pos_test_infectious_i::Jiggle
        pos_test_recovered::Jiggle
        pos_test_dead::Jiggle
        prob_test_susceptible::Jiggle
        prob_test_exposed::Jiggle
        prob_test_infectious_a::Jiggle
        prob_test_infectious_s::Jiggle
        prob_test_infectious_h::Jiggle
        prob_test_infectious_i::Jiggle
        prob_test_recovered::Jiggle
        prob_test_dead::Jiggle
        mean_test::Jiggle
        min_test::Jiggle
        isolation_period::Jiggle
        infectious_s_risk::Jiggle
    end

    const default_parameters = Parameters(
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        nothing,
        0,
        365,
        10000,
        1,
        20,
        0,
        true,
        [0.999, 1.0],
        0.1,
        HealthProbs([0.0, 0.0, 0.05, 0.1, 0.1, 0.1, 0.0, 0.0]),
        HealthProbs([
            POS_SUSCEPTIBLE,
            POS_EXPOSED,
            POS_INFECTIOUS_A,
            POS_INFECTIOUS_S,
            POS_INFECTIOUS_H,
            POS_INFECTIOUS_I,
            POS_RECOVERED,
            POS_DEAD,
        ]),
        HealthProbs([
            TEST_SUSCEPTIBLE,
            TEST_EXPOSED,
            TEST_INFECTIOUS_A,
            TEST_INFECTIOUS_S,
            TEST_INFECTIOUS_H,
            TEST_INFECTIOUS_I,
            TEST_RECOVERED,
            TEST_DEAD,
        ]),
        0.0,
        1.0,
        0,
        0.9,
        0,
        0.8,
        0.65,
        0.15,
        0.3,
        0.15,
        Jiggle(0.25, nothing),
        Jiggle(0.75, nothing),
        Jiggle(0.5, nothing),
        Jiggle(38, nothing),
        Jiggle(0, nothing),
        Jiggle(38, nothing),
        Jiggle(POS_SUSCEPTIBLE, nothing),
        Jiggle(POS_EXPOSED, nothing),
        Jiggle(POS_INFECTIOUS_A, nothing),
        Jiggle(POS_INFECTIOUS_S, nothing),
        Jiggle(POS_INFECTIOUS_H, nothing),
        Jiggle(POS_INFECTIOUS_I, nothing),
        Jiggle(POS_RECOVERED, nothing),
        Jiggle(POS_DEAD, nothing),
        Jiggle(TEST_SUSCEPTIBLE, nothing),
        Jiggle(TEST_EXPOSED, nothing),
        Jiggle(TEST_INFECTIOUS_A, nothing),
        Jiggle(TEST_INFECTIOUS_S, nothing),
        Jiggle(TEST_INFECTIOUS_H, nothing),
        Jiggle(TEST_INFECTIOUS_I, nothing),
        Jiggle(TEST_RECOVERED, nothing),
        Jiggle(TEST_DEAD, nothing),
        Jiggle(2, nothing),
        Jiggle(2, nothing),
        Jiggle(10, nothing),
        Jiggle(0.2, nothing),
    )

    function set_parameters(parameters::Parameters; kwargs...)
        p = deepcopy(parameters)
        for (key, value) in Dict(kwargs)
            setfield!(p, key, value)
        end
        return p
    end

    function set_parameters!(parameters::Parameters; kwargs...)
        for (key, value) in Dict(kwargs)
            setfield!(parameters, key, value)
        end
    end

    mutable struct Agent
        id::Int64
        risk_infection::Float64
        risk_infecting::OrderedDict{Health,Float64}
        infector::Infection
        infected_by_me::AbstractVector{Infection}
        health_change_iters::HealthChangeIters
        tested::Int
        test_result::TestResult
        test_res_iter::IntNull
        isolation_iter::IntNull
        isolated::Float64
        health::Health
        asymptomatic::Bool
        recover_before_hospital::Bool
        recover_before_icu::Bool
        recover_before_death::Bool

        function Agent(p::Parameters, id::Int, rng::Random.AbstractRNG)
            risk_infecting_ = Vector{Float64}()
            for (h, r) in p.risk_infecting
                if h > EXPOSED && h < RECOVERED
                    append!(risk_infecting_, 
                            rand(rng, Distributions.Exponential(r)))
                end
            end
            if p.initial_infections == 0
                stage = 0
                for d in p.health
                    if rand_0_1(rng) < d
                        health_ = Health(stage)
                        break
                    end
                    stage += 1
                end
            else
                health_ = SUSCEPTIBLE
            end
            recover_before_hospital_ = true
            recover_before_icu_ = true
            recover_before_death_ = true
            # This appears to be Nathan's method using the "()" operator defined on 
            # a Jiggle, which returns l, but if the user specified a range instead 
            # of just a lower value is this the desired behaviour? See line 468
            asymptomatic_ = rand_0_1(rng) < p.asymptomatic.v
            if !asymptomatic_
                recover_before_hospital_ = rand_0_1(rng) < p.recover_before_hospital
                if !recover_before_hospital_
                    recover_before_icu_ = rand_0_1(rng) < p.recover_before_icu
                    if !recover_before_icu_
                        recover_before_death_ = rand_0_1(rng) < 
                            p.recover_before_death
                    end
                end
            end
            new(
                id,
                rand(rng, Distributions.Exponential(p.risk_infection)),
                HealthProbs(risk_infecting_),
                Infection(nothing,nothing),
                Vector{Infection}(),
                HealthChangeItersEmpty(),
                0,
                NEGATIVE,
                nothing,
                nothing,
                0.0,
                health_,
                asymptomatic_,
                recover_before_hospital_,
                recover_before_icu_,
                recover_before_death_,
            )
        end
    end

    mutable struct Simulation
        thread_id::Int64
        rng::Random.AbstractRNG
        parameters::Parameters
        agents::AbstractVector{Agent}
        iteration::IntNull
        num_agents_isolated::Int64
        num_isolated::Int64
        num_deisolated::Int64
        num_traced::Int64
        total_infected::Int64
        num_agents_tested::Int64
        num_tests::Int64
        num_positives::Int64
        peak::Int64
        peak_total_infections::Int64
        peak_iter::Int64
        results::AbstractVector{Dict{String,Float64}}
        # Without some black magic we can't initiate the agents inside this 
        # constructor because the init_agents function calls infect! which
        # in turn modifies the total_infected field of the Simulation struct
        # there's probably an elegant way of doing this, see: 
        # https://docs.julialang.org/en/v1/manual/constructors/#Incomplete-Initialization
        function Simulation(p::Parameters)
            new(
                Threads.threadid(),
                MersenneTwister(p.seed),
                p,
                Vector{Agent}(),
                nothing,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                Vector{Dict{String,Float64}}(),
            )
        end
    end

    # multiple dispatch rather than if from
    function infect!(s::Simulation, from::Agent, to::Agent)
        push!(from.infected_by_me, Infection(to.id, s.iteration))
        to.infector = Infection(from.id, s.iteration)
        s.total_infected += 1
    end

    # Nathan uses the constant INFECTED_BEFORE = -1 for from_id
    # does this become important later?
    function infect!(s::Simulation, to::Agent)
        to.infector = Infection(nothing, s.iteration)
        s.total_infected += 1
    end

    function isolate!(s::Simulation, a::Agent)
        if a.isolation_iter === nothing
            s.num_agents_isolated += 1
        end
        a.isolation_iter = s.iteration + s.parameters.isolation_period.v
        a.isolated = rand_0_1(s.rng) * 
            (s.parameters.max_isolation - s.parameters.min_isolation) +
            s.parameters.min_isolation
        s.num_isolated += 1
    end

    function deisolate!(s::Simulation, a::Agent)
        # I don't understand this. Why the -11 constant?
        a.isolation_iter = - (a.isolation_iter + ISOLATED_BEFORE)
        a.isolated = 0.0
        s.num_deisolated += 1
    end

    function init_agents!(s::Simulation; forced=false)
        if !forced && length(s.agents) != 0
            error("Simulation object already contains agents. Use forced=true to override")
        elseif forced
            s.agents = Vector{Agent}()
        end
        # Start with 1 because Julia is 1-indexed - could it cause problems?
        for i = 1:s.parameters.num_agents
            agent = Agent(s.parameters, i, s.rng)
            if agent.health > SUSCEPTIBLE && agent.health < DEAD
                infect!(s, agent)
            end
            push!(s.agents, agent)
        end
        if s.parameters.initial_infections > 0
            # ordered=true is probably unnecessary
            infect_indices = sample(s.rng, 1:length(s.agents), 
                                    s.parameters.initial_infections, 
                                    replace=false, ordered=true)
            for i in infect_indices
                s.agents[i].health = EXPOSED
                infect!(s, s.agents[i])
            end
        end
    end

    function stats!(s::Simulation; forced = false)
        if forced || s.iteration % s.parameters.stats_frequency == 0
            infections = 0
            for a in s.agents
                infections += Int(a.health > SUSCEPTIBLE && a.health < RECOVERED)
            end
            if s.peak < infections
                s.peak = infections
                s.peak_total_infections = s.total_infected
                s.peak_iter = s.iteration
            end
        end
    end

    function report(s::Simulation, io::IO=IOContext(stdout, :compact => false); 
                    forced = false)
        if forced || s.iteration % s.parameters.report_frequency == 0
            # array comprehension appears to be slightly faster than looping over
            # all agents once and updating counts with if else statements
            susceptible = sum([a.health == SUSCEPTIBLE ? 1 : 0 for a in s.agents])
            exposed = sum([a.health == EXPOSED ? 1 : 0 for a in s.agents])
            infectious_a = sum([a.health == INFECTIOUS_A ? 1 : 0 for a in s.agents])
            infectious_s = sum([a.health == INFECTIOUS_S ? 1 : 0 for a in s.agents])
            infectious_h = sum([a.health == INFECTIOUS_H ? 1 : 0 for a in s.agents])
            infectious_i = sum([a.health == INFECTIOUS_I ? 1 : 0 for a in s.agents])
            recovered = sum([a.health == RECOVERED ? 1 : 0 for a in s.agents])
            dead = sum([a.health == DEAD ? 1 : 0 for a in s.agents])
            active = exposed + infectious_a + infectious_s + infectious_h + 
                infectious_i
            
            lock(io)
            println(
                io,
                string(
                    s.thread_id, ",",
                    s.parameters.id, ",",
                    s.parameters.scenario, ",",
                    s.parameters.jiggle, ",",
                    s.parameters.run, ",",
                    s.iteration, ",",
                    susceptible, ",",
                    exposed, ",",
                    infectious_a, ",",
                    infectious_s, ",",
                    infectious_h, ",",
                    infectious_i, ",",
                    recovered, ",",
                    dead, ",",
                    active, ",",
                    s.total_infected, ",",
                    s.num_agents_isolated, ",",
                    s.num_isolated, ",",
                    s.num_deisolated, ",",
                    s.num_traced, ",",
                    s.num_agents_tested, ",",
                    s.num_tests, ",",
                    s.num_positives, ",",
                    s.parameters.k_assort.v, ",",
                    s.parameters.prob_test_infectious_s.v, ",",
                    s.parameters.mean_test.v, ",",
                    s.parameters.min_test.v, ",",
                    s.parameters.isolation_period.v, ",",
                    s.parameters.exposed_risk.v, ",",
                    s.parameters.asymptomatic.v, ",",
                    s.parameters.infectious_a_risk.v, ",",
                    s.parameters.infectious_s_risk.v, ",",
                    s.parameters.min_isolation, ",",
                    s.parameters.max_isolation, ",",
                    s.parameters.trace_effective
                )
                )
            unlock(io)
        end
    end

    function event_infect_assort!(s::Simulation)
        neighbours = Int64(round(s.parameters.k_assort.v / 2.0))
        n = length(s.agents)
        infected = Vector{Union{Nothing, Int64}}(nothing, n)
        indices = collect(1:n)
        shuffle!(s.rng, indices)
        for i in indices
            if s.agents[i].health > EXPOSED && s.agents[i].health < RECOVERED
                from = max(1, i - neighbours)
                to = min(i + neighbours, n) # i + 1 + neighbors on line 639
                for j in from:to
                    if s.agents[j].health > SUSCEPTIBLE
                        continue
                    end
                    risk = min(1.0 - s.agents[i].isolated, 1.0 - 
                        s.agents[j].isolated) * 
                        ((s.agents[i].risk_infecting[s.agents[i].health] + 
                        s.agents[j].risk_infection) / 2.0)
                    if rand_0_1(s.rng) < risk
                        infected[j] = i
                    end
                end
            end
        end
        for i in 1:n
            if infected[i] !== nothing
                # assert(agents_[infected[i]]->health_ > EXPOSED && 
                #     agents_[infected[i]]->health_ < RECOVERED);
                # assert(agents_[i]->health_ == SUSCEPTIBLE);
                # assert(std::abs(i - infected[i]) <= 
                #     std::round((double)parameters_.k_assort() / 2));
                s.agents[i].health = EXPOSED
                infect!(s, s.agents[infected[i]], s.agents[i])
            end
        end
    end

    function event_test!(s::Simulation)
        for a in s.agents
            if a.test_res_iter === nothing && a.test_result == NEGATIVE && 
                    rand_0_1(s.rng) < s.parameters.prob_test[a.health]
                i = rand(s.rng, Distributions.Poisson(s.parameters.mean_test.v))
                a.test_res_iter = s.iteration + max(i, s.parameters.min_test.v)
                if rand_0_1(s.rng) < s.parameters.risk_positive[a.health]
                    a.test_result = POSITIVE
                    s.num_positives += 1
                else
                    a.test_result = NEGATIVE
                end
                s.num_tests += 1
                if a.tested == 0
                    s.num_agents_tested += 1
                end
                a.tested += 1
            end
        end
    end

    function event_isolate!(s::Simulation)
        if s.parameters.min_before_isolate > 0
            infections = 0
            for a in  s.agents
                if a.health > INFECTIOUS_A
                    infections += 1
                end
            end
            if infections < s.parameters.min_before_isolate
                return
            else
                s.parameters.min_before_isolate = 0
            end
        end
        for a in s.agents
            if a.test_res_iter == s.iteration &&
                    a.test_result == POSITIVE &&
                    a.isolated == 0.0
                isolate!(s, a)
            end
        end
    end

    function event_deisolate!(s::Simulation)
        for a in s.agents
            if a.isolation_iter == s.iteration
                deisolate!(s, a)
            end
        end
    end

    function event_trace!(s::Simulation)
        if s.parameters.min_before_trace > 0
            infections = 0
            for a in  s.agents
                if a.health > INFECTIOUS_A
                    infections += 1
                end
            end
        end
        s.parameters.min_before_trace = 0
        for a in s.agents
            if a.test_res_iter == s.iteration &&
                    a.test_result == POSITIVE
                neighbours = Int64(round(s.parameters.k_assort.v / 2.0))
                from = max(1, a.id - neighbours)
                to = min(a.id + neighbours, length(s.agents))
                for i in from:to
                    if i != a.id && s.agents[i].isolated == 0.0 &&
                            s.agents[i].health < RECOVERED
                        if rand_0_1(s.rng) < s.parameters.trace_effective
                            s.num_traced += 1
                            isolate!(s, s.agents[i])
                        end
                    end
                end
            end
        end
    end

    function event_result!(s::Simulation)
        for a in s.agents
            if a.test_res_iter == s.iteration
                a.test_res_iter = nothing
            end
        end
    end

    function advance_infection!(rng::Random.AbstractRNG, a::Agent, 
                                stage_from::Health, stage_to::Health, 
                                prob::Float64, recover::Bool, iteration::Integer)
        if a.health == stage_from
            if rand_0_1(rng) < prob
                if recover
                    a.health = RECOVERED
                    append!(a.health_change_iters, RECOVERED, iteration)
                else
                    a.health = stage_to
                    append!(a.health_change_iters, stage_to, iteration)
                end
            end
        end
    end

    function event_exposed!(s::Simulation)
        for a in s.agents
            advance_infection!(s.rng, a, EXPOSED, INFECTIOUS_A, 
                            s.parameters.exposed_risk.v, false, s.iteration)
        end
    end

    function event_infectious_a!(s::Simulation)
        for a in s.agents
            advance_infection!(s.rng, a, INFECTIOUS_A, INFECTIOUS_S, 
                            s.parameters.exposed_risk.v, a.asymptomatic, s.iteration)
        end
    end

    function event_infectious_s!(s::Simulation)
        for a in s.agents
            advance_infection!(s.rng, a, INFECTIOUS_S, INFECTIOUS_H, 
                            s.parameters.exposed_risk.v, a.recover_before_hospital, s.iteration)
        end
    end

    function event_infectious_h!(s::Simulation)
        for a in s.agents
            advance_infection!(s.rng, a, INFECTIOUS_H, INFECTIOUS_I, 
                            s.parameters.exposed_risk.v, a.recover_before_icu, s.iteration)
        end
    end

    function event_infectious_i!(s::Simulation)
        for a in s.agents
            advance_infection!(s.rng, a, INFECTIOUS_I, DEAD, 
                            s.parameters.exposed_risk.v, a.recover_before_death, s.iteration)
        end
    end

    function iterate!(s::Simulation)
        final_iteration = s.iteration + s.parameters.num_iterations - 1
        for i in s.iteration:final_iteration
            s.iteration = i
            stats!(s)
            report(s)
            if s.parameters.k_assort.v > 0
                event_infect_assort!(s)
            end
            if s.parameters.k_unassort.v > 0
                event_infect_unassort!(s)
            end
            event_test!(s)
            if s.parameters.max_isolation > 0.0
                event_isolate!(s)
            end
            if s.parameters.max_isolation > 0.0
                event_deisolate!(s)
            end
            if s.parameters.trace_effective > 0.0
                event_trace!(s)
            end
            event_exposed!(s)
            event_infectious_a!(s)
        end
        stats!(s, forced=true)
        report(s, forced=true)
    end

    function simulate!(s::Simulation)
        init_agents!(s) 
        s.iteration = 0
        iterate!(s)
    end

    # Nathan has ~Simulation() that deletes the agents from a Simulation object

    function write_csv_header(io::IO=IOContext(stdout, :compact => false))
        # This lock may be unnecessary
        lock(io)
        println(
            io,
            "thread,id,scenario,jiggle,run,iteration,susceptible,exposed," * 
                "asymptomatic,symptomatic,hospital,icu,recover,dead,active," * 
                "total_infected,agents_isolated,isolated,deisolated,traced," * 
                "agents_tested,tested,positives,k,test_infectious_s,mean_test," *
                "min_test,isolation_period,exposed_risk,asymp_prob,inf_a_risk," *
                "inf_s_risk,min_isolation,max_isolation,trace_effective"
            )
        unlock(io)
    end

    # This probably has no purpose here
    function run_one_sim!(s::Simulation)
        Random.seed!(s.rng, s.parameters.seed)
        simulate!(s)
    end

    function set_jiggles!(rng::Random.AbstractRNG, p::Parameters)
        set!(rng, p.exposed_risk)
        set!(rng, p.asymptomatic)
        set!(rng, p.infectious_a_risk)
        set!(rng, p.k_assort)
        set!(rng, p.prob_test_susceptible)
        set!(rng, p.prob_test_exposed)
        set!(rng, p.prob_test_infectious_a)
        set!(rng, p.prob_test_infectious_s)
        set!(rng, p.prob_test_infectious_h)
        set!(rng, p.prob_test_infectious_i)
        set!(rng, p.prob_test_recovered)
        set!(rng, p.prob_test_dead)
        set!(rng, p.mean_test)
        set!(rng, p.min_test)
        set!(rng, p.isolation_period)
        set!(rng, p.infectious_s_risk)
    end

    function set_jiggles!(p::Parameters)
        set_jiggles!(Random.GLOBAL_RNG, p)
    end

    function get_jiggles!(from::Parameters, to::Parameters)
        get!(from.exposed_risk, to.exposed_risk)
        get!(from.asymptomatic, to.asymptomatic)
        get!(from.infectious_a_risk, to.infectious_a_risk)
        get!(from.k_assort, to.k_assort)
        get!(from.prob_test_susceptible, to.prob_test_susceptible)
        to.prob_test[SUSCEPTIBLE] = to.prob_test_susceptible.v
        get!(from.prob_test_exposed, to.prob_test_exposed)
        to.prob_test[EXPOSED] = to.prob_test_exposed.v
        get!(from.prob_test_infectious_a, to.prob_test_infectious_a)
        to.prob_test[INFECTIOUS_A] = to.prob_test_infectious_a.v
        get!(from.prob_test_infectious_s, to.prob_test_infectious_s)
        to.prob_test[INFECTIOUS_S] = to.prob_test_infectious_s.v
        get!(from.prob_test_infectious_s, to.prob_test_infectious_s)
        to.prob_test[INFECTIOUS_H] = to.prob_test_infectious_h.v
        get!(from.prob_test_infectious_i, to.prob_test_infectious_i)
        to.prob_test[INFECTIOUS_I] = to.prob_test_infectious_i.v
        get!(from.prob_test_recovered, to.prob_test_recovered)
        to.prob_test[RECOVERED] = to.prob_test_recovered.v
        get!(from.prob_test_dead, to.prob_test_dead)
        to.prob_test[DEAD] = to.prob_test_dead.v
        get!(from.mean_test, to.mean_test)
        get!(from.min_test, to.min_test)
        get!(from.isolation_period, to.isolation_period)
        get!(from.infectious_s_risk, to.infectious_s_risk)
    end

    function run_simulations(parms::AbstractVector{Parameters})
        blocklock = SpinLock()
        id = parms[1].first_id
        num_jiggles = parms[1].num_jiggles
        num_runs_per_jiggle = parms[1].num_runs_per_jiggle
        template_parms = parms[1]
        
        global_seed = template_parms.seed

        # I'm confused by the ordering. set_jiggles is run on the template_parms
        # so if you passsed a vector of parms for scenarios, the jiggling is not
        # happening for each scenario. What if you are varying the Jiggles across
        # scenarios?
        Threads.@threads for i in 1:num_jiggles
            p = deepcopy(template_parms)
            set_jiggles!(p)
            for j in 1:num_runs_per_jiggle
                for k in 1:length(parms)
                    # there may be a more elegant way to set thread-local seeds
                    if global_seed === nothing
                        local_seed = nothing
                    else
                        local_seed = global_seed + Threads.threadid()
                    end
                    local_parms = deepcopy(parms[k])
                    get_jiggles!(p, local_parms)
                    local_parms.jiggle = i
                    local_parms.run = j
                    local_parms.scenario = k
                    lock(blocklock) do
                        id += 1
                    end
                    local_parms.id = id
                    local_parms.seed = local_seed
                    s = Simulation(local_parms)
                    simulate!(s)
                end
            end
        end
    end

    function run_simulations(parms::Parameters)
        run_simulations([parms])
    end

end