using Random
using Distributions
import Base: rand
using IterTools
using Match

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

# Convenience function
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

mutable struct HealthChangeIters
    stage::Health
    iteration::AbstractVector{Int64}
end

mutable struct RiskInfecting{T<:AbstractFloat}
    INFECTIOUS_A::T
    INFECTIOUS_S::T
    INFECTIOUS_H::T
    INFECTIOUS_I::T
end

function RiskInfecting(x::AbstractVector{<:AbstractFloat})
    RiskInfecting(x[1], x[2], x[3], x[4])
end

# Performance advantage to making immutable?
mutable struct Jiggle{T<:Union{Integer,AbstractFloat}}
    l::T
    u::Union{T,Nothing}
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
    risk_infecting::RiskInfecting
    risk_positive::AbstractVector{Float64}
    prob_test::AbstractVector{Float64}
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
    RiskInfecting(0.05, 0.1, 0.1, 0.1),
    [
        POS_SUSCEPTIBLE,
        POS_EXPOSED,
        POS_INFECTIOUS_A,
        POS_INFECTIOUS_S,
        POS_INFECTIOUS_H,
        POS_INFECTIOUS_I,
        POS_RECOVERED,
        POS_DEAD,
    ],
    [
        TEST_SUSCEPTIBLE,
        TEST_EXPOSED,
        TEST_INFECTIOUS_A,
        TEST_INFECTIOUS_S,
        TEST_INFECTIOUS_H,
        TEST_INFECTIOUS_I,
        TEST_RECOVERED,
        TEST_DEAD,
    ],
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

function set_parameters(parameters; kwargs...)
    p = deepcopy(parameters)
    for (key, value) in Dict(kwargs)
        setfield!(p, key, value)
    end
    return p
end

function set_parameters!(parameters; kwargs...)
    for (key, value) in Dict(kwargs)
        setfield!(parameters, key, value)
    end
end

mutable struct Agent
    id::Int64
    risk_infection::Float64
    risk_infecting::RiskInfecting
    infector::Infection
    infected_by_me::AbstractVector{Infection}
    health_change_iters::Union{HealthChangeIters,Nothing}
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
        asymptomatic_ = rand_0_1(rng) < p.asymptomatic.l
        if !asymptomatic_
            recover_before_hospital_ = rand_0_1(rng) < p.recover_before_hospital
            if !recover_before_hospital_
                recover_before_icu_ = rand_0_1(rng) < p.recover_before_icu
                if !recover_before_icu_
                    recover_before_death_ = rand_0_1(rng) < p.recover_before_death
                end
            end
        end
        new(
            id,
            rand(rng, Distributions.Exponential(p.risk_infection)),
            RiskInfecting([
                rand(rng, Distributions.Exponential(r))
                for r in fieldvalues(p.risk_infecting)
            ]),
            Infection(nothing,nothing),
            Vector{Infection}(),
            nothing,
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
    a.isolation_iter === nothing && s.num_agents_isolated += 1
    # again .l is equivalent to () operator on Jiggle in C++ code
    a.isolation_iter = s.iteration + s.parameters.isolation_period.l
    # I don't fully understand this bit
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

function init_agents!(s::Simulation; force=false)
    if !force && length(s.agents) != 0
        error("Simulation object already contains agents. Use force=true to override")
    elseif force
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
        infect_indices = sample(s.rng, 1:length(s.agents), s.parameters.initial_infections, replace=false, ordered=true)
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

# Where do we output the report?
function report(s::Simulation; forced = false)
    if forced || s.iteration % s.parameters.report_frequency == 0
        susceptible = 0
        exposed = 0
        infectious_a = 0
        infectious_s = 0
        infectious_h = 0
        infectious_i = 0
        recovered = 0
        dead = 0
        for a in s.agents
            @match a.health begin
                SUSCEPTIBLE  => susceptible += 1
                EXPOSED      => exposed += 1
                INFECTIOUS_A => infectious_a += 1
                INFECTIOUS_S => infectious_s += 1
                INFECTIOUS_H => infectious_h += 1
                INFECTIOUS_I => infectious_i += 1
                RECOVERED    => recovered += 1
                DEAD         => dead += 1
             end
        end
        active = exposed + infectious_a + infectious_s + infectious_h + 
            infectious_i
        
        # Write out the report (use CSV.jl? write plain text?)

    end
end




###*** Test code ***###

par = default_parameters
s = Simulation(par)
init_agents!(s)
infections = 0
for a in s.agents
    infections += (a.health > SUSCEPTIBLE && a.health < RECOVERED)
end
print(infections)

set_parameters!(par, initial_infections=10)
s = Simulation(par)
init_agents!(s)
infections = 0
which_infected = Vector{Int64}()
for a in s.agents
    infections += (a.health > SUSCEPTIBLE && a.health < RECOVERED)
    if a.health > SUSCEPTIBLE && a.health < RECOVERED
        append!(which_infected, a.id)
    end
end
print(infections)
print(which_infected)



rng = MersenneTwister(1234)
p1 = default_parameters
p2 = set_parameters(p1, first_id = 10)
set_parameters!(p2, first_id = 1)

j = Jiggle(10, nothing)
rand(rng, j, 10)
j = Jiggle(10, 10)
rand(rng, j, 10)
j = Jiggle(10, 20)
rand(rng, j, 10)
j = Jiggle(10, 9)
rand(rng, j, 10)

j = Jiggle(10.5, nothing)
rand(rng, j, 10)
j = Jiggle(10.5, 10.5)
rand(rng, j, 10)
j = Jiggle(10.5, 20.5)
rand(rng, j, 10)
j = Jiggle(10.5, 9.5)
rand(rng, j, 10)

agent = Agent(p1, 0, rng)
