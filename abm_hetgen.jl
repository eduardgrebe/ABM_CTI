#using Random
#using Distributions
#import Base: rand
using Random
using Distributions
import Base: rand

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

# Performance advantage to making immutable?
mutable struct Jiggle{T<:Union{Integer,AbstractFloat}}
    l::T
    u::Union{T, Nothing}
end

function rand(rng::Random.AbstractRNG, s::Jiggle{<:Integer}, n::Integer = 1)
    if s.u == nothing || s.u == s.l
        r = repeat([s.l],n)
    elseif s.u < s.l
        throw(DomainError(s.u, "must be ≥ " * string(s.l)))
    else
        range = s.l:s.u
        r = rand(rng, range, n)
    end
    return r
end

function rand(rng::Random.AbstractRNG, s::Jiggle{<:AbstractFloat}, n::Integer = 1)
    if s.u == nothing || s.u == s.l
        r = repeat([s.l],n)
    elseif s.u < s.l
        throw(DomainError(s.u, "must be ≥ " * string(s.l)))
    else
        d = Distributions.Uniform(s.l, s.u)
        r = rand(rng, d, n)
    end
    return r
end

mutable struct RiskInfecting{Float64}
    INFECTIOUS_A::Float64
    INFECTIOUS_S::Float64
    INFECTIOUS_H::Float64
    INFECTIOUS_I::Float64
end

# Make immutable for performance?
mutable struct Parameters
    first_id::Int64
    scenario::Int64
    jiggle::Int64
    run::Int64
    id::Int64
    num_jiggles::Int64
    num_runs_per_jiggle::Int64
    seed::Union{Int64, Nothing}
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

const default_parameters = Parameters(0,
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
                                    POS_DEAD
                                ],
                                [
                                    TEST_SUSCEPTIBLE,
                                    TEST_EXPOSED,
                                    TEST_INFECTIOUS_A,
                                    TEST_INFECTIOUS_S,
                                    TEST_INFECTIOUS_H,
                                    TEST_INFECTIOUS_I,
                                    TEST_RECOVERED,
                                    TEST_DEAD
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
                                Jiggle(0.2, nothing)

    )

function set_parameters(parameters; kwargs...)
    p = deepcopy(parameters)
    kwargs_dict = Dict(kwargs)
    for (key, value) in kwargs_dict
        setfield!(p, key, value)
    end
    return p
end

function set_parameters!(parameters; kwargs...)
    kwargs_dict = Dict(kwargs)
    for (key, value) in kwargs_dict
        setfield!(parameters, key, value)
    end
end


mutable struct Agent
    id::Int64
    risk_infection::Float64
    risk_infecting::Dict
    infector::Pair
    infected_by_me::Pair
    health_change_iters::Dict
    tested::Int
    test_result::TestResult
    test_res_iter::Int
    isolation_iter::Int
    isolated::Float64
    health::Health
    asymptomatic::Bool
    recover_before_hospital::Bool
    recover_before_icu::Bool
    recover_before_death::Bool
    function Agent(p::Parameters, id::Int)
        new(id, rand(Distributions.Exponential(1.0 / p.risk_infection)), p.risk_infecting)
    end
end



# Test code

p1 = default_parameters
p2 = set_parameters(p1, first_id = 10)
set_parameters!(p2, first_id = 100)

rng = MersenneTwister(1234)
j = Jiggle(10,nothing)
rand(rng, j, 10)
j = Jiggle(10,10)
rand(rng, j, 10)
j = Jiggle(10,20)
rand(rng, j, 10)
j = Jiggle(10,9)
rand(rng, j, 10)

j = Jiggle(10.5,nothing)
rand(rng, j, 10)
j = Jiggle(10.5,10.5)
rand(rng, j, 10)
j = Jiggle(10.5,20.5)
rand(rng, j, 10)
j = Jiggle(10.5,9.5)
rand(rng, j, 10)
