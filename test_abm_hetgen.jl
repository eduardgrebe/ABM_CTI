push!(LOAD_PATH, "/Users/eduard/dev/ABM_CTI/")
using Hetgen

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

par = set_parameters(
    default_parameters,
    num_jiggles = 12,
    num_runs_per_jiggle = 1,
    exposed_risk = Jiggle(0.15, 0.35),
    asymptomatic = Jiggle(0.65, 0.85),
    infectious_a_risk = Jiggle(0.4, 0.6),
    k_assort = Jiggle(28, 48),
    k_unassort = Jiggle(0, nothing),
    k_tracing = Jiggle(28, 48),
    pos_test_susceptible = Jiggle(POS_SUSCEPTIBLE, nothing),
    pos_test_exposed = Jiggle(POS_EXPOSED, nothing),
    pos_test_infectious_a = Jiggle(POS_INFECTIOUS_A, nothing),
    pos_test_infectious_s = Jiggle(POS_INFECTIOUS_S, nothing),
    pos_test_infectious_h = Jiggle(POS_INFECTIOUS_H, nothing),
    pos_test_infectious_i = Jiggle(POS_INFECTIOUS_I, nothing),
    pos_test_recovered = Jiggle(POS_RECOVERED, nothing),
    pos_test_dead = Jiggle(POS_DEAD, nothing),
    prob_test_susceptible = Jiggle(TEST_SUSCEPTIBLE, nothing),
    prob_test_exposed = Jiggle(TEST_EXPOSED, nothing),
    prob_test_infectious_a = Jiggle(TEST_INFECTIOUS_A, nothing),
    prob_test_infectious_s = Jiggle(TEST_INFECTIOUS_S, nothing),
    prob_test_infectious_h = Jiggle(TEST_INFECTIOUS_H, nothing),
    prob_test_infectious_i = Jiggle(TEST_INFECTIOUS_I, nothing),
    prob_test_recovered = Jiggle(TEST_RECOVERED, nothing),
    prob_test_dead = Jiggle(TEST_DEAD, nothing),
    mean_test = Jiggle(2, nothing),
    min_test = Jiggle(2, nothing),
    isolation_period = Jiggle(10, nothing),
    infectious_s_risk = Jiggle(0.1, 0.3)
    )

write_csv_header()
@time run_simulations(par)