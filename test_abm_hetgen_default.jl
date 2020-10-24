push!(LOAD_PATH, "./")
using Hetgen

par = default_parameters
write_csv_header()
run_simulations(par)