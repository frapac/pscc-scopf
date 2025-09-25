
using DelimitedFiles
using KNITRO

include(joinpath(@__DIR__, "..", "models", "contingency.jl"))
include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "data", "contingencies.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))

# Path to matpower should be specify in PATH
DATA_DIR = ENV["MATPOWER_DIR"]
# Matpower instance
case = "case_ACTIVSg2000"
# Number of contingencies (increase as much as you want)
nK = 16
# Load (line) contingencies
contingencies = CONTINGENCIES[case][1:nK]

nK = length(contingencies)

# Build JuMP model
model  = scopf_model(
   joinpath(DATA_DIR, "$(case).m"),
    contingencies;
    use_mpec=true,
    adjust=:mpecdroop,
    voltage_control=:pvpq,
    load_factor=1.00,
)

ind_cc1, ind_cc2 = parse_ccons!(model; reformulation=:mpec)

JuMP.set_optimizer(model, KNITRO.Optimizer)
JuMP.optimize!(model)

