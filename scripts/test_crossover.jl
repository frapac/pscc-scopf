
using JuMP
using DelimitedFiles
using MadNLP, MadNLPHSL
using MadNCL
using ExaModels

###
# Config
###
include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))
include(joinpath(@__DIR__, "..", "scripts", "crossover.jl"))
DATA_DIR = ENV["MATPOWER_DIR"]
case = "case14"
contingencies = [3, 4, 5, 8, 9, 11, 12, 13]

###
# Step 1: run phase I with MadNCL
###

# Build JuMP model
model  = scopf_model(
   joinpath(DATA_DIR, "$(case).m"),
    contingencies;
    adjust=:mpecdroop,
    voltage_control=:pvpq,
    load_factor=1.0,
)
# Reformulate complementarity constraints inside JuMP
ind_cc1, ind_cc2 = parse_ccons!(model; reformulation=:scholtes)
# Convert to ExaModels for fast evaluation
nlp = ExaModels.ExaModel(model)

# Solve with MadNCL
ncl_options = MadNCL.NCLOptions{Float64}(;
    opt_tol=1e-6,
    feas_tol=1e-6,
    slack_reset=false,
    scaling=true,
    scaling_max_gradient=100.0,
    extrapolation=true,
)
stats = @time MadNCL.madncl(
    nlp;
    print_level=MadNLP.ERROR,
    linear_solver=Ma57Solver,
    ma57_pivtol=0.0,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
    ncl_options=ncl_options,
    richardson_tol=1e-12,
    richardson_max_iter=20,
    max_iter=200,
)

###
# Step 2: run crossover
###

x = copy(stats.solution)

# Project solution onto feasible set
n_cc = length(ind_cc1)
tol = 1e-6
for k in 1:n_cc
    x1, x2 = x[ind_cc1[k]], x[ind_cc2[k]]
    if max(x1, x2) <= sqrt(tol)
        x[ind_cc1[k]] = 0.0
        x[ind_cc2[k]] = 0.0
    elseif x1 <= tol
        x[ind_cc1[k]] = 0.0
    elseif x2 <= tol
        x[ind_cc2[k]] = 0.0
    end
end

mpecopt!(nlp, ind_cc1, ind_cc2, x; tol=1e-5, tr_radius=1e-2)
