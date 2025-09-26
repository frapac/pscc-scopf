
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
nK = 6
screen = readdlm("data/screening/$(case).txt")
contingencies = findall(screen[:, 4] .== 0)[1:nK]

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
# n_cc = length(ind_cc1)
# tol = 1e-6
# for k in 1:n_cc
#     x1, x2 = x[ind_cc1[k]], x[ind_cc2[k]]
#     if max(x1, x2) <= sqrt(tol)
#         x[ind_cc1[k]] = 0.0
#         x[ind_cc2[k]] = 0.0
#     elseif x1 <= tol
#         x[ind_cc1[k]] = 0.0
#     elseif x2 <= tol
#         x[ind_cc2[k]] = 0.0
#     end
# end

c = cons(nlp, x)

# for i=1:length(c)
#     if c[i]<nlp.meta.lcon[i] - 1e-8 || c[i]>nlp.meta.ucon[i] + 1e-8
#         println("$(i): $(nlp.meta.lcon[i]) <= $(c[i]) <= $(nlp.meta.ucon[i])")
#     end
# end

# calculate necessary tr.
inf_c = mapreduce((lc, c_, uc) -> max(c_-uc, lc-c_, 0), max, nlp.meta.lcon, c, nlp.meta.ucon; init=0.0)
inf_x = mapreduce((lx, x_, ux) -> max(x_-ux, lx-x_, 0), max, nlp.meta.lvar, x, nlp.meta.uvar; init=0.0)
inf_cc = mapreduce((x1, x2, lx1, lx2) -> max(min(x1-lx1, x2-lx2), 0), max,
                   x[ind_cc1], x[ind_cc2], nlp.meta.lvar[ind_cc1], nlp.meta.lvar[ind_cc2];
                   init = 0.0)
println("inf_cc: $(inf_cc), inf_c: $(inf_c), inf_x: $(inf_x)")

# inf factor
inf_factor = 10.0
rho_min = inf_factor*max(inf_x, inf_cc, inf_c)
println("rho_min = $(rho_min)")

mpecopt!(nlp, ind_cc1, ind_cc2, x; tol=1e-6, tr_radius=rho_min, max_iter=10)
