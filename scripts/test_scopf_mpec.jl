
using Revise
using DelimitedFiles
using MadNLP, MadNLPHSL
using MadNCL
using ExaModels

include(joinpath(@__DIR__, "..", "data", "contingencies.jl"))
include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))

DATA_DIR = ENV["MATPOWER_DIR"]
# case = "case_ACTIVSg500"
case = "case14"
# contingencies = [2, 3, 5, 6, 8, 9] #collect(2:2)

nK = 8
screen = readdlm("data/screening/$(case).txt")
contingencies = findall(screen[:, 4] .== 0)[1:nK]
# contingencies = findall(screen[:, 3] .<= 1e-3)[1:nK]
# contingencies = CONTINGENCIES[case][1:nK]

nK = length(contingencies)
model  = scopf_model(
   joinpath(DATA_DIR, "$(case).m"),
    contingencies;
    use_mpec=true,
    adjust=:mpecdroop,
    voltage_control=:pvpq,
    load_factor=1.0,
    # scale_obj=0.0,
)

@info "# contingencies: $(nK)"

ind_cc1, ind_cc2 = parse_ccons!(model; reformulation=:scholtes)
# nlp = MathOptNLPModel(model)
nlp = ExaModels.ExaModel(model)

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
    # max_iter=10,
    nlp_scaling=false,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
    # ma57_automatic_scaling=true,
    ncl_options=ncl_options,
    richardson_tol=1e-12,
    richardson_max_iter=20,
    max_iter=200,
)

x1 = stats.solution[ind_cc1]
x2 = stats.solution[ind_cc2]
ind00 = findall(max.(x1, x2) .<= 1e-4)

