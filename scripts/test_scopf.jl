
using Revise
using DelimitedFiles
using MadNLP, MadNLPHSL
using MadNCL
using ExaModels

include(joinpath(@__DIR__, "..", "data", "contingencies.jl"))
include(joinpath(@__DIR__, "..", "models", "scopf.jl"))

DATA_DIR = "/home/fpacaud/dev/matpower/data/"
# case = "case_ACTIVSg500"
case = "case118"
# contingencies = [2, 3, 5, 6, 8, 9] #collect(2:2)

nK = 100
# screen = readdlm("data/screening/$(case).txt")
# contingencies = findall(screen[:, 4] .== 0)[1:nK]
# contingencies = findall(screen[:, 3] .<= 1e-3)[1:nK]
contingencies = CONTINGENCIES[case][1:nK]

nK = length(contingencies)
model  = scopf_model(
   joinpath(DATA_DIR, "$(case).m"),
    contingencies;
    use_mpec=false,
    adjust=:mpecdroop,
    voltage_control=:pvpq,
    load_factor=1.0,
    scale_cc=1.0,
    # scale_obj=0.0,
)

@info "# contingencies: $(nK)"

nlp = ExaModels.ExaModel(model)
ncl_options = MadNCL.NCLOptions{Float64}(;
    opt_tol=1e-6,
    feas_tol=1e-6,
    slack_reset=false,
    scaling=false,
)

stats = @time MadNCL.madncl(
    nlp;
    print_level=MadNLP.ERROR,
    linear_solver=Ma57Solver,
    ma57_pivtol=0.0,
    # max_iter=10,
    nlp_scaling=false,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
    scaling=false,
    ncl_options=ncl_options,
    richardson_tol=1e-12,
)

