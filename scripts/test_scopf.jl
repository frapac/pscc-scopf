
using DelimitedFiles
using MadNLP, MadNLPHSL
using MadNCL
using ExaModels

include(joinpath(@__DIR__, "..", "models", "scopf.jl"))

DATA_DIR = "/home/fpacaud/dev/matpower/data/"
case = "case1354pegase"
# contingencies = [2, 3, 5, 6, 8, 9] #collect(2:2)

screen = readdlm("data/screening/$(case).txt")
contingencies = findall(screen[:, 4] .== 0)[1:8]
# contingencies = findall(screen[:, 4] .<= 10)[1:8]

nK = length(contingencies)
model  = scopf_model(
   joinpath(DATA_DIR, "$(case).m"),
    contingencies;
    adjust=:mpecdroop,
    voltage_control=:pvpq,
    load_factor=1.0,
)

@info "# contingencies: $(nK)"

nlp = ExaModels.ExaModel(model)

ncl_options = MadNCL.NCLOptions(;
    opt_tol=1e-5,
    feas_tol=1e-5,
    slack_reset=false,
)

res = @time MadNCL.madncl(
    nlp;
    print_level=MadNLP.ERROR,
    linear_solver=Ma27Solver,
    max_iter=1000,
    nlp_scaling=false,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
    scaling=false,
    ncl_options=ncl_options,
)

