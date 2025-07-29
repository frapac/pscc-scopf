
using DelimitedFiles
using MadNLP
using MadNLPGPU
using MadNCL
using ExaModels
using CUDA

include(joinpath(@__DIR__, "..", "models", "scopf.jl"))

GC.gc(true)
CUDA.reclaim()
GC.gc(true)


DATA_DIR = "/home/fpacaud/dev/matpower/data/"
# case = "case_ACTIVSg2000"
case = "case2869pegase"
screen = readdlm("data/screening/$(case).txt")
# contingencies = findall(screen[:, 4] .== 0)[11:18]
# contingencies = findall(screen[:, 4] .<= 5)[1:16]
contingencies = findall(screen[:, 3] .<= 1e-4)[1:8]
# contingencies = collect(1:8)
# contingencies = [ 2, 3, 5, 8, 9, 10, 12, 15]

nK = length(contingencies)
model  = scopf_model(
    joinpath(DATA_DIR, "$(case).m"),
    contingencies;
    adjust=:mpecdroop,
    voltage_control=:pvpq,
    load_factor=1.0,
    scale_cc=1e-3,
)

@info "# contingencies: $(nK)"

nlp = ExaModels.ExaModel(model; backend=CUDABackend())

ncl_options = MadNCL.NCLOptions(;
    opt_tol=1e-5,
    feas_tol=1e-5,
    slack_reset=false,
)

res = @time MadNCL.madncl(
    nlp;
    print_level=MadNLP.ERROR,
    linear_solver=MadNLPGPU.CUDSSSolver,
    max_iter=1000,
    nlp_scaling=false,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
    cudss_algorithm=MadNLP.LDL,
    cudss_pivot_epsilon=1e-10,
    scaling=false,
    ncl_options=ncl_options,
)

