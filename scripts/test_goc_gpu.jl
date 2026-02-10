

using Revise
using MadNLP
using MadNCL
using ExaModels
using CUDA
using MadNLPGPU

include(joinpath(@__DIR__, "..", "models", "goc1.jl"))

dump_dir = "data/Original_Dataset_Offline_Edition_1/"
instance = "Network_10O-10/inputfiles.ini"

nK = 10
model = goc1_model(joinpath(dump_dir, instance), scenario, nK; tau_relax=1e-5, reg_l1=false)

nlp = ExaModels.ExaModel(model; backend=CUDABackend())

ncl_options = MadNCL.NCLOptions{Float64}(;
    opt_tol=1e-5,
    feas_tol=1e-5,
    scaling=true,
    scaling_max_gradient=100.0,
    rho_max=1e15,
    max_auglag_iter=20,
)

madncl = MadNCL.NCLSolver(
    nlp;
    print_level=MadNLP.ERROR,
    max_iter=2000,
    nlp_scaling=false,
    richardson_tol=1e-12,
    bound_relax_factor=1e-7,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
    ncl_options=ncl_options,
    linear_solver=MadNLPGPU.CUDSSSolver,
    cudss_pivot_epsilon=1e-10,
    cudss_algorithm=MadNLP.LDL,
)

stats = @time MadNCL.solve!(madncl)
