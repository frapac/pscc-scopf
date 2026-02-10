
using Revise
using JuMP
using PowerModels
using PowerModelsSecurityConstrained
using MadNLP
using MadNCL
using MadNLPHSL
using ExaModels

include(joinpath(@__DIR__, "..", "models", "goc1.jl"))

dump_dir = "data/Original_Dataset_Offline_Edition_1/"
instance = "Network_10O-10/inputfiles.ini"
#

nK = 5
model = goc1_model(joinpath(dump_dir, instance), scenario, nK; tau_relax=1e-5, reg_l1=true)

nlp = ExaModels.ExaModel(model)

ncl_options = MadNCL.NCLOptions{Float64}(;
    opt_tol=1e-5,
    feas_tol=1e-5,
    scaling=true,
    scaling_max_gradient=100.0,
    rho_max=1e15,
    max_auglag_iter=20,
    # mu_min=1e-6,
)

madncl = MadNCL.NCLSolver(
    nlp;
    print_level=MadNLP.ERROR,
    linear_solver=Ma57Solver,
    # ma57_pivtol=0.0,
    # ma57_automatic_scaling=true,
    # ma57_pivot_order=5,
    max_iter=1000,
    nlp_scaling=false,
    # bound_push=1.0,
    richardson_tol=1e-12,
    bound_relax_factor=1e-6,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
    ncl_options=ncl_options,
)

stats = @time MadNCL.solve!(madncl)
