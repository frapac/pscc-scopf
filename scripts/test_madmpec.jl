
using DelimitedFiles
using MadNLP, MadNLPHSL
using NLPModels
using NLPModelsJuMP
using MadMPEC

function NLPModels.jac_structure!(nlp::MathOptNLPModel, rows::AbstractVector, cols::AbstractVector)
    NLPModels.jac_lin_structure!(nlp, rows, cols)
    NLPModels.jac_nln_structure!(nlp, rows, cols)
    return rows, cols
end

include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "data", "contingencies.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))

# Path to matpower should be specify in PATH
DATA_DIR = ENV["MATPOWER_DIR"]
# Matpower instance
case = "case118"
# Number of contingencies (increase as much as you want)
nK = 8
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
    load_factor=1.0,
    scale_obj=1e-3,
)

# Parse contingencies
ind_cc1, ind_cc2 = parse_ccons!(model)
# Build NLPModel with NLPModelsJuMP
nlp = MathOptNLPModel(model)
# Build MPCC problem
mpcc = MadMPEC.MPCCModelVarVar(nlp, ind_cc1, ind_cc2)

# Solve SCOPF with MadNLPC
# madnlpc_opts = MadMPEC.MadNLPCOptions(; print_level=MadNLP.INFO)
# solver = MadMPEC.MadNLPCSolver(mpcc; max_iter=500, solver_opts=madnlpc_opts, print_level=MadNLP.INFO, nlp_scaling=true)


opts = MadMPEC.HomotopySolverOptions()
opts.print_level = MadNLP.INFO
opts.nlp_solver_options =
    Dict(:bound_relax_factor=>1e-12, :print_level=>MadNLP.ERROR, :max_iter=>500)
solver = MadMPEC.HomotopySolver(mpcc, MadNLPSolver, opts)
stats = @time MadMPEC.solve!(solver)


# madnlpell1_opts = MadMPEC.ExactPenaltyOptions{Float64}(; print_level=MadNLP.ERROR)
# solver = MadMPEC.ExactPenaltySolver(
#     mpcc;
#     solver_opts=madnlpell1_opts,
#     print_level=MadNLP.ERROR,
# )
# stats = MadMPEC.solve_homotopy!(solver)
