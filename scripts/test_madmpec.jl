
using DelimitedFiles
using MadNLP, MadNLPHSL
using MadNCL
using NLPModelsJuMP
using MadMPEC

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
)

# Parse contingencies
ind_cc1, ind_cc2 = parse_ccons!(model)
# Build NLPModel with NLPModelsJuMP
nlp = MathOptNLPModel(model)
# Build MPCC problem
mpcc = MadMPEC.MPCCModelVarVar(nlp, ind_cc1, ind_cc2)

# Solve SCOPF with MadNLPC
madnlpc_opts = MadMPEC.MadNLPCOptions(; print_level=MadNLP.INFO)
solver = MadMPEC.MadNLPCSolver(mpcc; madnlpc_opts=madnlpc_opts, print_level=MadNLP.INFO)
stats = MadMPEC.solve_homotopy!(solver)
