
#=
    Script for contingency screening.

    Check if a given contingency is feasible w.r.t. the base case solution.
    The constraints model the PV/PQ switches and the droop control with
    complementarity constraints.

=#

using Printf
using LinearAlgebra
using DelimitedFiles
using JuMP
using PowerModels
using MadNLP, MadNLPHSL
using MadNCL
using NLPModels
using ExaModels

PowerModels.silence()

include(joinpath(@__DIR__, "..", "models", "opf.jl"))
include(joinpath(@__DIR__, "..", "models", "contingency.jl"))

function extract_solution(model)
    return (
        va=JuMP.value.(model[:va]),
        vm=JuMP.value.(model[:vm]),
        pg=JuMP.value.(model[:pg]),
        qg=JuMP.value.(model[:qg]),
        p=JuMP.value.(model[:p]),
        q=JuMP.value.(model[:q]),
    )
end

function screen_contingency(case; tol=1e-6, max_cont=Inf)
    # Base case
    model  = opf_model(case)
    JuMP.set_optimizer(model, MadNLP.Optimizer)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.set_attribute(model, "max_iter", 500)
    JuMP.set_attribute(model, "linear_solver", Ma27Solver)
    JuMP.optimize!(model)
    base_case = extract_solution(model)

    # Contingency
    nl = div(length(model[:q]), 2)
    nK = isfinite(max_cont) ? min(nl, max_cont) : nl
    screen = zeros(nK, 4)
    @info "Screen contingency..."
    for k in 1:nK
        @printf "\rScreen contingency: %d / %d " k nK
        cnt_model = contingency_problem(case, base_case, k)

        nlp = ExaModels.ExaModel(cnt_model)
        snlp = MadNCL.ScaledModel(nlp)
        ncl = MadNCL.NCLModel(snlp)
        ncl.yk .= 0
        ncl.ρk[] = 1e6
        res = madnlp(
            ncl;
            print_level=MadNLP.ERROR,
            # linear_solver=MadNLPGPU.CUDSSSolver,
            linear_solver=Ma57Solver,
            # cholmod_algorithm=MadNLP.LDL,
            max_iter=200,
            tol=1e-8,
            kkt_system=MadNCL.K2rAuglagKKTSystem,
        )

        nvar = NLPModels.get_nvar(nlp)
        ncon = NLPModels.get_ncon(nlp)

        r = res.solution[nvar+1:nvar+ncon]
        screen[k, 1] = Int(res.status)
        screen[k, 2] = res.objective
        screen[k, 3] = norm(r, Inf)
        screen[k, 4] = length(findall(abs.(r) .> tol))
    end
    return screen
end

function demo()
    case = "../matpower/data/case9.m"

    model  = opf_model(case)
    JuMP.set_optimizer(model, MadNLP.Optimizer)
    JuMP.set_attribute(model, "linear_solver", Ma27Solver)
    JuMP.optimize!(model)
    base_case = extract_solution(model)

    index_contingency = 9
    cnt_model = contingency_problem(case, base_case, index_contingency)

    nlp = ExaModels.ExaModel(cnt_model)
    snlp = MadNCL.ScaledModel(nlp)
    ncl = MadNCL.NCLModel(snlp)
    ncl.yk .= 0
    ncl.ρk[] = 1e6
    res = madnlp(
        ncl;
        print_level=MadNLP.INFO,
        linear_solver=Ma27Solver,
        max_iter=200,
        tol=1e-8,
        kkt_system=MadNCL.K2rAuglagKKTSystem,
    )
end

DATA_DIR = "/home/fpacaud/dev/matpower/data"

case = "case118"
instance = joinpath(DATA_DIR, "$(case).m")
screen = @time screen_contingency(instance; max_cont=200)

