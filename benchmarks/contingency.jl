
using DelimitedFiles
using MadNLP, MadNLPHSL
using JuMP
using KNITRO
using NLPModels
using ExaModels
using MadNCL

include(joinpath(@__DIR__, "..", "models", "opf.jl"))
include(joinpath(@__DIR__, "..", "models", "contingency.jl"))
include(joinpath(@__DIR__, "common.jl"))

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

function screen_madncl(cases, nK; tol=1e-6, max_cont=Inf)
    n = length(cases)
    results = zeros(n, 8)

    # Contingency
    for (i, case) in enumerate(cases)
        @info case
        instance = joinpath(DATA_DIR, "$(case).m")
        # Base case
        model  = opf_model(instance)
        JuMP.set_optimizer(model, MadNLP.Optimizer)
        JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
        JuMP.set_attribute(model, "max_iter", 500)
        JuMP.set_attribute(model, "linear_solver", Ma27Solver)
        JuMP.optimize!(model)
        base_case = extract_solution(model)

        # Scan nK first contingencies
        for k in 1:nK
            cnt_model = contingency_problem(instance, base_case, k)

            nlp = ExaModels.ExaModel(cnt_model)
            snlp = MadNCL.ScaledModel(nlp)
            ncl = MadNCL.NCLModel(snlp)
            ncl.yk .= 0
            ncl.ρk[] = 1e6

            res = madnlp(
                ncl;
                print_level=MadNLP.ERROR,
                linear_solver=Ma57Solver,
                max_iter=200,
                tol=1e-8,
                kkt_system=MadNCL.K2rAuglagKKTSystem,
            )

            nvar = NLPModels.get_nvar(nlp)
            ncon = NLPModels.get_ncon(nlp)

            results[i, 1] = nvar
            results[i, 2] = ncon
            results[i, 3] += Int(res.status)
            results[i, 4] += res.iter
            results[i, 5] += res.objective
            results[i, 6] += res.counters.eval_function_time
            results[i, 7] += res.counters.linear_solver_time
            results[i, 8] += res.counters.total_time
        end
    end
    return [cases results]
end

function screen_knitro(cases, nK; tol=1e-6, max_cont=Inf)
    n = length(cases)
    results = zeros(n, 8)

    # Contingency
    for (i, case) in enumerate(cases)
        @info case
        instance = joinpath(DATA_DIR, "$(case).m")
        # Base case
        model  = opf_model(instance)
        JuMP.set_optimizer(model, KNITRO.Optimizer)
        # JuMP.set_attribute(model, "outlev", 0)
        JuMP.optimize!(model)
        base_case = extract_solution(model)

        # Scan nK first contingencies
        for k in 1:nK
            cnt_model = contingency_problem(instance, base_case, k; use_mpec=true)
            JuMP.set_optimizer(cnt_model, KNITRO.Optimizer)
            JuMP.set_optimizer_attribute(cnt_model, "maxit", 1000)
            JuMP.optimize!(cnt_model)

            results[i, 1] = JuMP.num_variables(cnt_model)
            results[i, 2] = 0
            results[i, 3] += JuMP.is_solved_and_feasible(cnt_model)
            results[i, 4] += MOI.get(cnt_model, MOI.BarrierIterations())
            results[i, 5] += JuMP.objective_value(cnt_model)
            results[i, 8] += JuMP.solve_time(cnt_model)
        end
    end
    return [cases results]
end

cases = [
    "case118",
    "case300",
    "case_ACTIVSg200",
    "case_ACTIVSg500",
    "case1354pegase",
    "case_ACTIVSg2000",
    "case2869pegase",
]
nK = 10
results = screen_knitro(cases, nK)
