
using DelimitedFiles
using JuMP
using KNITRO

include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))
include(joinpath(@__DIR__, "common.jl"))


function benchmark_knitro()
    n = length(CASES)

    results = zeros(n, 8)
    k = 1

    for (case, nK) in CASES
        println((case, nK))
        contingencies = CONTINGENCIES[case][1:nK]

        model = scopf_model(
            joinpath(DATA_DIR, "$(case).m"),
            contingencies;
            adjust=:mpecdroop,
            voltage_control=:pvpq,
            load_factor=1.0,
            scale_cc=1e-4,
        )
        ind_cc1, ind_cc2 = parse_ccons!(model; reformulation=:mpec)
        JuMP.set_optimizer(model, KNITRO.Optimizer)
        JuMP.set_optimizer_attribute(model, "maxit", 1000)
        JuMP.optimize!(model)

        results[k, 1] = JuMP.num_variables(model)
        results[k, 2] = 0
        results[k, 3] = Int(JuMP.is_solved_and_feasible(model))
        results[k, 4] = MOI.get(model, MOI.BarrierIterations())
        results[k, 5] = JuMP.objective_value(model)
        results[k, 6] = 0.0
        results[k, 7] = 0.0
        results[k, 8] = JuMP.solve_time(model)

        k += 1
    end
    names = [case for (case, _) in CASES]
    return [names results]
end

results = benchmark_knitro()
writedlm(joinpath(@__DIR__, "..", "results", "scopf-knitro.csv"), results)


