
using DelimitedFiles
using NLPModels
using MadNLP, MadNLPHSL
using MadNCL
using ExaModels

include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))
include(joinpath(@__DIR__, "common.jl"))

function benchmark_scopf(; options...)
    n = length(CASES)

    results = zeros(n, 8)
    k = 1

    for (case, nK) in CASES
        println((case, nK))
        contingencies = CONTINGENCIES[case][1:nK]

        model  = scopf_model(
        joinpath(DATA_DIR, "$(case).m"),
            contingencies;
            use_mpec=false,
            adjust=:mpecdroop,
            voltage_control=:pvpq,
            load_factor=1.0,
            scale_cc=1.0,
            tau_relax=1e-5,
        )

        nlp = ExaModels.ExaModel(model)

        ncl_options = MadNCL.NCLOptions{Float64}(;
            opt_tol=1e-5,
            feas_tol=1e-5,
            slack_reset=false,
            scaling=true,
            scaling_max_gradient=100.0,
            extrapolation=true,
        )

        res = @time MadNCL.madncl(
            nlp;
            ncl_options=ncl_options,
            options...
        )

        results[k, 1] = NLPModels.get_nvar(nlp)
        results[k, 2] = NLPModels.get_ncon(nlp)
        results[k, 3] = Int(res.status)
        results[k, 4] = res.iter
        results[k, 5] = res.objective
        results[k, 6] = res.counters.eval_function_time
        results[k, 7] = res.counters.linear_solver_time
        results[k, 8] = res.counters.total_time

        k += 1
    end
    names = [case for (case, _) in CASES]
    return [names results]
end

results = benchmark_scopf(;
    print_level=MadNLP.ERROR,
    linear_solver=Ma57Solver,
    richardson_tol=1e-12,
    richardson_max_iter=20,
    max_iter=1000,
    kkt_system=MadNCL.K2rAuglagKKTSystem,
)
writedlm(joinpath(@__DIR__, "..", "results", "scopf-madncl-k2r-ma57.csv"), results)


