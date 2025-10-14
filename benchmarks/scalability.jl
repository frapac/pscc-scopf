
using DelimitedFiles
using JuMP
using KNITRO

using NLPModels
using MadNLP, MadNLPHSL
using MadNCL
using ExaModels

using MadNLPGPU
using CUDA

include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))
include(joinpath(@__DIR__, "common.jl"))

function refresh()
    GC.gc(true)
    CUDA.reclaim()
    GC.gc(true)
end

function benchmark_knitro(case, nscenarios)
    n = length(nscenarios)

    results = zeros(n, 8)

    # parse contingencies
    screen = readdlm("data/screening/$(case).txt")
    ind = sortperm(screen[:, 2])

    k = 1
    for nK in nscenarios
        println((case, nK))
        contingencies = ind[1:nK]

        model = scopf_model(
            joinpath(DATA_DIR, "$(case).m"),
            contingencies;
            adjust=:mpecdroop,
            voltage_control=:pvpq,
            use_mpec=true,
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
    return [nscenarios results]
end

function benchmark_madncl(case, nscenarios)
    n = length(nscenarios)

    results = zeros(n, 8)

    # parse contingencies
    screen = readdlm("data/screening/$(case).txt")
    ind = sortperm(screen[:, 2])

    k = 1
    for nK in nscenarios
        println((case, nK))
        contingencies = ind[1:nK]

        model = scopf_model(
            joinpath(DATA_DIR, "$(case).m"),
            contingencies;
            adjust=:mpecdroop,
            voltage_control=:pvpq,
            use_mpec=false,
            scale_cc=1.0,
            tau_relax=1e-5,
        )
        nlp = ExaModels.ExaModel(model)
        ncl_options = MadNCL.NCLOptions{Float64}(;
            opt_tol=1e-6,
            feas_tol=1e-6,
            scaling=true,
            scaling_max_gradient=100.0,
            extrapolation=true,
            mu_min=1e-7,
            max_auglag_iter=40,
        )

        res = @time MadNCL.madncl(
            nlp;
            ncl_options=ncl_options,
            linear_solver=Ma57Solver,
            richardson_tol=1e-12,
            richardson_max_iter=20,
            max_iter=1000,
            kkt_system=MadNCL.K2rAuglagKKTSystem,
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
    return [nscenarios results]
end

function benchmark_madncl_gpu(case, nscenarios)
    n = length(nscenarios)

    results = zeros(n, 8)

    # parse contingencies
    screen = readdlm("data/screening/$(case).txt")
    ind = sortperm(screen[:, 2])

    k = 1
    for nK in nscenarios
        println((case, nK))
        refresh()
        contingencies = ind[1:nK]

        model = scopf_model(
            joinpath(DATA_DIR, "$(case).m"),
            contingencies;
            adjust=:mpecdroop,
            voltage_control=:pvpq,
            use_mpec=false,
            scale_cc=1.0,
            tau_relax=1e-5,
        )

        nlp = ExaModels.ExaModel(model; backend=CUDABackend())
        ncl_options = MadNCL.NCLOptions{Float64}(;
            opt_tol=1e-6,
            feas_tol=1e-6,
            scaling=true,
            scaling_max_gradient=100.0,
            extrapolation=true,
            mu_min=1e-7,
            max_auglag_iter=40,
        )

        # Warm-up
        MadNCL.madncl(nlp; linear_solver=MadNLPGPU.CUDSSSolver, max_iter=1, kkt_system=MadNCL.K2rAuglagKKTSystem)

        res = @time MadNCL.madncl(
            nlp;
            ncl_options=ncl_options,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            cudss_pivot_epsilon=1e-10,
            richardson_tol=1e-12,
            richardson_max_iter=20,
            max_iter=1000,
            kkt_system=MadNCL.K2rAuglagKKTSystem,
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
    return [nscenarios results]
end

case = "case_ACTIVSg500"
nscens = [2^i for i in 1:8]
results = benchmark_madncl(case, nscens)
writedlm(joinpath(@__DIR__, "..", "results", "scopf-madncl-k2r-ma57.csv"), results)


