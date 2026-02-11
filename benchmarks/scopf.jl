
using DelimitedFiles
using JuMP
using NLPModels
using MadNLP, MadNLPHSL
using MadNLPGPU
using MadNCL
using ExaModels
using KNITRO

include(joinpath(@__DIR__, "..", "models", "scopf.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))
include(joinpath(@__DIR__, "common.jl"))

function refresh()
    GC.gc(true)
    CUDA.reclaim()
    GC.gc(true)
end

function benchmark_knitro()
    n = length(CASES)

    results = zeros(n, 9)
    k = 1

    for (case, nK) in CASES
        println((case, nK))
        contingencies = CONTINGENCIES[case][1:nK]

        model = scopf_model(
            joinpath(DATA_DIR, "$(case).m"),
            contingencies;
        )
        ind_cc1, ind_cc2 = parse_ccons!(model; reformulation=:mpec)
        JuMP.set_optimizer(model, KNITRO.Optimizer)
        JuMP.set_optimizer_attribute(model, "maxit", 1000)
        JuMP.optimize!(model)

        results[k, 1] = nK
        results[k, 2] = JuMP.num_variables(model)
        results[k, 3] = 0
        results[k, 4] = Int(JuMP.is_solved_and_feasible(model))
        results[k, 5] = MOI.get(model, MOI.BarrierIterations())
        results[k, 6] = JuMP.objective_value(model)
        results[k, 7] = 0.0
        results[k, 8] = 0.0
        results[k, 9] = JuMP.solve_time(model)

        k += 1
    end
    names = [case for (case, _) in CASES]
    return [names results]
end

function benchmark_madncl_cpu(; options...)
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

        nlp = ExaModels.ExaModel(model; backend=CUDABackend())

        ncl_options = MadNCL.NCLOptions{Float64}(;
            opt_tol=1e-5,
            feas_tol=1e-5,
            slack_reset=false,
            scaling_max_gradient=100.0,
            scaling=true,
            mu_min=1e-7,
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

        refresh()
    end
    names = [case for (case, _) in CASES]
    return [names results]
end

function parse_args(args::Vector{String})
    solver = nothing
    for arg in args
        if startswith(arg, "--solver=")
            solver = split(arg, "=")[2]
        end
    end
    return solver
end

function @main(args::Vector{String})
    solver = parse_args(args)

    if solver == "knitro"
        results = benchmark_knitro()
        writedlm(joinpath(@__DIR__, "..", "results", "scopf-knitro.csv"), results)
    elseif solver == "madncl-cpu"
        results = benchmark_scopf(;
            print_level=MadNLP.ERROR,
            linear_solver=Ma57Solver,
            richardson_tol=1e-12,
            richardson_max_iter=20,
            max_iter=1000,
            kkt_system=MadNCL.K2rAuglagKKTSystem,
        )
        writedlm(joinpath(@__DIR__, "..", "results", "scopf-madncl-k2r-ma57.csv"), results)
    elseif solver == "madncl-cuda"
        results = benchmark_scopf(;
            print_level=MadNLP.ERROR,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            cudss_pivot_epsilon=1e-10,
            richardson_tol=1e-12,
            richardson_max_iter=20,
            max_iter=1000,
            kkt_system=MadNCL.K2rAuglagKKTSystem,
        )
        writedlm(joinpath(@__DIR__, "..", "results", "scopf-madncl-k2r-cudss.csv"), results)
    end
end

