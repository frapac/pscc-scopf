
using DelimitedFiles
using JuMP
using NLPModels
using MadNLP, MadNLPHSL
using MadNLPGPU
using CUDA
using MadNCL
using ExaModels
using KNITRO

include(joinpath(@__DIR__, "..", "models", "goc1.jl"))
include(joinpath(@__DIR__, "..", "scripts", "utils.jl"))

function refresh()
    GC.gc(true)
    CUDA.reclaim()
    GC.gc(true)
end

function parse_dataset(dataset)
    instances = []
    for network in readdir(dataset)
        for files in readdir(joinpath(dataset, network))
            if startswith(files, "scenario")
                push!(instances, (network, files))
            end
        end
    end
    return instances
end

function benchmark_knitro(dataset)
    instances = parse_dataset(dataset)
    n = length(instances)
    contingencies = [10, 50]

    results = zeros(n * length(contingencies), 8)
    k = 1

    for (case, scenario) in instances, nK in contingencies
        model = goc1_model(
            joinpath(dataset, case),
            scenario,
            nK;
            tau_relax=1e-5,
            reg_l1=false,
            use_mpec=true,
        )
        ind_cc1, ind_cc2 = parse_ccons!(model; reformulation=:mpec)
        JuMP.set_optimizer(model, KNITRO.Optimizer)
        JuMP.set_optimizer_attribute(model, "maxit", 1000)
        JuMP.set_optimizer_attribute(model, "maxtime", 1200)
        JuMP.optimize!(model)

        results[k, 1] = nK
        results[k, 2] = JuMP.num_variables(model)
        results[k, 3] = Int(JuMP.is_solved_and_feasible(model))
        results[k, 4] = MOI.get(model, MOI.BarrierIterations())
        results[k, 5] = JuMP.objective_value(model)
        results[k, 6] = 0.0
        results[k, 7] = 0.0
        results[k, 8] = JuMP.solve_time(model)

        k += 1
    end
    names = String[]
    for (case, scenario) in instances, nK in contingencies
        push!(names, "$(case)_$(scenario)")
    end
    return [names results]
end

function benchmark_madncl(dataset; backend=nothing, options...)
    instances = parse_dataset(dataset)
    n = length(instances)
    contingencies = [10, 50]

    results = zeros(n * length(contingencies), 8)
    k = 1

    for (case, scenario) in instances, nK in contingencies
        model = goc1_model(
            joinpath(dataset, case),
            scenario,
            nK;
            use_mpec=false,
            tau_relax=1e-5,
            reg_l1=false,
        )

        nlp = ExaModels.ExaModel(model; backend=backend)

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

        refresh()

        k += 1
    end
    names = String[]
    for (case, scenario) in instances, nK in contingencies
        push!(names, "$(case)_$(scenario)")
    end
    return [names results]
end

function parse_args(args::Vector{String})
    solver = nothing
    dataset = joinpath("data", "trial_0")
    device = 0

    for arg in args
        if startswith(arg, "--solver=")
            solver = split(arg, "=")[2]
        elseif startswith(arg, "--dataset=")
            dataset = split(arg, "=")[2]
        elseif startswith(arg, "--device=")
            device = parse(Int, split(arg, "=")[2])
        end
    end
    return solver, dataset, device
end

function @main(args::Vector{String})
    solver, dataset, device = parse_args(args)
    flag = split(dataset, "/")[end]
    println(flag)

    CUDA.device!(device)

    if solver == "knitro"
        results = benchmark_knitro(dataset)
        writedlm(joinpath(@__DIR__, "..", "results", "goc1-$(flag)-knitro.csv"), results)
    elseif solver == "madncl-cpu"
        results = benchmark_madncl(
            dataset;
            print_level=MadNLP.ERROR,
            linear_solver=Ma57Solver,
            richardson_tol=1e-12,
            richardson_max_iter=20,
            bound_relax_factor=1e-7,
            max_iter=1000,
            max_wall_time=1200.0,
            kkt_system=MadNCL.K2rAuglagKKTSystem,
        )
        writedlm(joinpath(@__DIR__, "..", "results", "goc1-$(flag)-madncl-k2r-ma57.csv"), results)
    elseif solver == "madncl-cuda"
        results = benchmark_madncl(
            dataset;
            backend=CUDABackend(),
            print_level=MadNLP.ERROR,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            cudss_pivot_epsilon=1e-10,
            richardson_tol=1e-12,
            richardson_max_iter=20,
            bound_relax_factor=1e-7,
            max_iter=1000,
            max_wall_time=1200.0,
            kkt_system=MadNCL.K2rAuglagKKTSystem,
        )
        writedlm(joinpath(@__DIR__, "..", "results", "goc1-$(flag)-madncl-k2r-cudss.csv"), results)
    end
end

