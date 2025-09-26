
using JuMP
using NLPModels
using LinearAlgebra
using SparseArrays
using HiGHS
using Gurobi
using Printf

struct LPEC
    n::Int
    m::Int
    n_cc::Int
    c::Vector{Float64}
    g::Vector{Float64}
    Ji::Vector{Int}
    Jj::Vector{Int}
    Jx::Vector{Float64}
    lvar::Vector{Float64}
    uvar::Vector{Float64}
    lcon::Vector{Float64}
    ucon::Vector{Float64}
    ind_cc1::Vector{Int}
    ind_cc2::Vector{Int}
end

function build_lpec(nlp::AbstractNLPModel, ind_cc1, ind_cc2)
    @assert length(ind_cc1) == length(ind_cc2)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    n_cc = length(ind_cc1)
    nnzj = NLPModels.get_nnzj(nlp)
    Ji = zeros(Int, nnzj)
    Jj = zeros(Int, nnzj)
    NLPModels.jac_structure!(nlp, Ji, Jj)
    return LPEC(
        n, m, n_cc,
        zeros(m),
        zeros(n),
        Ji,
        Jj,
        zeros(nnzj),
        NLPModels.get_lvar(nlp),
        NLPModels.get_uvar(nlp),
        NLPModels.get_lcon(nlp),
        NLPModels.get_ucon(nlp),
        ind_cc1,
        ind_cc2,
    )
end

function update!(lpec::LPEC, nlp::AbstractNLPModel, x)
    NLPModels.cons!(nlp, x, lpec.c)
    NLPModels.grad!(nlp, x, lpec.g)
    NLPModels.jac_coord!(nlp, x, lpec.Jx)
    # Clamp solution between bounds to ensure we are feasible
    lpec.c .= clamp.(lpec.c, lpec.lcon, lpec.ucon)
    return
end

function build_lpec_model(lpcc::LPEC, x, rho, I0, I1, I2; solver=:gurobi, initialize=false)
    ind_cc1, ind_cc2 = lpcc.ind_cc1, lpcc.ind_cc2
    J = sparse(lpcc.Ji, lpcc.Jj, lpcc.Jx, lpcc.m, lpcc.n)
    n = lpcc.n
    n0 = length(I0)
    M = 100.0
    if solver == :gurobi
        model = Model(Gurobi.Optimizer)
        JuMP.set_optimizer_attribute(model, "FeasibilityTol", 1e-6)
        JuMP.set_optimizer_attribute(model, "IntFeasTol", 1e-6)
        JuMP.set_optimizer_attribute(model, "OptimalityTol", 1e-6)
        JuMP.set_optimizer_attribute(model, "MIPGap", 1e-4)
        JuMP.set_optimizer_attribute(model, "MIPGapAbs", 1e-9)
        JuMP.set_optimizer_attribute(model, "MIPFocus", 1)
        JuMP.set_optimizer_attribute(model, "Presolve", 1)
    elseif solver == :highs
        model = Model(HiGHS.Optimizer)
        JuMP.set_optimizer_attribute(model, "kkt_tolerance", 1e-6)
        JuMP.set_optimizer_attribute(model, "mip_feasibility_tolerance", 1e-6)
        JuMP.set_optimizer_attribute(model, "optimality_tolerance", 1e-6)
        JuMP.set_optimizer_attribute(model, "mip_rel_gap", 1e-4)
        JuMP.set_optimizer_attribute(model, "mip_abs_gap", 1e-9)
        JuMP.set_optimizer_attribute(model, "presolve", "on")
        JuMP.set_optimizer_attribute(model, "mip_heuristic_effort", 0.0)
        #JuMP.set_optimizer_attribute(model, "solver", "ipm")
        #JuMP.set_optimizer_attribute(model, "parallel", "on")
        JuMP.set_attribute(model, HiGHS.ComputeInfeasibilityCertificate(), false)
    end
    #JuMP.set_silent(model)
    @variable(model, -rho <= d[1:lpcc.n] <= rho)
    @variable(model, y[1:n0], Bin)
    #@variable(model, 0<= y[1:n0] <= 1)
    @objective(model, Min, dot(lpcc.g, d))
    # Bound constraints
    @constraint(model, lpcc.lvar .<= x .+ d .<= lpcc.uvar)
    # Generic constraints
    @constraint(model, lpcc.lcon .<= lpcc.c .+ J * d .<= lpcc.ucon)
    # I1
    @constraint(model, [i in I1], x[ind_cc1[i]] + d[ind_cc1[i]] == lpcc.lvar[ind_cc1[i]])
    @constraint(model, [i in I1], x[ind_cc2[i]] + d[ind_cc2[i]] >= lpcc.lvar[ind_cc2[i]])
    # I2
    @constraint(model, [i in I2], x[ind_cc1[i]] + d[ind_cc1[i]] >= lpcc.lvar[ind_cc1[i]])
    @constraint(model, [i in I2], x[ind_cc2[i]] + d[ind_cc2[i]] == lpcc.lvar[ind_cc2[i]])
    # I0
    @constraint(model, [i in I0], x[ind_cc1[i]] + d[ind_cc1[i]] >= lpcc.lvar[ind_cc1[i]])
    @constraint(model, [i in I0], x[ind_cc2[i]] + d[ind_cc2[i]] >= lpcc.lvar[ind_cc2[i]])
    @constraint(model, [i in 1:n0], x[ind_cc1[I0[i]]] + d[ind_cc1[I0[i]]] - lpcc.lvar[ind_cc1[i]] <= M * y[i])
    @constraint(model, [i in 1:n0], x[ind_cc2[I0[i]]] + d[ind_cc2[I0[i]]] - lpcc.lvar[ind_cc2[i]] <= M * (1.0 - y[i]))

    if initialize
        set_start_value.(d[1:lpcc.n], 0.0)
        set_start_value.(y[1:n0], 0.0)
    end

    return model
end

function find_partition(lpcc::LPEC, x, tol)
    ind_cc1, ind_cc2 = lpcc.ind_cc1, lpcc.ind_cc2
    partition = zeros(Int, lpcc.n_cc)
    for i in 1:lpcc.n_cc
        x1, x2 = x[ind_cc1[i]], x[ind_cc2[i]]
        lx1, lx2 = lpcc.lvar[ind_cc1[i]], lpcc.lvar[ind_cc2[i]]
        if x1-lx1 <= tol && x2-lx2 <= tol
            partition[i] = 0
        elseif x1-lx1 <= tol
            partition[i] = 1
        elseif x2-lx2 <= tol
            partition[i] = 2
        else
            println((i, x1, x2))
            error("Current point is not feasible")
        end
        #println("I: $(partition[i]) $(x1 - lx1), $(x2 - lx2)")
    end
    return partition
end

function solve_lpec!(lpcc::LPEC, x, rho; initialize=false)
    partition = find_partition(lpcc, x, rho)
    I0 = findall(isequal(0), partition)
    I1 = findall(isequal(1), partition)
    I2 = findall(isequal(2), partition)
    println("I0 cnt: $(length(I0))")
    if isempty(I0)
        # I0 is empty, therefore partition is good enough already
        return zeros(Float64, lpcc.n), partition
    end
    model = build_lpec_model(lpcc, x, rho, I0, I1, I2; initialize=initialize)
    JuMP.optimize!(model)
    # Get descent direction
    d = JuMP.value.(model[:d])
    y = JuMP.value.(model[:y])
    # Find branch partition using MILP solution
    for i in I1
        partition[i] = 1
    end
    for i in I2
        partition[i] = 2
    end
    for k in 1:length(I0)
        partition[I0[k]] = y[k] >= 0.5 ? 2 : 1
    end
    return d, partition
end

function solve_branch_nlp!(lpcc, nlp, partition)
    ind_cc1, ind_cc2 = lpcc.ind_cc1, lpcc.ind_cc2
    # Freeze variables to build branch NLP.
    for k in 1:lpcc.n_cc
        i1, i2 = ind_cc1[k], ind_cc2[k]
        # manual upper bound relax
        ubound_relax_factor = 0.0
        if partition[k] == 1 # belong to I1
            #print("Pushing x[$(i1)] by $(nlp.meta.x0[i1] - nlp.meta.lvar[i1])")
            nlp.meta.x0[i1] = nlp.meta.lvar[i1]
            nlp.meta.uvar[i1] = nlp.meta.lvar[i1] + ubound_relax_factor

            #println(" And x[$(i2)] upper bound is $(lpcc.uvar[i2])")
            nlp.meta.uvar[i2] = lpcc.uvar[i2]
        else                 # belong to I2
            nlp.meta.uvar[i1] = lpcc.uvar[i1]

            #print("Pushing x[$(i2)] by $(nlp.meta.x0[i2] - nlp.meta.lvar[i2])")
            nlp.meta.x0[i2] = nlp.meta.lvar[i2]
            nlp.meta.uvar[i2] = nlp.meta.lvar[i2] + ubound_relax_factor

            #println(" and x[$(i1)] upper bound is $(lpcc.uvar[i1])")
        end
    end
    # Call MadNLP
    return madnlp(nlp; linear_solver=Ma57Solver, print_level=MadNLP.INFO, bound_push=1e-6, bound_fac=1e-6, bound_relax_factor=0.0, max_iter=10000, mu_init=1e-1)
end

function mpecopt!(
    nlp,
    bnlp,
    ind_cc1,
    ind_cc2,
    x;
    tol=1e-6,
    max_iter=10,
    tr_alpha = 0.1,
    tr_radius = 1e-3,
    tr_0 = 1e-3,
    tr_min = 1e-7,
)
    lpcc = build_lpec(bnlp, ind_cc1, ind_cc2)
    current_objective = Inf#NLPModels.obj(nlp, x)

    status = MadNLP.INITIAL

    d = zeros(Float64, lpcc.n)
    println("Starting crossover method using MPECopt.")
    bnlp_feasible = false
    for i in 1:max_iter
        step_type = ""
        update!(lpcc, bnlp, x)
        # Solve LPCC
        (d, partition) = solve_lpec!(lpcc, x, tr_radius, initialize=bnlp_feasible)
        #println(norm(d,Inf))
        if bnlp_feasible && norm(d, Inf) <= tol
            status = MadNLP.SOLVE_SUCCEEDED
            break
        elseif bnlp_feasible && abs(dot(d, lpcc.g)) <= tol
            status = MadNLP.SOLVE_SUCCEEDED
            println("feasible BNLP with negligible descent direction")
            break
        elseif tr_radius <= tr_min
            status = MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL
            break
        end
        # Solve branch NLP
        bnlp.meta.x0 .= x
        results = solve_branch_nlp!(lpcc, bnlp, partition)

        x_trial = results.solution
        c = cons(bnlp, x_trial)
        inf_c = mapreduce((lc, c_, uc) -> max(c_-uc, lc-c_, 0), max, bnlp.meta.lcon, c, bnlp.meta.ucon; init=0.0)
        inf_x = mapreduce((lx, x_, ux) -> max(x_-ux, lx-x_, 0), max, lpcc.lvar, x_trial, lpcc.uvar; init=0.0)
        inf_cc = mapreduce((x1, x2, lx1, lx2) -> max(min(x1-lx1, x2-lx2), 0), max,
                           x_trial[ind_cc1], x_trial[ind_cc2], bnlp.meta.lvar[ind_cc1], bnlp.meta.lvar[ind_cc2];
                           init = 0.0)
        # Update parameters
        if results.status == MadNLP.SOLVE_SUCCEEDED
            bnlp_feasible = true
            if results.objective < current_objective
                current_objective = results.objective
                step_type = "fi"
                tr_radius = tr_0 # reset to phaseII trust radius
            else
                step_type = "fn"
                tr_radius *= tr_alpha
            end
            x .= x_trial # Update solution since the BNLP succeeded.
        else
            step_type = "I"
            tr_radius *= tr_alpha
        end
        @printf("%3i %10.7e %5.2e %5.2e %7.4e %7.4e %7.4e %s\n",
                i, current_objective, norm(d, Inf), tr_radius,
                inf_c, inf_x, inf_cc, step_type)
    end
    bnlp.meta.lvar .= lpcc.lvar
    bnlp.meta.uvar .= lpcc.uvar
    return status, x
end
