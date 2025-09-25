
using JuMP
using NLPModels
using LinearAlgebra
using SparseArrays
using HiGHS
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

function build_lpec_model(lpcc::LPEC, x, rho, I0, I1, I2)
    ind_cc1, ind_cc2 = lpcc.ind_cc1, lpcc.ind_cc2
    J = sparse(lpcc.Ji, lpcc.Jj, lpcc.Jx, lpcc.m, lpcc.n)
    n = lpcc.n
    n0 = length(I0)
    M = 1000.0
    model = Model(HiGHS.Optimizer)
    JuMP.set_optimizer_attribute(model, "kkt_tolerance", 1e-8)
    JuMP.set_optimizer_attribute(model, "mip_feasibility_tolerance", 1e-8)
    JuMP.set_optimizer_attribute(model, "optimality_tolerance", 1e-8)
    JuMP.set_optimizer_attribute(model, "mip_rel_gap", 1e-4)
    JuMP.set_optimizer_attribute(model, "mip_abs_gap", 1e-9)
    # JuMP.set_silent(model)
    @variable(model, -rho <= d[1:lpcc.n] <= rho)
    @variable(model, y[1:n0], Bin)
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
    return model
end

function find_partition(lpcc::LPEC, x, tol)
    ind_cc1, ind_cc2 = lpcc.ind_cc1, lpcc.ind_cc2
    partition = zeros(Int, lpcc.n_cc)
    for i in 1:lpcc.n_cc
        x1, x2 = x[ind_cc1[i]], x[ind_cc2[i]]
        if x1 <= tol && x2 <= tol
            partition[i] = 0
        elseif x1 <= tol
            partition[i] = 1
        elseif x2 <= tol
            partition[i] = 2
        else
            println((i, x1, x2))
            error("Current point is not feasible")
        end
    end
    return partition
end

function solve_lpec!(lpcc::LPEC, x, rho)
    partition = find_partition(lpcc, x, 1e-8)
    I0 = findall(isequal(0), partition)
    I1 = findall(isequal(1), partition)
    I2 = findall(isequal(2), partition)
    model = build_lpec_model(lpcc, x, rho, I0, I1, I2)
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
        partition[I0[k]] = y[k] + 1
    end
    return d, partition
end

function solve_branch_nlp!(lpcc, nlp, partition)
    ind_cc1, ind_cc2 = lpcc.ind_cc1, lpcc.ind_cc2
    # Freeze variables to build branch NLP.
    for k in 1:lpcc.n_cc
        i1, i2 = ind_cc1[k], ind_cc2[k]
        if partition[k] == 1 # belong to I1
            nlp.meta.x0[i1] = nlp.meta.lvar[i1]
            nlp.meta.uvar[i1] = nlp.meta.lvar[i1]
            nlp.meta.uvar[i2] = lpcc.uvar[i2]
        else                 # belong to I2
            nlp.meta.uvar[i1] = lpcc.uvar[i1]
            nlp.meta.x0[i2] = nlp.meta.lvar[i2]
            nlp.meta.uvar[i2] = nlp.meta.lvar[i2]
        end
    end
    # Call MadNLP
    return madnlp(nlp; linear_solver=Ma27Solver, print_level=MadNLP.ERROR)
end

function mpecopt!(
    nlp,
    ind_cc1,
    ind_cc2,
    x;
    tol=1e-6,
    max_iter=10,
    tr_alpha = 0.1,
    tr_radius = 1e-3,
    tr_min = 1e-7,
)
    lpcc = build_lpec(nlp, ind_cc1, ind_cc2)
    current_objective = NLPModels.obj(nlp, x)

    status = MadNLP.INITIAL

    println("Starting crossover method using MPECopt.")

    for i in 1:max_iter
        update!(lpcc, nlp, x)
        # Solve LPCC
        (d, partition) = solve_lpec!(lpcc, x, tr_radius)
        if norm(d, Inf) <= tol
            status = MadNLP.SOLVE_SUCCEEDED
            break
        elseif tr_radius <= tr_min
            status = MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL
            break
        end
        # Solve branch NLP
        nlp.meta.x0 .= x
        results = solve_branch_nlp!(lpcc, nlp, partition)
        # Update parameters
        if results.objective < current_objective
            current_objective = results.objective
            x .= results.solution
        else
            tr_radius *= tr_alpha
        end
        @printf("%3i %10.7e %5.2e %5.2e\n", i, current_objective, norm(d, Inf), tr_radius)
    end

    nlp.meta.lvar .= lpcc.lvar
    nlp.meta.uvar .= lpcc.uvar
    return status, x
end
