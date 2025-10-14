
using DelimitedFiles

function get_power_data_ref(case)
    data = PowerModels.parse_file(case)
    PowerModels.standardize_cost_terms!(data, order = 2)
    PowerModels.calc_thermal_limits!(data)
    return PowerModels.build_ref(data)[:it][:pm][:nw][0]
end

convert_data(data::N, backend) where {names,N<:NamedTuple{names}} =
    NamedTuple{names}(ExaModels.convert_array(d, backend) for d in data)
parse_ac_power_data(filename, backend) =
    convert_data(parse_ac_power_data(filename), backend)


function parse_ac_power_data(filename)
    ref = get_power_data_ref(filename)

    arcdict = Dict(a => k for (k, a) in enumerate(ref[:arcs]))
    busdict = Dict(k => i for (i, (k, v)) in enumerate(ref[:bus]))
    gendict = Dict(k => i for (i, (k, v)) in enumerate(ref[:gen]))
    branchdict = Dict(k => i for (i, (k, v)) in enumerate(ref[:branch]))

    return (
        bus = [
            begin
                bus_loads = [ref[:load][l] for l in ref[:bus_loads][k]]
                bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][k]]
                pd = sum(load["pd"] for load in bus_loads; init = 0.0)
                gs = sum(shunt["gs"] for shunt in bus_shunts; init = 0.0)
                qd = sum(load["qd"] for load in bus_loads; init = 0.0)
                bs = sum(shunt["bs"] for shunt in bus_shunts; init = 0.0)
                (i = busdict[k], pd = pd, gs = gs, qd = qd, bs = bs, bus_type = v["bus_type"])
            end for (k, v) in ref[:bus]
        ],
        gen = [
            (
                i = gendict[k],
                cost1 = v["cost"][1],
                cost2 = v["cost"][2],
                cost3 = v["cost"][3],
                bus = busdict[v["gen_bus"]],
            ) for (k, v) in ref[:gen]
        ],
        arc = [
            (i = k, rate_a = ref[:branch][l]["rate_a"], bus = busdict[i]) for
            (k, (l, i, j)) in enumerate(ref[:arcs])
        ],
        branch = [
            begin
                f_idx = arcdict[i, branch["f_bus"], branch["t_bus"]]
                t_idx = arcdict[i, branch["t_bus"], branch["f_bus"]]
                g, b = PowerModels.calc_branch_y(branch)
                tr, ti = PowerModels.calc_branch_t(branch)
                ttm = tr^2 + ti^2
                g_fr = branch["g_fr"]
                b_fr = branch["b_fr"]
                g_to = branch["g_to"]
                b_to = branch["b_to"]
                c1 = (-g * tr - b * ti) / ttm
                c2 = (-b * tr + g * ti) / ttm
                c3 = (-g * tr + b * ti) / ttm
                c4 = (-b * tr - g * ti) / ttm
                c5 = (g + g_fr) / ttm
                c6 = (b + b_fr) / ttm
                c7 = (g + g_to)
                c8 = (b + b_to)
                (
                    i = branchdict[i],
                    j = 1,
                    f_idx = f_idx,
                    t_idx = t_idx,
                    f_bus = busdict[branch["f_bus"]],
                    t_bus = busdict[branch["t_bus"]],
                    c1 = c1,
                    c2 = c2,
                    c3 = c3,
                    c4 = c4,
                    c5 = c5,
                    c6 = c6,
                    c7 = c7,
                    c8 = c8,
                    rate_a_sq = branch["rate_a"]^2,
                )
            end for (i, branch) in ref[:branch]
        ],
        ref_buses = [busdict[i] for (i, k) in ref[:ref_buses]],
        vmax = [v["vmax"] for (k, v) in ref[:bus]],
        vmin = [v["vmin"] for (k, v) in ref[:bus]],
        pmax = [v["pmax"] for (k, v) in ref[:gen]],
        pmin = [v["pmin"] for (k, v) in ref[:gen]],
        qmax = [v["qmax"] for (k, v) in ref[:gen]],
        qmin = [v["qmin"] for (k, v) in ref[:gen]],
        rate_a = [ref[:branch][l]["rate_a"] for (k, (l, i, j)) in enumerate(ref[:arcs])],
        angmax = [b["angmax"] for (i, b) in ref[:branch]],
        angmin = [b["angmin"] for (i, b) in ref[:branch]],
    )
end

function parse_sc_power_data(data, contingencies, corrective_action_ratio, backend)
    # Raw data
    nbus = length(data.bus)
    ngen = length(data.gen)
    narc = length(data.arc)
    # Contingencies
    K = length(contingencies) + 1 # +1 accounts for base case

    # Build indexes
    idx_ref = [(r, k) for r in data.ref_buses, k in 1:K]
    idx_branch = [(b, 1) for b in data.branch]
    idx_branch_down = similar(idx_branch, 0)
    # Scan contingencies
    for k in 2:K # k=1 is base case
        line_down = contingencies[k-1] # ID of line down
        for b in data.branch
            if b.i != line_down
                push!(idx_branch, (b, k))
            else
                push!(idx_branch_down, (b, k))
            end
        end
    end

    idx_bus = [(b, k) for b in data.bus, k in 1:K]
    idx_gen = [(b, k) for b in data.gen, k in 1:K]
    idx_arc = [(b, k) for b in data.arc, k in 1:K]
    idx_gen_cont = map(
        b -> NamedTuple{(:i, :cost1, :cost2, :cost3, :bus, :pmax, :pmin, :qmin, :qmax)}(
            (b.i, b.cost1, b.cost2, b.cost3, b.bus, data.pmax[b.i], data.pmin[b.i], data.qmin[b.i], data.qmax[b.i])
        ),
        data.gen
    )
    qmin_finite_data = filter(g -> isfinite(g.qmin), idx_gen_cont)
    qmax_finite_data = filter(g -> isfinite(g.qmax), idx_gen_cont)

    # Compute maximum injection in lines
    max_inj = repeat(data.rate_a, K)
    # Remove (pinj, qinj) for lines down  by setting the bounds to 0
    # so the solver can convert it as a fixed variable.
    for (b, k) in idx_branch_down
        max_inj[b.f_idx + (k-1) * narc] = 0.0
        max_inj[b.t_idx + (k-1) * narc] = 0.0
    end
    gen_buses = [b for b in data.bus if b[:bus_type] in [2, 3]] # Select buses with generators (=PV and REF buses)

    data = convert_data(
        (
            data...,
            idx_ref = idx_ref,
            idx_branch = idx_branch,
            idx_branch_down = idx_branch_down,
            idx_bus = idx_bus,
            idx_gen = idx_gen,
            idx_arc = idx_arc,
            max_inj = max_inj,
            Δp = corrective_action_ratio .* (data.pmax .- data.pmin),
            idx_gen_cont = [(g, k) for g in idx_gen_cont, k in 2:K], # discard base case
            idx_qmin_finite_cont = [(g, k) for g in qmin_finite_data, k in 2:K],
            idx_qmax_finite_cont = [(g, k) for g in qmax_finite_data, k in 2:K],
            idx_bus_cont = [(b, k) for b in gen_buses, k in 2:K],
        ),
        backend)
    return (
        data...,
        K = K,
        nbus = nbus,
        ngen = ngen,
        narc = narc,
    )
end

function scopf_exa_model(
    filename,
    contingencies;
    backend = nothing,
    data = parse_ac_power_data(filename),
    load_factor =1.0,
    adjust=:droop,
    voltage_control=:none,
    corrective_action_ratio = 0.1,
    scale_obj=1e-3,
    scale_cc=1e-3,
    T = Float64,
    kwargs...
)

    data = parse_sc_power_data(data, contingencies, corrective_action_ratio, backend)

    core = ExaModels.ExaCore(T; backend = backend)

    alpha = ones(data.ngen)

    # Voltage angle and magnitudes
    va = ExaModels.variable(core, 1:data.nbus, 1:data.K; start=zeros(data.nbus, data.K))
    vm = ExaModels.variable(
        core,
        1:data.nbus, 1:data.K;
        start = ones(data.nbus, data.K),
        lvar = repeat(data.vmin, data.K),
        uvar = repeat(data.vmax, data.K),
    )
    # Power generations
    pg = ExaModels.variable(
        core,
        1:data.ngen, 1:data.K;
        lvar = repeat(data.pmin, data.K),
        uvar = repeat(data.pmax, data.K),
    )
    qg = ExaModels.variable(
        core,
        1:data.ngen, 1:data.K;
        lvar = repeat(data.qmin, data.K),
        uvar = repeat(data.qmax, data.K),
    )
    # Power injection
    p = ExaModels.variable(
        core,
        1:data.narc, 1:data.K;
        lvar = -data.max_inj,
        uvar = data.max_inj,
    )
    q = ExaModels.variable(
        core,
        1:data.narc, 1:data.K;
        lvar = -data.max_inj,
        uvar = data.max_inj,
    )

    # Objective (cost for base case only)
    obj = ExaModels.objective(
        core,
        scale_obj * (g.cost1 * pg[g.i, 1]^2 + g.cost2 * pg[g.i, 1] + g.cost3) for g in data.gen
    )

    if adjust == :mpecdroop
        pp = ExaModels.variable(
            core,
            1:data.ngen, 2:data.K; # Don't think need counting
            lvar = 0,
            uvar = Inf
        )
        pn = ExaModels.variable(
            core,
            1:data.ngen, 2:data.K; # Don't think need counting
            lvar = 0,
            uvar = Inf
        )
    end

    if voltage_control == :pvpq
        vp = ExaModels.variable(
            core,
            1:data.ngen, 2:data.K; # Don't think need counting
            lvar = 0,
            uvar = Inf
        )
        vn = ExaModels.variable(
            core,
            1:data.ngen, 2:data.K; # Don't think need counting
            lvar = 0,
            uvar = Inf
        )
    end

    delta = ExaModels.variable(core, 2:data.K;)

    # Constraints
    # Voltage angle at reference buses is 0
    c1 = ExaModels.constraint(core, va[i, k] for (i, k) in data.idx_ref)
    # Branch power injection
    # from, w.r.t active power
    c2 = ExaModels.constraint(
        core,
        p[b.f_idx, k] - b.c5 * vm[b.f_bus, k]^2 -
        b.c3 * (vm[b.f_bus, k] * vm[b.t_bus, k] * cos(va[b.f_bus, k] - va[b.t_bus, k])) -
        b.c4 * (vm[b.f_bus, k] * vm[b.t_bus, k] * sin(va[b.f_bus, k] - va[b.t_bus, k])) for
        (b, k) in data.idx_branch
    )
    # from, w.r.t reactive power
    c3 = ExaModels.constraint(
        core,
        q[b.f_idx, k] +
        b.c6 * vm[b.f_bus, k]^2 +
        b.c4 * (vm[b.f_bus, k] * vm[b.t_bus, k] * cos(va[b.f_bus, k] - va[b.t_bus, k])) -
        b.c3 * (vm[b.f_bus, k] * vm[b.t_bus, k] * sin(va[b.f_bus, k] - va[b.t_bus, k])) for
        (b, k) in data.idx_branch
    )
    # to, w.r.t active power
    c4 = ExaModels.constraint(
        core,
        p[b.t_idx, k] - b.c7 * vm[b.t_bus, k]^2 -
        b.c1 * (vm[b.t_bus, k] * vm[b.f_bus, k] * cos(va[b.t_bus, k] - va[b.f_bus, k])) -
        b.c2 * (vm[b.t_bus, k] * vm[b.f_bus, k] * sin(va[b.t_bus, k] - va[b.f_bus, k])) for
        (b, k) in data.idx_branch
    )
    # to, w.r.t reactive power
    c5 = ExaModels.constraint(
        core,
        q[b.t_idx, k] +
        b.c8 * vm[b.t_bus, k]^2 +
        b.c2 * (vm[b.t_bus, k] * vm[b.f_bus, k] * cos(va[b.t_bus, k] - va[b.f_bus, k])) -
        b.c1 * (vm[b.t_bus, k] * vm[b.f_bus, k] * sin(va[b.t_bus, k] - va[b.f_bus, k])) for
        (b, k) in data.idx_branch
    )

    # Line-flow constraints
    # from
    lcon = fill!(similar(data.branch, Float64, data.K * length(data.branch)), -Inf)
    c7 = ExaModels.constraint(
        core,
        p[b.f_idx, k]^2 + q[b.f_idx, k]^2 - b.rate_a_sq for (b, k) in data.idx_branch;
        lcon = lcon,
    )
    # to
    c8 = ExaModels.constraint(
        core,
        p[b.t_idx, k]^2 + q[b.t_idx, k]^2 - b.rate_a_sq for (b, k) in data.idx_branch;
        lcon = lcon,
    )

    # Power flow constraints
    c9 = ExaModels.constraint(core, load_factor*b.pd + b.gs * vm[b.i, k]^2 for (b, k) in data.idx_bus)
    c10 = ExaModels.constraint(core, load_factor*b.qd - b.bs * vm[b.i, k]^2 for (b, k) in data.idx_bus)
    # TODO: shifts of index is hacky.
    c11 = ExaModels.constraint!(core, c9, a.bus + (k-1)*data.nbus => p[a.i, k] for (a, k) in data.idx_arc)
    c12 = ExaModels.constraint!(core, c10, a.bus + (k-1)*data.nbus => q[a.i, k] for (a, k) in data.idx_arc)
    c13 = ExaModels.constraint!(core, c9, g.bus + (k-1)*data.nbus => -pg[g.i, k] for (g, k) in data.idx_gen)
    c14 = ExaModels.constraint!(core, c10, g.bus + (k-1)*data.nbus=> -qg[g.i, k] for (g, k) in data.idx_gen)


    # Corrective OPF formulation
    if adjust == :droop
        c_corrective_gen = ExaModels.constraint(
            core,
            pg[g.i, k] - pg[g.i, 1] - alpha[g.i]*delta[k] for (g, k) in data.idx_gen_cont; # JuMP use enumerate with a new variable j for alpha here, not sure if this is necessary
        )
    elseif adjust == :mpecdroop
        p_corrective_gen = ExaModels.constraint(
            core,
            pp[g.i, k] - pn[g.i, k] - pg[g.i, k] + pg[g.i, 1] + delta[k] for (g, k) in data.idx_gen_cont; # JuMP use enumerate with a new variable j for alpha here, not sure if this is necessary
        )
        ExaModels.constraint(
            core,
            scale_cc * pn[g.i, k] * (g.pmax - pg[g.i, k]) for (g,k) in data.idx_gen_cont;
            lcon = -Inf
        )
        ExaModels.constraint(
            core,
            scale_cc * pp[g.i, k] * (pg[g.i, k] - g.pmin) for (g,k) in data.idx_gen_cont;
            lcon = -Inf
        )
    elseif adjust == :preventive
        ExaModels.constraint(
            core,
            pg[g.i, k] - pg[g.i, 1] for (g,k) in data.idx_gen_cont;
        )
    elseif adjust == :relaxed
        ExaModels.constraint(
            core,
            pg[g.i, k] - pg[g.i, 1] for (g, k) in data.idx_gen_cont;
            lcon = repeat(-data.Δp, data.K),
            ucon = repeat(data.Δp, data.K),
        )
    end

    if voltage_control == :pvpq
        ExaModels.constraint(
            core,
            vp[g.i, k] - vn[g.i, k] - (vm[g.bus, k] - vm[g.bus, 1]) for (g, k) in data.idx_gen_cont; # no i need gen bus
        )
        ind_cc = ExaModels.constraint(
            core,
            scale_cc * vn[g.i , k] * (g.qmax - qg[g.bus, k]) for (g, k) in data.idx_qmax_finite_cont;
            lcon = -Inf
        )
        ExaModels.constraint(
            core,
            scale_cc * vn[g.i, k] * (qg[g.bus, k] - g.qmin) for (g, k) in data.idx_qmin_finite_cont;
            lcon = -Inf
        )
    else
        ExaModels.constraint(
            core,
            vm[g.bus, k] - vm[g.bus, 1] for (g, k) in data.idx_gen_cont
        )
    end

    model =ExaModels.ExaModel(core; kwargs...)

    vars = (
        va = va,
        vm = vm,
        vp = vp,
        vn = vn,
        pg = pg,
        qg = qg,
        p = p,
        q = q,
        pp = pp,
        pn = pn,
        delta = delta,
    )

    return model, vars, ind_cc

end

