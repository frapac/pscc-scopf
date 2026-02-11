
#=
    Implement corrective SCOPF as formulated in the GO competition.
=#

using JuMP
using PowerModels
using PowerModelsSecurityConstrained

PowerModels.silence()

function goc1_data(instance, scenario)
    case = parse_c1_case(joinpath(instance, "inputfiles.ini"); scenario_id=scenario)
    data = build_c1_pm_model(case)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    return PowerModels.build_ref(data)[:it][:pm][:nw][0]
end

function goc1_model(
    instance,
    scenario,
    nK;
    tau_relax = 1e-5,
    load_factor = 1.0,
    reg_l1=true,
    use_mpec=false,
)
    ref = goc1_data(instance, scenario)

    NN = 3
    sbase = ref[:baseMVA]
    pviolmax = [2.0, 50.0, Inf] ./ sbase
    qviolmax = [2.0, 50.0, Inf] ./ sbase
    sviolmax = [2.0, 50.0, Inf] ./ sbase
    pviolcost = [1e3, 5e3, 1e6]
    qviolcost = [1e3, 5e3, 1e6]
    sviolcost = [1e3, 5e3, 1e6]
    δ = 0.5

    npw = maximum([g["ncost"] for (i, g) in ref[:gen]])

    ngen = length(ref[:gen])
    alpha = [ref[:gen][i]["alpha"] for i in keys(ref[:gen])] ./ sbase

    # Parse contingencies
    contingencies = [cont.idx for cont in ref[:branch_contingencies]][1:nK]
    K = length(contingencies) + 1

    # Build model
    model = JuMP.Model()

    JuMP.@variable(model, va[i in keys(ref[:bus]), 1:K], start=ref[:bus][i]["va"])
    JuMP.@variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus]), 1:K] <= ref[:bus][i]["vmax"], start=ref[:bus][i]["vm"])
    JuMP.@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen]), 1:K] <= ref[:gen][i]["pmax"], start=ref[:gen][i]["pg"])
    JuMP.@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen]), 1:K] <= ref[:gen][i]["qmax"], start=ref[:gen][i]["qg"])
    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs], 1:K] <= ref[:branch][l]["rate_a"])
    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs], 1:K] <= ref[:branch][l]["rate_a"])
    # Piecewise linear objective
    JuMP.@variable(model, 0.0 <= tgh[i in keys(ref[:gen]), 1:npw])
    # Deviation
    if reg_l1
        JuMP.@variable(model, 0.0 <= σppn[i in keys(ref[:bus]), 1:K, n in 1:NN] <= pviolmax[n], start=0.0)
        JuMP.@variable(model, 0.0 <= σpp[i in keys(ref[:bus]), 1:K], start=0.0)

        JuMP.@variable(model, 0.0 <= σpnn[i in keys(ref[:bus]), 1:K, n in 1:NN] <= pviolmax[n], start=0.0)
        JuMP.@variable(model, 0.0 <= σpn[i in keys(ref[:bus]), 1:K], start=0.0)

        JuMP.@variable(model, 0.0 <= σqpn[i in keys(ref[:bus]), 1:K, n in 1:NN] <= qviolmax[n], start=0.0)
        JuMP.@variable(model, 0.0 <= σqp[i in keys(ref[:bus]), 1:K], start=0.0)

        JuMP.@variable(model, 0.0 <= σqnn[i in keys(ref[:bus]), 1:K, n in 1:NN] <= qviolmax[n], start=0.0)
        JuMP.@variable(model, 0.0 <= σqn[i in keys(ref[:bus]), 1:K], start=0.0)

        JuMP.@variable(model, 0.0 <= σsn[i in keys(ref[:branch]), 1:K, n in 1:NN] <= sviolmax[n], start=0.0)
        JuMP.@variable(model, 0.0 <= σs[i in keys(ref[:branch]), 1:K], start=0.0)
    end
    # Complementarity formulation
    JuMP.@variable(model, 0.0 <= ρp[i in keys(ref[:gen]), 2:K])
    JuMP.@variable(model, 0.0 <= ρn[i in keys(ref[:gen]), 2:K])
    JuMP.@variable(model, 0.0 <= vp[i in keys(ref[:gen]), 2:K])
    JuMP.@variable(model, 0.0 <= vn[i in keys(ref[:gen]), 2:K])
    # Automatic adjustment of generators
    JuMP.@variable(model, Δ[1:K-1])

    if reg_l1
        JuMP.@objective(
            model,
            Min,
            sum(gen["cost"][2*n]*tgh[i, n] for (i, gen) in ref[:gen], n in 1:gen["ncost"]) +
            δ * (
                sum(pviolcost[n] * (σppn[i, 1, n] + σpnn[i, 1, n]) for i in keys(ref[:bus]), n in 1:NN) +
                sum(qviolcost[n] * (σqpn[i, 1, n] + σqnn[i, 1, n]) for i in keys(ref[:bus]), n in 1:NN) +
                sum(sviolcost[n] * σsn[i, 1, n] for i in keys(ref[:branch]), n in 1:NN)
            ) +
            (1-δ) / nK * (
                sum(pviolcost[n] * (σppn[i, k, n] + σpnn[i, k, n]) for i in keys(ref[:bus]), k in 2:K, n in 1:NN) +
                sum(qviolcost[n] * (σqpn[i, k, n] + σqnn[i, k, n]) for i in keys(ref[:bus]), k in 2:K, n in 1:NN) +
                sum(sviolcost[n] * σsn[i, k, n] for i in keys(ref[:branch]), k in 2:K, n in 1:NN)
            )
        )
    else
        JuMP.@objective(
            model,
            Min,
            sum(gen["cost"][2*n]*tgh[i, n] for (i, gen) in ref[:gen], n in 1:gen["ncost"])
        )
    end

    @constraint(model, [g in keys(ref[:gen])], sum(tgh[g, n] for n in 1:ref[:gen][g]["ncost"]) == 1.0)
    @constraint(model, [g in keys(ref[:gen])], sum(tgh[g, n] * ref[:gen][g]["cost"][2*(n-1)+1] for n in 1:ref[:gen][g]["ncost"]) == pg[g, 1])

    if reg_l1
        @constraint(model, [i in keys(ref[:bus]), k=1:K], σpp[i, k] == sum(σppn[i, k, n] for n in 1:NN))
        @constraint(model, [i in keys(ref[:bus]), k=1:K], σpn[i, k] == sum(σpnn[i, k, n] for n in 1:NN))
        @constraint(model, [i in keys(ref[:bus]), k=1:K], σqp[i, k] == sum(σqpn[i, k, n] for n in 1:NN))
        @constraint(model, [i in keys(ref[:bus]), k=1:K], σqn[i, k] == sum(σqnn[i, k, n] for n in 1:NN))
        @constraint(model, [i in keys(ref[:branch]), k=1:K], σs[i, k] == sum(σsn[i, k, n] for n in 1:NN))
    end

    for k in 2:K
        # Droop control
        for (j, i) in enumerate(keys(ref[:gen]))
            pmin, pmax = ref[:gen][i]["pmin"], ref[:gen][i]["pmax"]
            @constraint(model, ρp[i, k] - ρn[i, k] == pg[i, k] - pg[i, 1] - alpha[j] * Δ[k-1])
            if use_mpec
                @constraint(model, [(pmax - pg[i, k]), ρn[i, k]] in MOI.Complements(2))
                @constraint(model, [(pg[i, k] - pmin), ρp[i, k]] in MOI.Complements(2))
            else
                @constraint(model, ρn[i, k] * (pmax - pg[i, k]) <= tau_relax)
                @constraint(model, ρp[i, k] * (pg[i, k] - pmin) <= tau_relax)
            end
        end
        # Voltage magnitude are not adjusted at PV buses
        for g in keys(ref[:gen])
            b = ref[:gen][g]["gen_bus"]
            qmin, qmax = ref[:gen][g]["qmin"], ref[:gen][g]["qmax"]
            @constraint(model, vp[g, k] - vn[g, k] == vm[b, k] - vm[b, 1])
            if isfinite(qmax)
                if use_mpec
                    @constraint(model, [(qmax - qg[g, k]), vn[g, k]] in MOI.Complements(2))
                else
                    @constraint(model, vn[g, k] * (qmax - qg[g, k]) <= tau_relax)
                end
            end
            if isfinite(qmin)
                if use_mpec
                    @constraint(model, [(qg[g, k] - qmin), vp[g, k]] in MOI.Complements(2))
                else
                    @constraint(model, vp[g, k] * (qg[g, k] - qmin) <= tau_relax)
                end
            end
        end
        # Set flux to 0
        for (l, i, j) in ref[:arcs]
            if l == contingencies[k-1]
                @constraint(model, p[(l, i, j), k] == 0.0)
                @constraint(model, q[(l, i, j), k] == 0.0)
            end
        end
    end

    for k in 1:K
        for (i, bus) in ref[:ref_buses]
            JuMP.@constraint(model, va[i, k] == 0)
        end

        for (i,bus) in ref[:bus]
            bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
            bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

            if reg_l1
                JuMP.@constraint(model,
                    sum(p[a, k] for a in ref[:bus_arcs][i]) ==
                    sum(pg[g, k] for g in ref[:bus_gens][i]) -
                    sum(load_factor * load["pd"] for load in bus_loads) -
                    sum(shunt["gs"] for shunt in bus_shunts)*vm[i, k]^2
                    + σpp[i, k] - σpn[i, k]
                )
                JuMP.@constraint(model,
                    sum(q[a, k] for a in ref[:bus_arcs][i]) ==
                    sum(qg[g, k] for g in ref[:bus_gens][i]) -
                    sum(load_factor * load["qd"] for load in bus_loads) +
                    sum(shunt["bs"] for shunt in bus_shunts)*vm[i, k]^2
                    + σqp[i, k] - σqn[i, k]
                )
            else
                JuMP.@constraint(model,
                    sum(p[a, k] for a in ref[:bus_arcs][i]) ==
                    sum(pg[g, k] for g in ref[:bus_gens][i]) -
                    sum(load_factor * load["pd"] for load in bus_loads) -
                    sum(shunt["gs"] for shunt in bus_shunts)*vm[i, k]^2
                )
                JuMP.@constraint(model,
                    sum(q[a, k] for a in ref[:bus_arcs][i]) ==
                    sum(qg[g, k] for g in ref[:bus_gens][i]) -
                    sum(load_factor * load["qd"] for load in bus_loads) +
                    sum(shunt["bs"] for shunt in bus_shunts)*vm[i, k]^2
                )
            end
        end

        # Branch power flow physics and limit constraints
        for (i,branch) in ref[:branch]
            if (k >= 2) && i == contingencies[k-1]
                continue
            end
            f_idx = (i, branch["f_bus"], branch["t_bus"])
            t_idx = (i, branch["t_bus"], branch["f_bus"])

            p_fr = p[f_idx, k]
            q_fr = q[f_idx, k]
            p_to = p[t_idx, k]
            q_to = q[t_idx, k]

            vm_fr = vm[branch["f_bus"], k]
            vm_to = vm[branch["t_bus"], k]
            va_fr = va[branch["f_bus"], k]
            va_to = va[branch["t_bus"], k]

            g, b = PowerModels.calc_branch_y(branch)
            tr, ti = PowerModels.calc_branch_t(branch)
            ttm = tr^2 + ti^2
            g_fr = branch["g_fr"]
            b_fr = branch["b_fr"]
            g_to = branch["g_to"]
            b_to = branch["b_to"]

            # From side of the branch flow
            JuMP.@constraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
            JuMP.@constraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )

            # To side of the branch flow
            JuMP.@constraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
            JuMP.@constraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )

            # Apparent power limit, from side and to side
            if reg_l1
                JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2 + σs[i, k])
                JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2 + σs[i, k])
            else
                JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
                JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
            end
        end
    end
    return model
end

