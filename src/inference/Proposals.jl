using Turing, DynamicPPL
using LinearAlgebra
using LogExpFunctions
using Plots, StatsPlots
using Random
using Distributions

torus_proposal(v) = MixtureModel([WrappedNormal(v, I), WrappedNormal(v, 20*I)], [0.8, 0.2])
mv_rw_proposal(v::AbstractVector, cov) = MvNormal(v, cov)
rw_proposal(x, var) = Normal(x, var)

Θ_samplers = [MH(Symbol("Θ.μ") => v -> torus_proposal(v)),
              MH(Symbol("Θ.σ²") => v -> mv_rw_proposal(v, 0.1*I)),
              MH(Symbol("Θ.α") => v -> mv_rw_proposal(v, 0.1*I)),
              MH(Symbol("Θ.α_corr") => x -> rw_proposal(x, 0.05)),
              MH(Symbol("Θ.γ") => x -> rw_proposal(x, 0.5))]

τ_samplers = [MH(Symbol("τ.λμ") => v -> mv_rw_proposal(v, [0.2 0.02; 0.02 0.2])),
              MH(Symbol("τ.r") => x -> rw_proposal(x, 0.2))]

t_samplers = [MH(Symbol("t") => v -> mv_rw_proposal(v, 0.3*I))]
