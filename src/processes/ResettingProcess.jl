# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Resetting process - returns to stationary distribution with some rate γ

struct ResettingProcessTransition{S <: Distribution, T <: Distribution, P <: Real} <: ContinuousMultivariateDistribution
    statdist::S
    raw_transdist::T
    jump_prob::P # probability of resetting to the stationary distribution
    lp::P # log(1 - exp(-γt))
    lnp::P # -γt
end

struct ResettingProcess{S <: Distribution, T <: Distribution, P <: AbstractProcess{S, T}, G <: Real
                      } <: AbstractProcess{S, T}
    raw_process::P         # original process
    γ::G                   # resetting rate
end

resetting(p::AbstractProcess, rate::Real) = ResettingProcess(p, rate)

ResettingWrappedDiffusion(args...; γ::Real) = resetting(WrappedDiffusion(args...), γ)

function resetprobs(γ::Real, t::Real)
    lnp = -γ * t
    lp = log1mexp(lnp)
    jump_prob = -expm1(lnp)
    (; jump_prob, lp, lnp)
end

statdist(j::ResettingProcess) = statdist(j.raw_process)
transdist(j::ResettingProcess, t::Real, x₀::AbstractVector{<:Real}) = ResettingProcessTransition(j, t, x₀)
function transdist!(r::AbstractVector{<:ResettingProcessTransition}, j::ResettingProcess{S, T},
                    t::Real, X₀::AbstractVecOrMat{<:Real}) where {S, T}
    (; jump_prob, lp, lnp) = resetprobs(j.γ, t)
    a = Array{T}(undef, size(r)); transdist!(a, j.raw_process, t, X₀)
    r .= ResettingProcessTransition.(Ref(statdist(j.raw_process)),
                             a, Ref(jump_prob), Ref(lp), Ref(lnp))
    r
end

# optimized for resetting processes
# Create a matrix r[i, j] = ℙ[Xᵢ, Yⱼ | process p]
# which gives the joint probability of points Xᵢ and Yⱼ under process p
function jointlogpdf!(r::AbstractMatrix{<:Real}, p::ResettingProcess{S, T}, t::Real,
                      X::AbstractVecOrMat{<:Real},
                      Y::AbstractVecOrMat{<:Real}) where {S, T}
    n = size(X, 2)
    m = size(Y, 2)
    # Construct transition distributions for each datapoint in Y
    transdists = Array{T}(undef, m)
    transdist!(transdists, p.raw_process, t, Y)

    (; lp, lnp) = resetprobs(p.γ, t)

    # Make each row of r into the log probability of the stationary distribution at Y
    # Add to each column the log transition density from Y to X
    tpd = similar(r, n); tpd .=-Inf
    stX = similar(tpd)
    logpdf!(stX, statdist(p), X)
    for j ∈ 1:m
        logpdf!(tpd, transdists[j], X)
        stY = logpdf(statdist(p), Y[:, j])
        r[:, j] .= stY .+ logaddexp.(lp .+ stX, lnp .+ tpd)
    end
    return r
end

# Print distribution parameters
Base.show(io::IO, j::ResettingProcess) = print(io, "ResettingProcess(" *
                                            "\nprocess: " * string(j.raw_process) *
                                            "\njump rate γ: " * string(j.γ) *
                                            "\n)")



function ResettingProcessTransition(j::ResettingProcess, t::Real, x₀::AbstractVector{<:Real})
    (; jump_prob, lp, lnp) = resetprobs(j.γ, t)
    ResettingProcessTransition(statdist(j.raw_process), transdist(j.raw_process, t, x₀),
                        jump_prob, lp, lnp)
end

Base.length(d::ResettingProcessTransition) = length(d.statdist)
Base.eltype(d::ResettingProcessTransition) = eltype(d.statdist)

function Distributions._rand!(rng::AbstractRNG, d::ResettingProcessTransition, x::AbstractVecOrMat{<:Real})
    n = size(x, 2)
    has_jumped = rand(rng, Bernoulli(d.jump_prob), n)
    stat_indices = findall(has_jumped)
    trans_indices = findall(.!has_jumped)
    if length(stat_indices) > 0
        x[:, stat_indices] .= rand(rng, d.statdist, length(stat_indices))
    end
    if length(trans_indices) > 0
        x[:, trans_indices] .= rand(rng, d.raw_transdist, length(trans_indices))
    end

    return x

end

function Distributions._logpdf(d::ResettingProcessTransition, x::AbstractVector{<:Real})
    return logaddexp(d.lp + logpdf(d.statdist, x), d.lnp + logpdf(d.raw_transdist, x))
end

function Distributions._logpdf!(r::AbstractArray{<: Real},
                  d::ResettingProcessTransition, X::AbstractVecOrMat{<: Real})
    r .= logaddexp.(d.lp .+ logpdf(d.statdist, X),
                    d.lnp .+ logpdf(d.raw_transdist, X))
    return r
end

Base.show(io::IO, n::ResettingProcessTransition) = print(io, "ResettingProcessTransition(" *
                                            "\nstat dist: " * string(n.statdist) *
                                            "\nraw trans dist: " * string(n.raw_transdist) *
                                            "\njump prob: " * string(n.jump_prob) *
                                            "\n)")

domain(d::ResettingProcessTransition) = domain(d.statdist)
