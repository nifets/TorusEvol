
# Wrapped Diffusion Transition Distribution Node

# This defines a pdf p(x) = tpdₜ(θ₀, x | 𝚯)
# which is the distribution of a particle moving according to a wrapped normal diffusion
# 𝚯, started at location θ₀ and allowed to diffuse for time t
struct WrappedDiffusionTransition{P,T,V,D,W} <: ContinuousMultivariateDistribution
    process::P
    t::T
    x₀::V
    driftdists::D # driftdists[k] = WN(μᵐₜ, Γₜ), where m = 𝕃[k]
    # winddist[k] = wₖ(θ) gives the probability of the winding number of the unwrapped
    # X ~ N(μ, Σ) being equal to 𝕃[k] ∈ 2πℤᵈ, given θ = cmod(X), sampled over the
    # distribution of WN(μ, Σ); i.e. wₖ(θ) is a measure of the contribution of each point in the 2πℤᵈ lattice to the overall pdf
    logweights::W
end

Base.length(d::WrappedDiffusionTransition) = Base.length(d.process);
Base.eltype(d::WrappedDiffusionTransition) = Base.eltype(d.process);
Base.show(io::IO, n::WrappedDiffusionTransition) = print(io, "WrappedDiffusionTransition(" *
                                              "\n𝚯: " * string(n.process) *
                                              "\nt: " * string(n.t) *
                                              "\nx₀: " * string(n.x₀) *
                                              "\n)")

# Wrapped Diffusion
# WD(μ, Σ, A) - analogue of OU(μ, Σ, A) over 𝕋ᵈ, with stationary distribution ~ WN(μ, statcov), statcov = ½A⁻¹Σ
struct WrappedDiffusion{V,S,M,C,W<:WrappedNormal} <: AbstractProcess{W,WrappedDiffusionTransition}
    μ::V           # mean
    Σ::S           # infinitesimal covariance
    A::M           # drift
    statcov::C     # stationary covariance
    statdist::W    # stationary distribution
end

Base.show(io::IO, 𝚯::WrappedDiffusion) = print(io, "WrappedDiffusion(" *
                                          "\ndim: " * string(Base.length(𝚯)) *
                                          "\nμ: " * string(𝚯.μ) *
                                          "\nΣ: " * string(𝚯.Σ) *
                                          "\nA: " * string(𝚯.A) *
                                          "\n)")

Base.length(𝚯::WrappedDiffusion) = Base.length(𝚯.μ)
Base.eltype(𝚯::WrappedDiffusion) = Base.eltype(𝚯.statdist);

Statistics.mean(𝚯::WrappedDiffusion) = 𝚯.μ
lattice(𝚯::WrappedDiffusion) = lattice(𝚯.statdist)

statdist(𝚯::WrappedDiffusion) = 𝚯.statdist
transdist(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector{<:Real}) =
    WrappedDiffusionTransition(𝚯, t, θ₀)



recenter(𝚯::WrappedDiffusion, μ::AbstractVector{<:Real}) = WrappedDiffusion(μ, 𝚯.Σ, 𝚯.A)

function stationary_covariance(A, Σ)
    d = size(A, 1)
    T = promote_type(eltype(A), eltype(Σ))
    Id = Matrix{T}(I, d, d)
    operator = kron(Id, A) + kron(A, Id)
    C = reshape(operator \ vec(Matrix(Σ)), d, d)
    return PDMat(Symmetric(C))
end

# Ensure Σ positive definite
function WrappedDiffusion(μ::AbstractVector{<:Real},
                          Σ::AbstractMatrix{<:Real},
                          A::AbstractMatrix{<:Real})
    d = length(μ)
    size(Σ) == (d, d) || throw(DimensionMismatch("Σ must be $d x $d"))
    size(A) == (d, d) || throw(DimensionMismatch("A must be $d x $d"))
    Σpd = PDMat(Symmetric(Σ))
    statcov = stationary_covariance(A, Σpd)
    wrapped_μ = cmod(μ)
    statdist = WrappedNormal(wrapped_μ, statcov)
    return WrappedDiffusion(wrapped_μ, Σpd, A, statcov, statdist)
end

# Constructor for 2-dimensional drift
# σ -- variance coefficient of each angle
# α -- drift coefficient for each angle, as well as their drift covariance
function WrappedDiffusion(μ_𝜙::Real, μ_𝜓::Real,
                          σ_𝜙::Real, σ_𝜓::Real,
                          α_𝜙::Real, α_𝜓::Real, α_cov::Real)
    σ_𝜙 > 0 || throw(ArgumentError("σ_𝜙 must be positive"))
    σ_𝜓 > 0 || throw(ArgumentError("σ_𝜓 must be positive"))
    α_𝜙 * α_𝜓 > α_cov^2 || throw(ArgumentError("drift parameters must define a stable process"))

    Σ = PDiagMat([σ_𝜙^2, σ_𝜓^2])
    A = [α_𝜙 (σ_𝜙*α_cov/σ_𝜓); (σ_𝜓*α_cov/σ_𝜙) α_𝜓]
    WrappedDiffusion([μ_𝜙, μ_𝜓], Σ, A)
end

## Transition logic

# e⁻ᵗᴬ - the 'drag' of the diffusion - attraction to current point
drift_coefficient(𝚯::WrappedDiffusion, t::Real) = drift_coefficient(𝚯.A, t)

function drift_coefficient(A::AbstractMatrix, t::Real)
    if size(A) == (2,2)
        return drift_coefficient_2d(A, t)
    end
    return exp(-t * A)
end

function drift_coefficient_2d(A, t)
    r = tr(A) / 2
    q² = r^2 - det(A)
    q² < 0 && return exp(-t * A)
    q = sqrt(q²)
    return 𝑎(-t, r, q) * I + 𝑏(-t, r, q) * A
end
𝑎(t, r, q) =  exp(r * t) * cosh(q*t) - r * 𝑏(t, r, q)
𝑏(t, r, q) = iszero(q) ? exp(r * t) * t : exp(r * t) * sinh(q * t) / q

# Γₜ - accumulated covariance through time
function transition_covariance(𝚯::WrappedDiffusion, t::Real)
    t > zero(t) || throw(ArgumentError("t must be positive"))
    B = drift_coefficient(𝚯, t)
    return transition_covariance(B, 𝚯.statcov)
end

function transition_covariance(B, C)
    Cmat = Matrix(C)
    PDMat(Symmetric(Cmat - B*Cmat*transpose(B)))
end

function WrappedDiffusionTransition(𝚯::WrappedDiffusion, t::Real, x₀::AbstractVector{<:Real})
    length(x₀) == length(𝚯) || throw(DimensionMismatch("x₀ has the wrong dimension"))
    t > zero(t) || throw(ArgumentError("t must be positive"))
    B = drift_coefficient(𝚯, t)
    Γₜ = transition_covariance(B, 𝚯.statcov)

    wn = statdist(𝚯)
    shifted_lattice = x₀ .+ lattice(wn)
    μᴹₜ = cmod(mean(𝚯) .+ B * (shifted_lattice .- mean(𝚯)))

    transition_lattice = lattice(WrappedNormal(zeros(length(𝚯)), Γₜ))
    driftdists = WrappedNormal.(eachcol(μᴹₜ), Ref(Γₜ), Ref(transition_lattice))

    logweights = logpdf(unwrapped(wn), shifted_lattice) .- logpdf(wn, x₀)
    logweights .-= logsumexp(logweights)

    return WrappedDiffusionTransition(𝚯, t, x₀, driftdists, logweights)
end

# Optimized for several transition starting points
function transdist!(r::AbstractVector, 𝚯::WrappedDiffusion,
                    t::Real, X₀::AbstractVecOrMat{<:Real})
    size(X₀, 1) == length(𝚯) || throw(DimensionMismatch("X₀ has the wrong dimension"))
    t > zero(t) || throw(ArgumentError("t must be positive"))

    B = drift_coefficient(𝚯, t)
    Γₜ = transition_covariance(B, 𝚯.statcov)
    wn = statdist(𝚯)
    transition_lattice = lattice(WrappedNormal(zeros(length(𝚯)), Γₜ))

    for i in axes(X₀, 2)
        @views x₀ = X₀[:, i]
        shifted_lattice = x₀ .+ lattice(wn)
        μᴹₜ = cmod(mean(𝚯) .+ B * (shifted_lattice .- mean(𝚯)))
        driftdists = WrappedNormal.(eachcol(μᴹₜ), Ref(Γₜ), Ref(transition_lattice))
        logweights = logpdf(unwrapped(wn), shifted_lattice) .- logpdf(wn, x₀)
        logweights .-= logsumexp(logweights)

        r[i] = WrappedDiffusionTransition(𝚯, t, x₀, driftdists, logweights)
    end
    return r
end

function Distributions._rand!(rng::AbstractRNG, d::WrappedDiffusionTransition, x::AbstractVecOrMat{<:Real})
    n = size(x, 2)

    # step 1 - sample from winddist
    windnums = rand(rng, Categorical(exp.(d.logweights)), n)

    # step 2 - sample from WN(μᵐₜ, Γₜ)
    for w ∈ eachindex(d.driftdists)
        indx = windnums .== w
        if any(indx)
            @views Distributions._rand!(rng, d.driftdists[w], x[:, indx])
        end
    end

    x
end

# Log density of WN over 𝕋ᵈ
Distributions._logpdf(d::WrappedDiffusionTransition, x::AbstractVector{<: Real}) =
    logsumexp(logpdf.(d.driftdists, Ref(cmod(x))) .+ d.logweights)

function Distributions._logpdf!(r::AbstractArray{<:Real},
                                d::WrappedDiffusionTransition, X::AbstractVecOrMat{<: Real})
    wrapped_X = cmod.(X)

    r .= -Inf
    tape = similar(r)
    for i ∈ eachindex(d.driftdists)
        Distributions._logpdf!(tape, d.driftdists[i], wrapped_X)
        r .= logaddexp.(r, tape .+ d.logweights[i])
    end

    r
end

domain(d::WrappedDiffusionTransition) = domain(d.process)
