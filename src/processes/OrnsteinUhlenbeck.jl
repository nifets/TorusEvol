struct OrnsteinUhlenbeck{V,S,M,C} <: AbstractProcess{MvNormal, MvNormal}
    μ::V
    Σ::S
    A::M
    statcov::C
end

function OrnsteinUhlenbeck(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, A::AbstractMatrix{<:Real})
    d = length(μ)
    size(Σ) == (d, d) || throw(DimensionMismatch("Σ must be $d x $d"))
    size(A) == (d, d) || throw(DimensionMismatch("A must be $d x $d"))
    Σpd = PDMat(Symmetric(Σ))
    return OrnsteinUhlenbeck(μ, Σpd, A, stationary_covariance(A, Σpd))
end

Base.length(p::OrnsteinUhlenbeck) = length(p.μ)
Statistics.mean(p::OrnsteinUhlenbeck) = p.μ

statdist(p::OrnsteinUhlenbeck) = MvNormal(p.μ, p.statcov)

function transdist(p::OrnsteinUhlenbeck, t::Real, x₀::AbstractVector{<:Real})
    B = drift_coefficient(p.A, t)
    return MvNormal(p.μ .+ B * (x₀ .- p.μ), transition_covariance(B, p.statcov))
end

OrnsteinUhlenbeck(μ::AbstractVector{<:Real}, variance::Real, rate::Real) =
    OrnsteinUhlenbeck(μ, ScalMat(length(μ), 2*rate*variance), rate*I, ScalMat(length(μ), variance))

const IsotropicOU = OrnsteinUhlenbeck{<:Any, <:ScalMat, <:UniformScaling}

drift_coefficient(A::UniformScaling, t::Real) = exp(-t * A.λ) * I

transition_covariance(B::UniformScaling, C::ScalMat) = ScalMat(C.dim, C.value * (1-B.λ^2))

domain(d::MvNormal) = GaussianDomain(d)
