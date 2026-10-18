# Map x to [-π, π)
cmod(x::Real) = rem(x,2π, RoundNearest)
cmod(v::AbstractArray{<:Real}) = cmod.(v)

# Returns [-r,r]ᵈ ⊂ ℤᵈ
hyper_cube(r::Integer, d::Integer) =
    reshape(hcat(map(collect, vec(collect(Base.product(fill(-r:r, d)...))))...), d, :)

# Returns 2π[-r,r]ᵈ ⊂ 2πℤᵈ
twoπ_hyper_cube(r::Integer, d::Integer) =
    2π .* hyper_cube(r, d)

function discrete_ellipsoid(𝛷::MvNormal; tol=1e-10)
    0 < tol < 1 || throw(ArgumentError("tol must lie between 0 and 1"))

    d = length(𝛷)
    μ = mean(𝛷)
    Σ = Matrix(cov(𝛷))
    centred_normal = MvNormal(zeros(d), Σ)

    radius = sqrt(quantile(Chisq(d), 1 - tol))
    corners = hcat((collect(x) for x in Iterators.product(fill((-π, π), d)...))...)
    cell_radius = sqrt(maximum(sqmahal(centred_normal, corners)))
    expanded_radius = radius + cell_radius

    winding_ranges = map(1:d) do i
        extent = expanded_radius * sqrt(Σ[i, i])
        first_winding = ceil(Int, (μ[i] - extent) / (2π))
        last_winding = floor(Int, (μ[i] + extent) / (2π))
        first_winding:last_winding
    end

    windings = hcat((collect(k) for k in Iterators.product(winding_ranges...))...)
    lattice = 2π * windings
    keep = sqmahal(𝛷, lattice) .<= expanded_radius^2
    return lattice[:, keep]
end


# Wrapped Normal Distribution
# WN(μ, Σ) - analogue of N(μ, Σ) over 𝕋ᵈ
struct WrappedNormal{D<:ContinuousMultivariateDistribution, L<:AbstractMatrix} <: ContinuousMultivariateDistribution
    𝛷::D # the underlying unwrapped distribution
    𝕃::L             # 𝕃 ⊂ 2πℤᵈ truncated torus lattice
                     #    used to collect (most of) the probability mass
                     #    of 𝛷 into [-π, π)ᵈ
    function WrappedNormal{D, L}(𝛷::D, 𝕃::L) where {D<:ContinuousMultivariateDistribution, L<:AbstractMatrix}
        size(𝕃, 1) == length(𝛷) || throw(DimensionMismatch("lattice dimension does not match distribution"))
        return new{D, L}(𝛷, 𝕃)
    end
end

WrappedNormal(𝛷::D, 𝕃::L) where {D<:ContinuousMultivariateDistribution, L<:AbstractMatrix} =
    WrappedNormal{D, L}(𝛷, 𝕃)

function WrappedNormal(μ::AbstractVector{<:Real}, Σ; tol=1e-10)
    𝛷 = MvNormal(cmod(μ), Σ)
    𝕃 = discrete_ellipsoid(𝛷; tol)
    return WrappedNormal(𝛷, 𝕃)
end

function WrappedNormal(μ::AbstractVector{<:Real}, Σ, winding_radius::Integer)
    winding_radius >= 0 ||throw(ArgumentError("winding_radius must be nonnegative"))
    𝛷 = MvNormal(cmod(μ), Σ)
    𝕃 = twoπ_hyper_cube(winding_radius, length(μ))

    return WrappedNormal(𝛷, 𝕃)
end

function WrappedNormal(μ::AbstractVector{<:Real}, Σ, 𝕃::AbstractMatrix)
    return WrappedNormal(MvNormal(cmod(μ), Σ), 𝕃)
end

function WrappedNormal(μ::Real, σ²::Real; kwargs...)
    WrappedNormal([μ], [σ²;;]; kwargs...)
end

function WrappedNormal(μ::Real, σ²::Real, winding_radius::Integer)
    WrappedNormal([μ], [σ²;;], winding_radius)
end

unwrapped(wn::WrappedNormal) = wn.𝛷
lattice(wn::WrappedNormal) = wn.𝕃

Base.length(wn::WrappedNormal) = length(wn.𝛷)
Base.eltype(wn::WrappedNormal) = eltype(wn.𝛷)

function Distributions._rand!(rng::AbstractRNG, wn::WrappedNormal, x::AbstractVecOrMat)
    Distributions._rand!(rng, wn.𝛷, x)
    x .= cmod.(x)
    return x
end

function Distributions._logpdf(wn::WrappedNormal, x::AbstractVector{<: Real})
    return logsumexp(logpdf(wn.𝛷, cmod.(x) .+ wn.𝕃))
end

function Distributions._logpdf!(r::AbstractVector,
                                wn::WrappedNormal,
                                X::AbstractMatrix)
    size(X, 1) == length(wn) || throw(DimensionMismatch("expected $(length(wn))"))
    wrapped_X = cmod.(X)
    shifted_X = similar(wrapped_X)
    shifted_logp = similar(r)

    r .= -Inf
    for shift in eachcol(wn.𝕃)
        shifted_X .= wrapped_X .+ reshape(shift, :, 1)
        logpdf!(shifted_logp, wn.𝛷, shifted_X)
        r .= logaddexp.(r, shifted_logp)
    end
    r
end

Statistics.mean(wn::WrappedNormal) = mean(wn.𝛷)

Base.show(io::IO, wn::WrappedNormal) = print(io, "WrappedNormal(" *
                                          "\ndim: " * string(length(wn)) *
                                          "\nμ: " * string(wn.𝛷.μ) *
                                          "\nΣ: " * string(wn.𝛷.Σ) *
                                          "\n)")

domain(d::WrappedNormal) = DomainSets.ProductDomain(ntuple(_ -> PeriodicInterval(-π, π), length(d))...)
