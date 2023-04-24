using Distributions, DistributionsAD
using LinearAlgebra
using Memoization


import Base: length, eltype, show
import Distributions: _logpdf, _logpdf!, mean, _rand!

# __________________________________________________________________________________________
# Interface for abstract (continuous time) Markovian Processes
abstract type AbstractProcess{D <: Distribution} end

# The stationary distribution of the process
function statdist(::AbstractProcess) end

# The transition distribution at time t starting from x₀
function transdist(::AbstractProcess, t::Real, x₀::AbstractVector{<:Real}) end

# The transition distributions at time t starting from each column of X₀
function transdist!(r::AbstractVector, p::AbstractProcess,
                    t::Real, X₀::AbstractVecOrMat{<:Real})
    r .= transdist.(Ref(p), Ref(t), eachcol(X₀))
    return r
end

# Create a matrix r[i, j] = ℙ[Xᵢ, Yⱼ | process p]
# which gives the joint probability of points Xᵢ and Yⱼ under process p
@memoize function jointlogpdf!(r::AbstractMatrix{<:Real}, p::AbstractProcess{D}, t::Real,
                               X::AbstractVecOrMat,
                               Y::AbstractVecOrMat) where D <: Distribution
    m = size(Y, 2)
    # Construct transition distributions for each datapoint in Y
    transdists = Array{D}(undef, m); transdist!(transdists, p, t, Y)

    # Make each row of r into the log probability of the stationary distribution at Y
    #r .= logpdf(statdist(p), Y)'

    # Add to each column the log transition density from Y to X
    for j ∈ 1:m
        r[:, j] .= logpdf(transdists[j], X) .+ logpdf(statdist(p), Y[:, j])
    end
    return r
end

@memoize function statlogpdf!(r::AbstractVector{<:Real}, p::AbstractProcess,
                              X::AbstractVector)
    logpdf!(r, statdist(p), X)
end

function randjoint(p::AbstractProcess, t::Real)
    x = rand(statdist(p))
    y = rand(transdist(p, t, x))
    return x, y
end