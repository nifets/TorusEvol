# Interface for abstract (continuous time) reversible Markovian Processes
abstract type AbstractProcess{S <: Distribution, T <: Distribution} end

"""
    statdist(p::AbstractProcess)

Return the stationary distribution of `p`
"""
function statdist end

"""
    transdist(p::AbstractProcess, t, x₀)

Return the transition distribution of `p` at time `t` with initial state `x₀`
"""
function transdist end

# The transition distributions at time t starting from each column of X₀
function transdist!(r::AbstractVector{T},
                    p::AbstractProcess{S,T},
                    t,
                    X₀::AbstractVecOrMat) where {S,T}
    r .= transdist.(Ref(p), Ref(t), eachcol(X₀))
    return r
end

# Create a matrix r[i, j] = ℙ[Xᵢ, Yⱼ | process p, time t]
# which gives the joint probability of points Xᵢ and Yⱼ under process p
function jointlogpdf!(r::AbstractMatrix,
                      p::AbstractProcess{S,T},
                      t,
                      X::AbstractVecOrMat,
                      Y::AbstractVecOrMat) where {S,T}
    m = size(Y, 2)
    # Construct transition distributions for each datapoint in Y
    transdists = Vector{T}(undef, m); transdist!(transdists, p, t, Y)

    # Add to each column the log transition density from Y to X
    for j ∈ 1:m
        @views logpdf!(r[:, j], transdists[j], X)
    end
    # Add to each column of r the log probability of the stationary distribution at Y
    r .+= logpdf(statdist(p), Y)'
    return r
end

# Create a vector r[i] = lℙ[Xᵢ| process p]
function statlogpdf!(r, p::AbstractProcess, X)
    logpdf!(r, statdist(p), X)
    return r
end

# Create a matrix r[i, j] = ℙ[Yⱼ | Xᵢ, process p, time t]
# which gives the joint probability of points Xᵢ and Yⱼ under process p
function translogpdf!(r::AbstractMatrix, p::AbstractProcess,
                      t,
                      X::AbstractVecOrMat,
                      Y::AbstractVecOrMat)
    jointlogpdf!(r, p, t, X, Y)
    # remove from each row the statdist of X
    r .-= logpdf(statdist(p), X)
    return r
end

# Create a matrix r[i, j]
# which gives the full log probability of points Xᵢ and Yⱼ under process p
# (i.e. joint probability and stationary probability of each X and Y)
# r[i, j] = lℙ[Xᵢ, Yⱼ | process p, time t]
# r[i, m+1] = lℙ[Xᵢ | process p]
# r[n+1, j] = lℙ[Yⱼ | process p]
function fulllogpdf!(r::AbstractMatrix, p::AbstractProcess,
                     t,
                     X::AbstractVecOrMat,
                     Y::AbstractVecOrMat)
    n = size(X, 2)
    m = size(Y, 2)
    r[n+1, m+1] = 0
    @views jointlogpdf!(r[1:n, 1:m], p, t, X, Y)
    @views statlogpdf!(r[1:n, m+1], p, X)
    @views statlogpdf!(r[n+1, 1:m], p, Y)
    return r
end


function randjoint(rng::AbstractRNG, p::AbstractProcess, t)
    x = rand(rng, statdist(p))
    y = rand(rng, transdist(p, t, x))
    return x, y
end

randjoint(p::AbstractProcess, t) = randjoint(Random.default_rng(), p, t)

randtrans(rng::AbstractRNG, p::AbstractProcess, t, x) = rand(rng, transdist(p, t, x))

randtrans(p::AbstractProcess, t, x) = randtrans(Random.default_rng(), p, t, x)

Base.length(p::AbstractProcess) = length(statdist(p))
Base.eltype(p::AbstractProcess) = eltype(statdist(p))

domain(p::AbstractProcess) = domain(statdist(p))
