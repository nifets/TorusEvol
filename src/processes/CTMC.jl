# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Continuous Time Markov Chain (time-reversible with discrete state)
struct CTMC{D<:Categorical, M<:AbstractMatrix} <: AbstractProcess{D, Categorical}
    statdist::D
    Q::M
end

function CTMC(π::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real})
    n = length(π)
    size(Q) == (n, n) || throw(DimensionMismatch("Q must be $n x $n"))
    all(π .>= 0) || throw(ArgumentError("stationary probabilities must be nonnegative"))
    isapprox(sum(π), one(eltype(π)); atol=1e-10) || throw(ArgumentError("stationary probabilities must sum to one"))
    all(isapprox.(vec(sum(Q; dims=2)), 0; atol=1e-10)) || throw(ArgumentError("rows of Q must sum to zero"))
    all(Q[i, j] >= 0 for i in 1:n for j in 1:n if i != j) || throw(ArgumentError("off-diagonal entries of Q must be nonnegaive"))
    all(isapprox.(transpose(π) * Q, 0; atol=1e-10)) || throw(ArgumentError("π must be stationary for Q"))

    return CTMC(Categorical(π), Q)
end

function CTMC(Q::AbstractMatrix{<:Real})
    # compute statdist from eq: statdist * Q = 0 and 1ᵀπ=1
    # can replace last row of Q with 1's to capture both.
    n, m = size(Q)
    n == m || throw(DimensionMismatch("Q must be square"))

    A = copy(transpose(Q))
    b = zeros(eltype(Q), n)
    A[end, :] .= one(eltype(Q))
    b[end] = one(eltype(Q))
    π = A \ b
    return CTMC(π, Q)
end

# Create a matrix r[i, j] = ℙ[Xᵢ, Yⱼ | process p]
# which gives the joint probability of points Xᵢ and Yⱼ under process p
function jointlogpdf!(r::AbstractMatrix, c::CTMC, t::Real,
                      X::AbstractVecOrMat,
                      Y::AbstractVecOrMat)
    r .= jointlp(c, t)[vec(X), vec(Y)]
    return r
end

function statlogpdf!(r::AbstractVector{<:Real}, c::CTMC,
                     X::AbstractVecOrMat)
    r .= logpdf.(Ref(statdist(c)), vec(X))
    return r
end

transmat(c::CTMC, t::Real) = exp(t * c.Q)

statdist(c::CTMC) = c.statdist

jointlp(c::CTMC, t::Real) = log.(transmat(c, t)) .+ reshape(log.(probs(statdist(c))), :, 1)

function transdist(c::CTMC, t::Real, x₀::Integer)
    probabilities = max.(transmat(c, t)[x₀, :], 0)
    return Categorical(probabilities ./ sum(probabilities))
end
transdist(c::CTMC, t::Real, x₀::AbstractArray) = transdist(c, t, only(x₀))

num_states(c::CTMC) = size(c.Q, 1)


Base.show(io::IO, c::CTMC) = print(io, "CTMC(" *
                                  "\nnum states: " * string(num_states(c)) *
                                  "\nstat dist: " * string(round.(statdist(c).p; digits=3)) *
                                  "\nQ matrix: " * string(round.(c.Q; digits=3)) *
                                  "\n)")


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# CTMC based substitution process

# given some rate matrix Q, we can find the transition matrix P(t)
# by exponentiating Q

# i.e. P(t) = exp(t * Q)

# p(i, j) = π(i) * P(t)[i, j]

# p(i, _) = π(i)

# so the substitution model should implement likelihood Methods
# for i -> j as well as equilibrium frequencies

# Q = S diag(Π) where S is symmetric exchangeability matrix,
#        Π is equilibrium frequencies

function substitution_process(S::LowerTriangular, Π::AbstractVector; r = 1.0)
    r > 0 || throw(ArgumentError("rate must be positive"))
    S = Matrix(Symmetric(S, :L))
    S[diagind(S)] .= 0.0

    Q = S * Diagonal(Π)
    Q[diagind(Q)] .= vec(sum(-Q, dims = 2))

    mean_replacement = -dot(Q[diagind(Q)], Π)
    mean_replacement > 0 || throw(ArgumentError("mean substitution rate must be positive"))

    Q ./= mean_replacement
    Q .*= r

    return CTMC(Π, Q)
end
