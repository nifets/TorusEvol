struct RegimeMixture{W<:AbstractVector{<:Real},D<:AbstractVector} <: Distribution{TupleVariate,ValueSupport}
    logweights::W
    components::D
end

Base.length(d::RegimeMixture) = 1 + length(first(d.components))

Distributions.logpdf(d::RegimeMixture, x::Tuple) = d.logweights[first(x)] + logpdf(d.components[first(x)], Base.tail(x))
Distributions.pdf(d::RegimeMixture, x::Tuple) = exp(logpdf(d, x))

function Base.rand(rng::AbstractRNG, d::RegimeMixture)
    regime = rand(rng, Categorical(exp.(d.logweights)))
    return (regime, rand(rng, d.components[regime])...)
end
Base.rand(rng::AbstractRNG, d::RegimeMixture, n::Int) = [rand(rng, d) for _ in 1:n]

domain(d::RegimeMixture) = (eachindex(d.components), domain(first(d.components))...)

Distributions.logpdf(d::MixtureModel{TupleVariate}, x::Tuple) =
    logsumexp(log(prior) + logpdf(component, x) for (prior, component) in zip(probs(d), components(d)) if prior > 0)
Distributions.pdf(d::MixtureModel{TupleVariate}, x::Tuple) = exp(logpdf(d, x))
domain(d::MixtureModel) = domain(first(components(d)))

Base.rand(rng::AbstractRNG, d::MixtureModel{TupleVariate}) = rand(rng, component(d, rand(rng, d.prior)))
Base.rand(rng::AbstractRNG, d::MixtureModel{TupleVariate}, n::Int) = [rand(rng, d) for _ in 1:n]

struct RegimeProcess{C<:CTMC,R<:AbstractVector{<:Tuple}} <: AbstractProcess{RegimeMixture,RegimeMixture}
    regime_chain::C
    regimes::R
    function RegimeProcess(regime_chain::C, regimes::R) where {C,R}
        num_states(regime_chain) == length(regimes) || throw(DimensionMismatch("the regime process components and regimes should match"))
        new{C,R}(regime_chain, regimes)
    end
end

frozen_regimes(weights::AbstractVector) = CTMC(weights ./ sum(weights), zeros(length(weights), length(weights)))

function switching_regimes(weights::AbstractVector, rate::Real)
    normalised = weights ./ sum(weights)
    return CTMC(normalised, rate .* (ones(length(normalised)) * normalised' - I))
end

ProductProcess(processes::AbstractProcess...) = RegimeProcess(frozen_regimes([1.0]), [processes])

num_regimes(p::RegimeProcess) = length(p.regimes)
num_coords(p::RegimeProcess) = length(first(p.regimes))

regime_statdist(regime::Tuple) = TupleProduct(map(statdist, regime))
regime_transdist(regime::Tuple, t, x::Tuple) = TupleProduct(map((process, coordinate) -> transdist(process, t, coordinate), regime, x))

statdist(p::RegimeProcess) =
    RegimeMixture(log.(probs(statdist(p.regime_chain))), [regime_statdist(regime) for regime in p.regimes])

function transdist(p::RegimeProcess, t, x::Tuple)
    regime, coordinates = first(x), Base.tail(x)
    switching_probs = max.(transmat(p.regime_chain, t)[regime, :], 0)
    stay_prob = exp(t * p.regime_chain.Q[regime, regime])
    returned_prob = max(switching_probs[regime] - stay_prob, 0)
    components = Distribution{TupleVariate,ValueSupport}[regime_statdist(other) for other in p.regimes]
    components[regime] = MixtureModel(
        Distribution{TupleVariate,ValueSupport}[regime_transdist(p.regimes[regime], t, coordinates), components[regime]],
        [stay_prob, returned_prob] ./ (stay_prob + returned_prob))
    return RegimeMixture(log.(switching_probs), components)
end

Base.show(io::IO, p::RegimeProcess) =
    print(io, "RegimeProcess(regimes: ", num_regimes(p), ", coords: ", num_coords(p), ", weights: ", probs(statdist(p.regime_chain)), ")")
