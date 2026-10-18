struct TupleVariate <: VariateForm end

struct TupleProduct{D <: Tuple} <: Distribution{TupleVariate, ValueSupport}
    components::D
end

TupleProduct(components::Distribution...) = TupleProduct(components)

Base.length(d::TupleProduct) = length(d.components)

Distributions.logpdf(d::TupleProduct, x::Tuple) = sum(map(logpdf, d.components, x))
Distributions.pdf(d::TupleProduct, x::Tuple) = exp(logpdf(d, x))

Base.rand(rng::AbstractRNG, d::TupleProduct) = map(component -> rand(rng, component), d.components)
Base.rand(rng::AbstractRNG, d::TupleProduct, n::Int) = [rand(rng, d) for _ in 1:n]

domain(d::TupleProduct) = map(domain, d.components)
