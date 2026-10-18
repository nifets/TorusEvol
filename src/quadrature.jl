abstract type QuadratureRule end

struct Midpoint <: QuadratureRule
    n::Int
end
Midpoint() = Midpoint(100)

struct Adaptive <: QuadratureRule
    rtol::Float64
    maxevals::Int
end
Adaptive(; rtol=1e-6, maxevals=10^6) = Adaptive(rtol, maxevals)

function quadrature(rule::Midpoint, interval::DomainSets.AbstractInterval)
    spacing = DomainSets.width(interval) / rule.n
    nodes = permutedims(DomainSets.leftendpoint(interval) .+ ((0:rule.n-1) .+ 0.5) .* spacing)
    return nodes, fill(spacing, rule.n)
end

quadrature(::QuadratureRule, states::AbstractVector) = states, ones(length(states))

function quadrature(rule::Midpoint, product::DomainSets.ProductDomain)
    factors = [quadrature(rule, component) for component in DomainSets.components(product)]
    indices = vec(collect(Iterators.product((axes(nodes, 2) for (nodes, _) in factors)...)))
    nodes = reduce(vcat, [factor_nodes[:, getindex.(indices, k)] for (k, (factor_nodes, _)) in enumerate(factors)])
    weights = [prod(factor_weights[index[k]] for (k, (_, factor_weights)) in enumerate(factors)) for index in indices]
    return nodes, weights
end

quadrature(domain; n=100) = quadrature(Midpoint(n), domain)

eachpoint(points::AbstractMatrix) = eachcol(points)
eachpoint(points::AbstractVector) = points

function integrate(f, domain::DomainSets.Domain, rule::Midpoint)
    nodes, weights = quadrature(rule, domain)
    return sum(f(x) * weight for (x, weight) in zip(eachpoint(nodes), weights))
end

integrate(f, states::AbstractVector, ::QuadratureRule) = sum(f, states)

integrate(f, domain::DomainSets.Domain, rule::Adaptive) =
    first(HCubature.hcubature(f, vcat(DomainSets.infimum(domain)), vcat(DomainSets.supremum(domain)); rtol=rule.rtol, maxevals=rule.maxevals))
