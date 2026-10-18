abstract type QuadratureRule end

struct Grid <: QuadratureRule
    n::Int
end
Grid() = Grid(100)

struct Adaptive <: QuadratureRule
    rtol::Float64
    maxevals::Int
end
Adaptive(; rtol=1e-6, maxevals=10^6) = Adaptive(rtol, maxevals)

struct PeriodicInterval{T} <: DomainSets.Domain{T}
    interval::DomainSets.ClosedInterval{T}
end
PeriodicInterval(a, b) = PeriodicInterval(DomainSets.ClosedInterval(float(a), float(b)))

DomainSets.infimum(d::PeriodicInterval) = DomainSets.infimum(d.interval)
DomainSets.supremum(d::PeriodicInterval) = DomainSets.supremum(d.interval)
DomainSets.width(d::PeriodicInterval) = DomainSets.width(d.interval)
DomainSets.leftendpoint(d::PeriodicInterval) = DomainSets.leftendpoint(d.interval)
Base.in(x, d::PeriodicInterval) = in(x, d.interval)

struct GaussianDomain{D<:MvNormal}
    reference::D
end

function quadrature(rule::Grid, interval::PeriodicInterval)
    spacing = DomainSets.width(interval) / rule.n
    nodes = permutedims(DomainSets.leftendpoint(interval) .+ ((0:rule.n-1) .+ 0.5) .* spacing)
    return nodes, fill(spacing, rule.n)
end

function quadrature(rule::Grid, domain::GaussianDomain)
    z, w = FastGaussQuadrature.gausshermite(rule.n)
    d = length(domain.reference)
    grid = vec(collect(Iterators.product(ntuple(_ -> eachindex(z), d)...)))
    L = cholesky(cov(domain.reference)).L
    nodes = reduce(hcat, [mean(domain.reference) .+ sqrt(2) .* (L * z[collect(i)]) for i in grid])
    weights = [prod(w[collect(i)]) / π^(d / 2) / pdf(domain.reference, x) for (i, x) in zip(grid, eachcol(nodes))]
    return nodes, weights
end

quadrature(::QuadratureRule, states::AbstractVector) = states, ones(length(states))

function quadrature(rule::Grid, product::DomainSets.ProductDomain)
    factors = [quadrature(rule, component) for component in DomainSets.components(product)]
    indices = vec(collect(Iterators.product((axes(nodes, 2) for (nodes, _) in factors)...)))
    nodes = reduce(vcat, [factor_nodes[:, getindex.(indices, k)] for (k, (factor_nodes, _)) in enumerate(factors)])
    weights = [prod(factor_weights[index[k]] for (k, (_, factor_weights)) in enumerate(factors)) for index in indices]
    return nodes, weights
end

quadrature(domain; n=100) = quadrature(Grid(n), domain)

eachpoint(points::AbstractMatrix) = eachcol(points)
eachpoint(points::AbstractVector) = points

function quadrature(rule::Grid, domains::Tuple)
    factors = map(component -> quadrature(rule, component), domains)
    grid = vec(collect(Iterators.product(map(((nodes, weights),) -> zip(eachpoint(nodes), weights), factors)...)))
    return [map(first, point) for point in grid], [prod(map(last, point)) for point in grid]
end

function integrate(f, domain::Union{DomainSets.Domain,GaussianDomain,Tuple}, rule::Grid)
    nodes, weights = quadrature(rule, domain)
    return sum(f(x) * weight for (x, weight) in zip(eachpoint(nodes), weights))
end

integrate(f, states::AbstractVector, ::QuadratureRule) = sum(f, states)

integrate(f, domain::DomainSets.Domain, rule::Adaptive) =
    first(HCubature.hcubature(f, vcat(DomainSets.infimum(domain)), vcat(DomainSets.supremum(domain)); rtol=rule.rtol, maxevals=rule.maxevals))

integrate(f, ::GaussianDomain, ::Adaptive) = throw(ArgumentError("use Grid for GaussianDomain"))

integrate(f, domains::Tuple{Any}, rule::Adaptive) = integrate(x -> f((x,)), only(domains), rule)

integrate(f, domains::Tuple, rule::Adaptive) =
    integrate(x -> integrate(rest -> f((x, rest...)), Base.tail(domains), rule), first(domains), rule)
