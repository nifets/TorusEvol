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

function quadrature(rule::Midpoint, I::DomainSets.AbstractInterval)
    h = DomainSets.width(I) / rule.n
    X = permutedims(DomainSets.leftendpoint(I) .+ ((0:rule.n-1) .+ 0.5) .* h)
    return X, fill(h, rule.n)
end

quadrature(::Midpoint, S::AbstractVector; kw...) = permutedims(S), ones(length(S))

function quadrature(rule::Midpoint, D::DomainSets.ProductDomain)
    rules = [quadrature(rule, c) for c in DomainSets.components(D)]
    I = vec(collect(Iterators.product((axes(X, 2) for (X, _) in rules)...)))
    X = reduce(vcat, [Xk[:, getindex.(I, k)] for (k, (Xk, _)) in enumerate(rules)])
    w = [prod(wk[i[k]] for (k, (_, wk)) in enumerate(rules)) for i in I]
    return X, w
end

quadrature(D; n=100) = quadrature(Midpoint(n), D)

function integrate(f, D, rule::Midpoint)
    nodes, weights = quadrature(rule, D)
    return sum(f(x) * w for (x, w) in zip(eachcol(nodes), weights))
end

integrate(f, states::AbstractVector, ::Adaptive) = sum(s -> f([s]), states)

box(interval::DomainSets.AbstractInterval) = [DomainSets.leftendpoint(interval)], [DomainSets.rightendpoint(interval)]
function box(D::DomainSets.ProductDomain)
    boxes = box.(DomainSets.components(D))
    return reduce(vcat, first.(boxes)), reduce(vcat, last.(boxes))
end

integrate(f, D, rule::Adaptive) = first(HCubature.hcubature(f, box(D)...; rtol=rule.rtol, maxevals=rule.maxevals))
