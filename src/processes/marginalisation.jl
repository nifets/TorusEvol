function ancestral_logjoint(p::AbstractProcess, ts, descendants; rule=Grid())
    nodes, weights = quadrature(rule, domain(p))
    logps = [log(w) + logpdf(statdist(p), x) +
             sum(logpdf(transdist(p, t, x), y) for (t, y) in zip(ts, descendants); init=0.0)
             for (x, w) in zip(eachpoint(nodes), weights)]
    return collect(eachpoint(nodes)), logps
end

descendants_logpdf(p::AbstractProcess, ts, descendants; rule=Grid()) =
    logsumexp(last(ancestral_logjoint(p, ts, descendants; rule)))

function ancestral_posterior(p::AbstractProcess, ts, descendants; rule=Grid())
    nodes, logps = ancestral_logjoint(p, ts, descendants; rule)
    return nodes, softmax(logps)
end

function descendants_logpdf(p::IsotropicOU, ts, descendants; rule=nothing)
    decays = exp.(-p.A.λ .* ts)
    var = p.statcov.value
    cov = var .* (decays * decays')
    cov[diagind(cov)] .= var
    d = MvNormal(PDMat(Symmetric(cov)))
    return sum(logpdf(d, getindex.(descendants, i) .- p.μ[i]) for i in eachindex(p.μ))
end

function ancestral_posterior(p::IsotropicOU, ts, descendants; rule=nothing)
    decays = exp.(-p.A.λ .* ts)
    var = p.statcov.value
    precisions = decays .^ 2 ./ (var .* (1 .- decays .^ 2))
    precision = 1 / var + sum(precisions)
    shift = sum(λ / b .* (y .- p.μ) for (λ, b, y) in zip(precisions, decays, descendants))
    return MvNormal(p.μ .+ shift ./ precision, ScalMat(length(p), 1 / precision))
end
