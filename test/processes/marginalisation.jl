fallback_logpdf(p, ts, ys; rule) =
    invoke(descendants_logpdf, Tuple{TorusEvol.AbstractProcess, Any, Any}, p, ts, ys; rule)

fallback_posterior(p, ts, ys; rule) =
    invoke(ancestral_posterior, Tuple{TorusEvol.AbstractProcess,Any,Any}, p, ts, ys; rule)

@testset "marginalisation" begin
    Random.seed!(TEST_SEED)

    ts = [0.4, 1.0, 2.5]

    substitution = WAG_substitution_process()
    states = [3, 3, 17]
    P = [TorusEvol.transmat(substitution, t) for t in ts]
    bruteforce = log(sum(probs(statdist(substitution))[x] * prod(P[k][x, states[k]] for k in eachindex(ts)) for x in 1:20))
    @test descendants_logpdf(substitution, ts, states) ≈ bruteforce
    @test descendants_logpdf(substitution, Float64[], Int[]) ≈ 0 atol=1e-12
    @test sum(last(ancestral_posterior(substitution, ts, states))) ≈ 1

    ou_iso = OrnsteinUhlenbeck([0.5, -1.0], 1.5, 0.7)
    ys = [rand(transdist(ou_iso, t, [0.0, 0.0])) for t in ts]
    @test descendants_logpdf(ou_iso, ts, ys) ≈ fallback_logpdf(ou_iso, ts, ys; rule=Grid(30)) rtol=1e-8
    @test descendants_logpdf(ou_iso, ts[1:1], ys[1:1]) ≈ logpdf(statdist(ou_iso), ys[1])

    posterior = ancestral_posterior(ou_iso, ts, ys)
    nodes, weights = fallback_posterior(ou_iso, ts, ys; rule=Grid(30))
    grid_mean = sum(w.* x for (x,w) in zip(nodes, weights))
    grid_cov = sum(w .* (x .- grid_mean) * (x .- grid_mean)' for (x,w) in zip(nodes, weights))
    @test mean(posterior) ≈ grid_mean rtol=1e-8
    @test cov(posterior) ≈ grid_cov rtol=1e-8

    ou = OrnsteinUhlenbeck(ou_iso.μ, 2 * 0.7 * 1.5 * Matrix(1.0I, 2, 2), 0.7 * Matrix(1.0I, 2, 2))
    @test fallback_logpdf(ou, ts, ys; rule=Grid(30)) ≈ descendants_logpdf(ou_iso, ts, ys) rtol=1e-8
end
