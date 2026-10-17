function test_normalised(d; rule=Midpoint(), atol=1e-6)
    @test integrate(x -> pdf(d, x), domain(d), rule) ≈ 1 atol=atol
end

function test_batch_logpdf(d; n=20)
    points = rand(d, n)
    @test logpdf(d, points) ≈ [logpdf(d, x) for x in eachcol(points)]
end

function test_sampler(d; statistics, n=20_000, σ=5)
    nodes, weights = quadrature((domain(d)))
    masses = exp.(logpdf(d, nodes)) .* weights
    samples = rand(d, n)
    for stat in statistics
        values = stat.(eachcol(nodes))
        expected = sum(values .* masses)
        variance = sum(values .^ 2 .* masses) - expected^2
        @test mean(stat, eachcol(samples)) ≈ expected atol=σ*sqrt(max(variance, 0) / n) + 1e-10
    end
end

function test_distribution(d; statistics)
    test_normalised(d)
    test_batch_logpdf(d)
    test_sampler(d; statistics)
end

function test_reversible(process, ts; n=5, atol=1e-8)
    stat = statdist(process)
    for t in ts, (x, y) in zip(eachcol(rand(stat, n)), eachcol(rand(stat, n)))
        forward = logpdf(stat, x) + logpdf(transdist(process, t, x), y)
        backward = logpdf(stat, y) + logpdf(transdist(process, t, y), x)
        @test forward ≈ backward atol=atol
    end
end

function test_stationary(process, ts; n=3, rule=Adaptive(), rtol=1e-4)
    stat = statdist(process)
    for t in ts, y in eachcol(rand(stat, n))
        mass = integrate(x -> pdf(stat, x) * pdf(transdist(process, t, x), y), domain(process), rule)
        @test mass ≈ pdf(stat, y) rtol=rtol
    end
end

function test_converges(process; t=50.0, atol=1e-6)
    nodes, _ = quadrature(domain(process))
    stat = statdist(process)
    @test logpdf(transdist(process, t, rand(stat)), nodes) ≈ logpdf(stat, nodes) atol=atol
end

function test_semigroup(process, s, t; n=3, rule=Adaptive(), rtol=1e-4)
    stat = statdist(process)
    for x in eachcol(rand(stat, n))
        z = rand(transdist(process, s + t, x))
        composed = integrate(y -> pdf(transdist(process, s, x), y) * pdf(transdist(process, t, y), z),
                             domain(process), rule)
        @test composed ≈ pdf(transdist(process, s + t, x), z) rtol=rtol
    end
end

function test_process(process; statistics, ts=(0.1, 1.0, 5.0), t_converged=50.0,
                      semigroup_times=(0.3, 0.5), semigroup_rtol=1e-4)
    stat = statdist(process)
    test_distribution(stat; statistics)
    for t in ts
        test_distribution(transdist(process, t, rand(stat)); statistics)
    end
    test_reversible(process, ts)
    test_stationary(process, ts)
    test_converges(process; t=t_converged)
    test_semigroup(process, semigroup_times...; rtol=semigroup_rtol)
end
