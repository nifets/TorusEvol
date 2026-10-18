@testmodule Properties begin
    using TorusEvol, Test, Distributions, Statistics

    export batchlogpdf, test_normalised, test_batch_logpdf, test_sampler, test_distribution,
           test_reversible, test_stationary, test_converges, test_semigroup, test_process

    batchlogpdf(d::UnivariateDistribution, points) = logpdf.(d, points)
    batchlogpdf(d::MultivariateDistribution, points) = logpdf(d, points)
    batchlogpdf(d::Distribution{TorusEvol.TupleVariate}, points) = logpdf.(Ref(d), points)

    function test_normalised(d; rule=Grid(), atol=1e-6)
        @test integrate(x -> pdf(d, x), domain(d), rule) ≈ 1 atol=atol
    end

    function test_batch_logpdf(d; n=20)
        points = rand(d, n)
        @test batchlogpdf(d, points) ≈ [logpdf(d, x) for x in TorusEvol.eachpoint(points)]
    end

    function test_sampler(d; statistics, grid=Grid(), n=20_000, σ=5)
        nodes, weights = quadrature(grid, domain(d))
        masses = exp.(batchlogpdf(d, nodes)) .* weights
        samples = rand(d, n)
        for stat in statistics
            values = stat.(TorusEvol.eachpoint(nodes))
            expected = sum(values .* masses)
            variance = sum(values .^ 2 .* masses) - expected^2
            @test mean(stat, TorusEvol.eachpoint(samples)) ≈ expected atol=σ*sqrt(max(variance, 0) / n) + 1e-10
        end
    end

    function test_distribution(d; statistics, grid=Grid())
        test_normalised(d; rule=grid)
        test_batch_logpdf(d)
        test_sampler(d; statistics, grid)
    end

    function test_reversible(process, ts; n=5, atol=1e-8)
        stat = statdist(process)
        for t in ts, (x, y) in zip(TorusEvol.eachpoint(rand(stat, n)), TorusEvol.eachpoint(rand(stat, n)))
            forward = logpdf(stat, x) + logpdf(transdist(process, t, x), y)
            backward = logpdf(stat, y) + logpdf(transdist(process, t, y), x)
            @test forward ≈ backward atol=atol
        end
    end

    function test_stationary(process, ts; n=3, rule=Adaptive(), rtol=1e-4)
        stat = statdist(process)
        for t in ts, y in TorusEvol.eachpoint(rand(stat, n))
            mass = integrate(x -> pdf(stat, x) * pdf(transdist(process, t, x), y), domain(process), rule)
            @test mass ≈ pdf(stat, y) rtol=rtol
        end
    end

    function test_converges(process; t=50.0, grid=Grid(), atol=1e-6)
        nodes, _ = quadrature(grid, domain(process))
        stat = statdist(process)
        @test batchlogpdf(transdist(process, t, rand(stat)), nodes) ≈ batchlogpdf(stat, nodes) atol=atol
    end

    function test_semigroup(process, s, t; n=3, rule=Adaptive(), rtol=1e-4)
        stat = statdist(process)
        for x in TorusEvol.eachpoint(rand(stat, n))
            z = rand(transdist(process, s + t, x))
            composed = integrate(y -> pdf(transdist(process, s, x), y) * pdf(transdist(process, t, y), z),
                                 domain(process), rule)
            @test composed ≈ pdf(transdist(process, s + t, x), z) rtol=rtol
        end
    end

    function test_process(process; statistics, ts=(0.1, 1.0, 5.0), t_converged=50.0,
                          rule=Adaptive(), grid=Grid(), converges=true, stationary_rtol=1e-4,
                          semigroup_times=(0.3, 0.5), semigroup_rtol=1e-4)
        stat = statdist(process)
        test_distribution(stat; statistics, grid)
        for t in ts
            test_distribution(transdist(process, t, rand(stat)); statistics, grid)
        end
        test_reversible(process, ts)
        test_stationary(process, ts; rule, rtol=stationary_rtol)
        converges && test_converges(process; t=t_converged, grid)
        test_semigroup(process, semigroup_times...; rule, rtol=semigroup_rtol)
    end
end
