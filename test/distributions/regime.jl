site_statistics = [[x -> x[1] == regime for regime in 1:2];
                   [x -> x[2] == state for state in (1, 8, 14)];
                   [x -> cos(x[3][1]), x -> sin(x[3][2]), x -> cos(x[3][1] - x[3][2])]]

@testset "regime process" begin
    Random.seed!(TEST_SEED)
    helix = (WAG_substitution_process(), WrappedDiffusion(-1.0, -0.8, 1.0, 1.0, 1.0, 1.0, 0.5))
    sheet = (WAG_substitution_process(; r=2.0), WrappedDiffusion(-2.0, 2.2, 1.0, 3.0, 1.0, 2.0, 0.5))
    grid = Midpoint(40)
    rule = Midpoint(40)
    ts = (0.5, 2.0)

    product = ProductProcess(helix...)
    x = rand(statdist(product))
    @test logpdf(statdist(product), x) ≈ logpdf(statdist(helix[1]), x[2]) + logpdf(statdist(helix[2]), x[3])
    @test logpdf(transdist(product, 0.5, x), x) ≈
          logpdf(transdist(helix[1], 0.5, x[2]), x[2]) + logpdf(transdist(helix[2], 0.5, x[3]), x[3])

    frozen = RegimeProcess(frozen_regimes([0.3, 0.7]), [helix, sheet])
    test_process(frozen; statistics=site_statistics, ts, rule, grid, converges=false, stationary_rtol=1e-3, semigroup_rtol=0.15)

    switching = RegimeProcess(switching_regimes([0.3, 0.7], 0.7), [helix, sheet])
    test_process(switching; statistics=site_statistics, ts, rule, grid, stationary_rtol=1e-3, semigroup_rtol=0.15)

    instant = RegimeProcess(switching_regimes([0.3, 0.7], 1e6), [helix, sheet])
    nodes, _ = quadrature(Midpoint(10), domain(instant))
    @test batchlogpdf(transdist(instant, 0.5, x), nodes) ≈ batchlogpdf(statdist(instant), nodes)

    @test_throws DimensionMismatch RegimeProcess(frozen_regimes([1.0]), [helix, sheet])
end
