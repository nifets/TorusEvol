gaussian_statistics = [x -> x[1], x -> x[2]^2, x -> x[1] * x[2]]

@testset "ornstein uhlenbeck" begin
    Random.seed!(TEST_SEED)
    ts = (0.3, 1.0, 3.0)

    C = [1.0 0.3; 0.3 0.5]
    S = [1.0 0.2; 0.2 0.8]
    general = OrnsteinUhlenbeck([0.5, -1.0], 2S, S / C)
    @test cov(statdist(general)) ≈ C
    test_process(general; statistics=gaussian_statistics, ts, rule=Grid(30), stationary_rtol=1e-6, semigroup_rtol=1e-6)

    var, rate = 1.5, 0.7
    isotropic = OrnsteinUhlenbeck([0.5, -1.0, 2.0], var, rate)
    test_process(isotropic; statistics=gaussian_statistics, ts, rule=Grid(30), grid=Grid(20), stationary_rtol=1e-6, semigroup_rtol=1e-6)

    equivalent = OrnsteinUhlenbeck(isotropic.μ, 2rate * var * Matrix(1.0I, 3, 3), rate * Matrix(1.0I, 3, 3))
    x = rand(statdist(isotropic))
    @test cov(statdist(equivalent)) ≈ cov(statdist(isotropic))
    for t in ts
        @test mean(transdist(isotropic, t, x)) ≈ isotropic.μ .+ exp(-rate * t) .* (x .- isotropic.μ)
        @test cov(transdist(isotropic, t, x)) ≈ var * (1 - exp(-2rate * t)) * I
        @test mean(transdist(equivalent, t, x)) ≈ mean(transdist(isotropic, t, x))
        @test cov(transdist(equivalent, t, x)) ≈ cov(transdist(isotropic, t, x))
    end

    @test_throws ArgumentError integrate(x -> 1.0, domain(isotropic), Adaptive())
    @test_throws DimensionMismatch OrnsteinUhlenbeck([0.0], Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2))
end
