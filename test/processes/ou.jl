@testitem "OU general properties" setup=[Common, Properties] tags=[:property] begin
    C = [1.0 0.3; 0.3 0.5]
    S = [1.0 0.2; 0.2 0.8]
    general = OrnsteinUhlenbeck([0.5, -1.0], 2S, S / C)
    @test cov(statdist(general)) ≈ C
    test_process(general; statistics=[x -> x[1], x -> x[2]^2, x -> x[1] * x[2]], ts=(0.3, 1.0, 3.0),
                 rule=Grid(30), stationary_rtol=1e-6, semigroup_rtol=1e-6)
end

@testitem "OU isotropic properties" setup=[Common, Properties] tags=[:property] begin
    isotropic = OrnsteinUhlenbeck([0.5, -1.0, 2.0], 1.5, 0.7)
    test_process(isotropic; statistics=[x -> x[1], x -> x[2]^2, x -> x[1] * x[2]], ts=(0.3, 1.0, 3.0),
                 rule=Grid(30), grid=Grid(20), stationary_rtol=1e-6, semigroup_rtol=1e-6)
end

@testitem "OU isotropic closed form and equivalence" setup=[Common] begin
    variance, rate = 1.5, 0.7
    isotropic = OrnsteinUhlenbeck([0.5, -1.0, 2.0], variance, rate)
    equivalent = OrnsteinUhlenbeck(isotropic.μ, 2rate * variance * Matrix(1.0I, 3, 3), rate * Matrix(1.0I, 3, 3))
    x = rand(statdist(isotropic))
    @test cov(statdist(equivalent)) ≈ cov(statdist(isotropic))
    for t in (0.3, 1.0, 3.0)
        @test mean(transdist(isotropic, t, x)) ≈ isotropic.μ .+ exp(-rate * t) .* (x .- isotropic.μ)
        @test cov(transdist(isotropic, t, x)) ≈ variance * (1 - exp(-2rate * t)) * I
        @test mean(transdist(equivalent, t, x)) ≈ mean(transdist(isotropic, t, x))
        @test cov(transdist(equivalent, t, x)) ≈ cov(transdist(isotropic, t, x))
    end
end

@testitem "OU errors" setup=[Common] begin
    @test_throws ArgumentError integrate(x -> 1.0, domain(OrnsteinUhlenbeck([0.0, 0.0], 1.0, 1.0)), Adaptive())
    @test_throws DimensionMismatch OrnsteinUhlenbeck([0.0], Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2))
end
