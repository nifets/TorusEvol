using ForwardDiff
using FiniteDifferences

torus_statistics(d) = let dim = length(d)
    [[x -> f(x[i]) for i in 1:dim for f in (cos, sin)];
     [x -> f(x[i] - x[j]) for i in 1:dim for j in i+1:dim for f in (cos, sin)]]
end

function test_periodic(d; n=10)
    for x in eachcol(rand(d, n))
        k = rand(-2:2, length(d))
        @test logpdf(d, x .+ 2π .* k) ≈ logpdf(d, x)
    end
end

function test_concentrates(process; t=1e-6, n=200, tol=0.1)
    x₀ = rand(statdist(process))
    samples = rand(transdist(process, t, x₀), n)
    @test maximum(abs, TorusEvol.cmod(samples .- x₀)) < tol
end

function accurate_logpdf(wn::WrappedNormal, X::AbstractMatrix{<:Real})
    d = size(X, 1); n = size(X, 2); m = size(wn.𝕃, 2)
    a = reshape(X, d, n, :) .+ reshape(wn.𝕃, d, :, m)
    lps = reshape(logpdf(wn.𝛷, reshape(a, d, :)), n, m)
    vec(logsumexp(lps; dims=2))
end

@testset "WrappedNormal" begin
    @testset "randomized parameters" begin
        for seed in 1:30
            rng = Xoshiro(TEST_SEED + seed)
            μ = rand(rng, Uniform(-π, π), 2)
            σ = sqrt.(rand(rng, InverseGamma(), 2)) ./ π
            Σ = Hermitian(σ .* rand(rng, LKJ(2, 1.0)) .* σ')
            wn = WrappedNormal(μ, Σ)
            x = rand(rng, Uniform(-π, π), 2, 50)
            expected = accurate_logpdf(wn, x)
            result = similar(expected)
            @test logpdf!(result, wn, x) === result
            @test result ≈ expected atol=1e-12
            test_normalised(wn; atol=1e-4)
        end
    end

    @testset "generic properties" begin
        Random.seed!(TEST_SEED)
        wn = WrappedNormal([0.4, -1.2], [1.2 0.4; 0.4 0.8])
        test_distribution(wn; statistics=torus_statistics(wn))
        test_periodic(wn)
    end

    @testset "lattice tolerance" begin
        tol = 1e-2
        wn = WrappedNormal([0.4, -1.2], [1.2 0.4; 0.4 0.8]; tol)
        mass = integrate(x -> pdf(wn, x), domain(wn), Grid(300))
        @test 1 - tol - 2e-4 <= mass <= 1 + 2e-4
    end

    @testset "AD" begin
        wn = WrappedNormal([0.2, -0.4], [0.8 0.2; 0.2 1.1]; tol=1e-12)
        f = x -> logpdf(wn, x)
        x = [0.2, -0.1]
        @test ForwardDiff.gradient(f, x) ≈ grad(central_fdm(5, 1), f, x)[1] rtol=1e-6
    end
end

@testset "WrappedDiffusion" begin
    Random.seed!(TEST_SEED)
    for process in (WrappedDiffusion(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5),
                    WrappedDiffusion(0.5, -1.0, 1.0, 3.0, 1.0, 2.0, 0.5))
        test_process(process; statistics=torus_statistics(process), semigroup_rtol=0.15)
        test_periodic(statdist(process))
        test_periodic(transdist(process, 0.5, rand(statdist(process))))
        test_concentrates(process)
        for t in (0.01, 0.5, 3.0)
            @test TorusEvol.drift_coefficient(process, t) ≈ exp(-t * process.A)
        end
        @test Matrix(process.statcov) ≈ 0.5 * inv(process.A) * Matrix(process.Σ)
    end
    repeated = WrappedDiffusion([0.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
    @test TorusEvol.drift_coefficient(repeated, 0.5) ≈ exp(-0.5 * repeated.A)
    @test_throws ArgumentError WrappedDiffusion(0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 0.5)
    @test_throws ArgumentError WrappedDiffusion(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0)
end

@testset "ResettingProcess" begin
    Random.seed!(TEST_SEED)
    raw = WrappedDiffusion(0.5, -1.0, 1.0, 3.0, 1.0, 2.0, 0.5)
    process = TorusEvol.resetting(raw, 0.7)
    test_process(process; statistics=torus_statistics(process), semigroup_rtol=0.15)
    test_periodic(transdist(process, 0.5, rand(statdist(process))))
    test_concentrates(process)

    nodes, _ = quadrature(domain(raw))
    x₀ = [1.0, -1.0]
    @test logpdf(transdist(TorusEvol.resetting(raw, 0.0), 0.5, x₀), nodes) ≈
          logpdf(transdist(raw, 0.5, x₀), nodes)
    @test logpdf(transdist(TorusEvol.resetting(raw, 1e6), 0.5, x₀), nodes) ≈
          logpdf(statdist(raw), nodes)

    stat = statdist(raw)
    X, Y = rand(stat, 5), rand(stat, 4)
    joint = zeros(5, 4); jointlogpdf!(joint, process, 0.5, X, Y)
    @test joint ≈ [logpdf(stat, y) + logpdf(transdist(process, 0.5, y), x)
                   for x in eachcol(X), y in eachcol(Y)]
end
