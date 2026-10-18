@testitem "substitution process properties" setup=[Common, Properties] tags=[:property] begin
    test_process(WAG_substitution_process(); statistics=[x -> x == state for state in 1:20], semigroup_rtol=1e-8)
end

@testitem "substitution process rates and joint" setup=[Common] begin
    process = WAG_substitution_process()
    @test TorusEvol.transmat(process, 0.0) ≈ I
    @test -dot(diag(process.Q), probs(statdist(process))) ≈ 1
    @test -dot(diag(WAG_substitution_process(; r=2.5).Q), probs(statdist(process))) ≈ 2.5
    states = collect(1:20)
    joint = zeros(20, 20); jointlogpdf!(joint, process, 0.5, states, states)
    @test joint ≈ [logpdf(statdist(process), x) + logpdf(transdist(process, 0.5, x), y) for x in states, y in states]
    @test_throws ArgumentError TorusEvol.CTMC([0.5, 0.5], [1.0 -1.0; 1.0 -1.0])
end
