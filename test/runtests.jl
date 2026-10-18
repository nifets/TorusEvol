# Test style inspired by the Distributions.jl package
using TorusEvol
using Test
using Random
using Statistics
using Distributions
using LogExpFunctions
using LinearAlgebra

const TEST_SEED = 123456
printstyled("Running tests:\n", color=:blue)

include("distributions/properties.jl")

const all_tests = ["distributions/torus", "distributions/substitution"]

@testset "TorusEvol" begin
    @testset "Test $t" for t in all_tests
        include("$t.jl")
    end
end
