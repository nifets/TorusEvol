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

include("processes/properties.jl")

const all_tests = ["processes/torus", "processes/substitution", "processes/regime", "processes/ou", "processes/marginalisation"]

@testset "TorusEvol" begin
    @testset "Test $t" for t in all_tests
        include("$t.jl")
    end
end
