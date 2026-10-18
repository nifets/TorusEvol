@testsnippet Common begin
    using Random, Statistics, Distributions, LinearAlgebra, LogExpFunctions
    const TEST_SEED = 123456
    Random.seed!(TEST_SEED)
end
