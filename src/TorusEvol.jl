module TorusEvol

using Statistics
using Random
using Distributions
using PDMats
import DomainSets
import HCubature
import FastGaussQuadrature

using LinearAlgebra
using LogExpFunctions

include("processes/AbstractProcess.jl")
include("quadrature.jl")
include("processes/WrappedNormal.jl")
include("processes/WrappedDiffusion.jl")
include("processes/ResettingProcess.jl")
include("processes/CTMC.jl")
include("processes/TupleProduct.jl")
include("processes/RegimeProcess.jl")
include("processes/OrnsteinUhlenbeck.jl")
include("processes/marginalisation.jl")

include("models/WAG.jl")

export
    WrappedNormal, WrappedDiffusion, ResettingProcess, substitution_process,
    statdist, transdist, transdist!, jointlogpdf!,
    TupleProduct, RegimeProcess, ProductProcess, frozen_regimes, switching_regimes,
    domain, Grid, Adaptive, quadrature, integrate,
    WAG_substitution_process,
    OrnsteinUhlenbeck, descendants_logpdf, ancestral_posterior

end #TorusEvol
