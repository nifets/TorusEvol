module TorusEvol

using Statistics
using Random
using Distributions
using PDMats
import DomainSets
import HCubature

using LinearAlgebra
using LogExpFunctions

include("distributions/AbstractProcess.jl")
include("quadrature.jl")
include("distributions/WrappedNormal.jl")
include("distributions/WrappedDiffusion.jl")
include("distributions/ResettingProcess.jl")
include("distributions/CTMC.jl")
include("distributions/TupleProduct.jl")
include("distributions/RegimeProcess.jl")

include("models/WAG.jl")

export
    WrappedNormal, WrappedDiffusion, ResettingProcess, substitution_process,
    statdist, transdist, transdist!, jointlogpdf!,
    TupleProduct, RegimeProcess, ProductProcess, frozen_regimes, switching_regimes,
    domain, Midpoint, Adaptive, quadrature, integrate,
    WAG_substitution_process

end #TorusEvol
