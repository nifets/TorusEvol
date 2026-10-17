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
include("distributions/quadrature.jl")
include("distributions/WrappedNormal.jl")
include("distributions/WrappedDiffusion.jl")
include("distributions/ResettingProcess.jl")
include("distributions/CTMC.jl")

include("models/WAG.jl")

export
    WrappedNormal, WrappedDiffusion, ResettingProcess, SubstitutionProcess,
    statdist, transdist, transdist!, jointlogpdf!,
    domain, Midpoint, Adaptive, quadrature, integrate,
    WAG_SubstitutionProcess

end #TorusEvol
