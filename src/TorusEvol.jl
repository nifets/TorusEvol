module TorusEvol

using Distributions, Random

import Base: length, eltype, show, size, setindex!, getindex, IndexStyle
import Distributions: _logpdf, _logpdf!, mean, _rand!

using TimerOutputs
const to = TimerOutput()

export
    to,

    # utils/Backbone
    build_biochain_from_aminoacids_dihedrals,
    build_biochain_from_aminoacids_dihedrals_alignment,
    build_biochain_from_triple_alignment,
    id_to_aa,
    aa_to_id,
    aminoacids,
    #build_chain_from_alignment,

    # objects/Polypeptide
    Polypeptide,
    from_pdb,
    from_file,
    to_file,
    num_residues,
    num_coords,
    data,
    chain,
    render,
    from_triple_alignment,
    from_observed_chain,
    from_aligned_polypeptide,
    from_primary_dihedrals,

    # objects/Chains
    AbstractChain,
    ObservedChain,
    num_sites,
    num_coords,
    num_regimes,
    slice,
    HiddenChain,
    logprobs,
    domains,


    # objects/Alignment
    Alignment,
    Domino,
    ids,
    num_sequences,
    sequence_lengths,
    row_index,
    subalignment,
    combine,
    glue,
    mask,
    show_filled_alignment,
    alignment_path,
    triple_distance,
    pairwise_distance,
    distance,
    covariance_sum,


    # models/TKF92
    TKF92,
    MAX_NUM_DESCENDANTS,
    transmat,
    num_descendants,
    proper_state_ids,
    state_ids,
    states,
    num_states,
    state_values,
    descendant_values,

    # models/WAG
    WAG_S,
    WAG_Π,
    WAG_SubstitutionProcess,

    # distributions/AbstractProcess
    Domain,
    area,
    AbstractProcess,
    domain,
    statdist,
    transdist,
    transdist!,
    jointlogpdf!,
    translogpdf!,
    statlogpdf!,
    fulltranslogpdf!,
    fulljointlogpdf!,
    fulljointlogpdf,
    randjoint,
    randstat,
    randtrans,

    #distributions/AlignmentProcess
    AlignmentDistribution,
    ConditionedAlignmentDistribution,
    LengthEquilibriumDistribution,
    LED,

    #distributions/ChainProcess
    ChainJointDistribution,
    ChainTransitionDistribution,
    logpdf,
    logpdfα!,
    logpdfαB!,
    get_α,
    get_B,
    rand,
    ChainPair,

    # distributions/WrappedNormal
    WrappedNormal,
    cmod,
    ang_dist,
    unwrapped,
    lattice,
    _logpdf!,

    # distributions/WrappedDiffusion
    WrappedDiffusion,
    WrappedDiffusionNode,
    recenter,

    # distributions/JumpingProcess
    JumpingProcess,
    JumpingProcessNode,
    rate,
    raw_process,
    jumping,
    JumpingWrappedDiffusion,

    # distributions/CTMC.jl
    CTMC,
    SubstitutionProcess,
    jointlp,

    # distributions/MixtureProductProcess.jl
    MixtureProductProcess,
    ProductProcess,
    weights,
    processes,
    hiddenchain_from_alignment,

    # distributions/EvolHMM.jl
    PairDataHMM,
    forward!,
    backward_sampling,
    forward_anc,
    backward_sampling_anc,
    _logpdf,

    # inference/Priors.jl
    tkf92_prior,
    jwndiff_prior,
    ScaledBeta,
    CompetingExponential,

    # inference/Proposals.jl
    torus_proposal,
    mv_rw_proposal,
    rw_proposal,
    Θ_samplers,
    τ_samplers,
    t_samplers,

    #inference/AncestorSampling.jl
    bisection_sampler,
    bisection_sampler_sub,
    sample_anc_coords_wn,
    sample_anc_coords_sub,
    sample_anc_coords,
    logpdfs,
    sample_anc_alignment,
    ancestor_sampling,
    trajectory_reconstruction,
    write_trajectory


# Objects
include("objects/Alignment.jl")
include("objects/Backbone.jl")
include("objects/Chains.jl")
include("objects/Polypeptide.jl")

# Models
include("models/TKF92.jl")


# Distributions
include("distributions/AbstractProcess.jl")

include("distributions/Common.jl")
include("distributions/WrappedNormal.jl")
include("distributions/WrappedDiffusion.jl")
include("distributions/JumpingProcess.jl")
include("distributions/CTMC.jl")
include("models/WAG.jl")
include("distributions/MixtureProductProcess.jl")
include("distributions/EvolHMM.jl")

include("distributions/AlignmentProcess.jl")
include("distributions/ChainProcess.jl")

include("inference/Priors.jl")
include("inference/Proposals.jl")
include("inference/AncestorSampling.jl")






TorusEvol

end #TorusEvol




# using Turing, Zygote
# using MCMCChains
# include("distributions/WrappedNormal.jl")
# include("distributions/WrappedDiffusion.jl")
# Turing.setadbackend(:zygote)

# @model function PairEvol(X_a::AbstractArray,
#                          X_b::AbstractArray)
#     # Priors
#     t ~ Exponential(1)
#     μ_ϕ ~ Uniform(-π, π)
#     μ_ψ ~ Uniform(-π, π)
#     σ_ϕ² ~ Gamma(1, 1)
#     σ_ψ² ~ Gamma(1, 1)
#     α_ϕ ~ Gamma(1, 1)
#     α_ψ ~ Gamma(1, 1)
#     l = √(α_ϕ * α_ψ)
#     α_corr ~ truncated(Normal(0, 1), -1, 1)

#     diff = WrappedDiffusion([μ_ϕ, μ_ψ], √σ_ϕ², √σ_ψ², α_ϕ, α_ψ, l*α_corr)

#     n = size(X_a, 1)
#     # Observe chain a
#     for i ∈ 1:n
#         X_a[:, i] ~ statdist(diff)
#     end
#     # Observe chain b
#     for i ∈ 1:n
#         X_b[:, i] ~ transdist(diff, t, X_a[:, i])
#     end

#     # Return sampled parameters
#     return t, μ_ϕ, μ_ψ, σ_ϕ², σ_ψ², α_ϕ, α_ψ, α_corr
# end

# @model function AncestorEvol(X_a::AbstractArray,
#                              X_b::AbstractArray)
#     # Priors
#     t ~ Exponential(1)
#     ratio ~ Uniform(0.000001, 1)
#     t1 = ratio*t
#     t2 = (1-ratio)*t
#     μ_ϕ ~ Uniform(-π, π)
#     μ_ψ ~ Uniform(-π, π)
#     σ_ϕ² ~ Gamma(1, 1)
#     σ_ψ² ~ Gamma(1, 1)
#     α_ϕ ~ Gamma(1, 1)
#     α_ψ ~ Gamma(1, 1)
#     l = √(α_ϕ * α_ψ)
#     α_corr ~ truncated(Normal(0, 1), -1, 1)

#     diff = WrappedDiffusion([μ_ϕ, μ_ψ], √σ_ϕ², √σ_ψ², α_ϕ, α_ψ, l*α_corr)

#     n = size(X_a, 1)
#     # Sample ancestor
#     A = zeros(2, n)
#     for i ∈ 1:n
#        A[:, i] ~ statdist(diff)
#     end
#     # Observe chain a
#     for i ∈ 1:n
#         X_a[:, i] ~ transdist(diff, t1, A[:, i])
#     end
#     # Observe chain b
#     for i ∈ 1:n
#         X_b[:, i] ~ transdist(diff, t2, A[:, i])
#     end

#     # Return sampled parameters
#     return t, μ_ϕ, μ_ψ, σ_ϕ², σ_ψ², α_ϕ, α_ψ, α_corr
# end

# @model function TreeEvol(X_a::AbstractArray,
#                          X_b::AbstractArray,
#                          X_c::AbstractArray)
#     # Priors
#     t1 ~ Exponential(1)
#     t2 ~ Exponential(1)
#     μ_ϕ ~ Uniform(-π, π)
#     μ_ψ ~ Uniform(-π, π)
#     σ_ϕ² ~ Gamma(1, 1)
#     σ_ψ² ~ Gamma(1, 1)
#     α_ϕ ~ Gamma(1, 1)
#     α_ψ ~ Gamma(1, 1)
#     l = √(α_ϕ * α_ψ)
#     α_corr ~ truncated(Normal(0, 1), -1, 1)

#     diff = WrappedDiffusion([μ_ϕ, μ_ψ], √σ_ϕ², √σ_ψ², α_ϕ, α_ψ, l*α_corr)

#     n = size(X_a, 1)
#     # Sample ancestor 1
#     A1 = zeros(2, n)
#     for i ∈ 1:n
#         A1[:, i] ~ statdist(diff)
#     end
#     # Observe chain a
#     for i ∈ 1:n
#         X_a[:, i] ~ transdist(diff, t1, A1[:, i])
#     end
#     # Sample ancestor 2
#     A2 = zeros(2, n)
#     for i ∈ 1:n
#         A2[:, i] ~ transdist(diff, t1, A1[:, i])
#     end
#     # Observe chain b
#     for i ∈ 1:n
#         X_b[:, i] ~ transdist(diff, t2, A2[:, i])
#     end
#     # Observe chain c
#     for i ∈ 1:n
#         X_c[:, i] ~ transdist(diff, t2, A2[:, i])
#     end

#     # Return sampled parameters
#     return diff
# end


# using BioStructures

# human_chain = read("data/1A3N.pdb", PDB)["A"]
# mouse_chain = read("data/3HRW.pdb", PDB)["A"]
# goose_chain = read("data/1hv4.pdb", PDB)["A"]

# angles(chain) = copy(transpose(hcat(filter(x -> !isnan(x), phiangles(chain)),
#                      filter(x -> !isnan(x), psiangles(chain)))))

# human_angles = angles(human_chain)
# mouse_angles = angles(mouse_chain)
# goose_angles = angles(goose_chain)

# model = TreeEvol(goose_angles, mouse_angles, human_angles)
# nsamples = 1000
# nthreads = 1
# samples = sample(model, MH(), MCMCThreads(), nsamples, nthreads)
# diff = generated_quantities(model, Turing.MCMCChains.get_sections(samples, :parameters))[nsamples]
