using TorusEvol
using Distributions
using Turing
using Random
using LogExpFunctions
using LinearAlgebra

@model function bisection_sampler(p, descs, t)
    D = length(descs)
    descs[1] ~ statdist(p)

    x = Vector{Real}(undef, length(p))
    x ~ transdist(p, t, descs[1])
    for d ∈ 2:D
        descs[d] ~ transdist(p, t, x)
    end
end

@model function bisection_sampler_sub(p, descs, t)
    D = length(descs)
    descs[1] ~ statdist(p)

    x ~ transdist(p, t, descs[1])
    for d ∈ 2:D
        descs[d] ~ transdist(p, t, x)
    end
end


function sample_anc_coords_wn(p, descs, t; burn_in=1000)
    sampler = MH(:x => v -> MixtureModel([WrappedNormal(v, I), WrappedNormal(v, 20*I)], [0.8, 0.2]))
    n = size(descs[1], 2)
    D = length(descs)
    X = Matrix{Real}(undef, length(p), n)
    for i ∈ 1:n
        descs_i = [descs[d][:, i] for d ∈ 1:D]
        model = bisection_sampler(p, descs_i, t)
        chn = sample(model, sampler, burn_in+1)
        X[1, i] = get(chn, :x).x[1][burn_in+1]
        X[2, i] = get(chn, :x).x[2][burn_in+1]
    end
    return X
end

function sample_anc_coords_sub(p, descs, t; burn_in=10)
    sampler = PG(30, :x)
    n = size(descs[1], 2)
    D = length(descs)
    X = Matrix{Real}(undef, length(p), n)
    for i ∈ 1:n
        descs_i = [descs[d][1, i] for d ∈ 1:D]
        model = bisection_sampler_sub(p, descs_i, t)
        chn = sample(model, sampler, burn_in+1)
        X[1, i] = get(chn, :x).x[burn_in+1]
    end
    return X
end

function sample_anc_coords_sub2(p, descs, t; burn_in=5)
    n = size(descs[1], 2)
    X = Matrix{Real}(undef, length(p), n)
    for i ∈ 1:n
        X[1, i] = 1
    end
    return X
end

function sample_anc_coords(p, descs, t)
    if eltype(p) <: Integer
        return sample_anc_coords_sub(p, descs, t)
    else
        return sample_anc_coords_wn(p, descs, t)
    end
end

# compute emission lp for each site in alignment, given pairwise emission matrix
function logpdfs(alignment, emission_lps)
    res = Real[]
    N, M = size(emission_lps) .- 1
    i = 0
    j = 0
    for v ∈ alignment
        if v == [1, 1]
            i += 1
            j += 1
            push!(res, emission_lps[i, j])
        elseif v == [1, 0]
            i += 1
            push!(res, emission_lps[i, M+1])
        else
            j += 1
            push!(res, emission_lps[N+1, j])
        end
    end
    return res
end

function sample_anc_alignment(M_YZ, Y, Z, t, ξ, Λ)
    emission_lps = fulljointlogpdf(ξ, t, Y, Z)
    align_emission_lps = logpdfs(M_YZ, emission_lps)
    (λ, μ, r) = Λ
    anc_align_model = TKF92([t/2, t/2], λ, μ, r; known_ancestor=false)
    α = forward_anc(M_YZ, anc_align_model, align_emission_lps)
    M_XYZ = backward_sampling_anc(α, anc_align_model)
    return M_XYZ
end


function ancestor_sampling(M_XYZW::Alignment,
                           Y::ObservedChain, Z::ObservedChain, W::ObservedChain,
                           t::Real, ξ::MixtureProductProcess, Λ)
    @info "reconstructing ancestor...\n"
    C = num_coords(ξ)
    E = num_regimes(ξ)

    # step 2 - sample coordinates of X
    alignment = M_XYZW
    X_mask = mask(alignment, [[1], [0,1], [0,1], [0,1]])
    alignmentX = slice(alignment, X_mask)
    Y_mask = mask(alignment, [[0,1], [1], [0,1], [0,1]])
    alignmentY = slice(alignment, Y_mask)
    Z_mask = mask(alignment, [[0,1], [0,1], [1], [0,1]])
    alignmentZ = slice(alignment, Z_mask)
    W_mask = mask(alignment, [[0,1], [0,1], [0,1], [1]])
    alignmentW = slice(alignment, W_mask)

    M = length(alignment)
    regimes = ones(M)

    X_maskX = mask(alignmentX, [[1], [0], [0], [0]])

    XY_maskX = mask(alignmentX, [[1], [1], [0], [0]])
    XY_maskY = mask(alignmentY, [[1], [1], [0], [0]])

    XZ_maskX = mask(alignmentX, [[1], [0], [1], [0]])
    XZ_maskZ = mask(alignmentZ, [[1], [0], [1], [0]])

    XW_maskX = mask(alignmentX, [[1], [0], [0], [1]])
    XW_maskW = mask(alignmentW, [[1], [0], [0], [1]])

    XYZ_maskX = mask(alignmentX, [[1], [1], [1], [0]])
    XYZ_maskY = mask(alignmentY, [[1], [1], [1], [0]])
    XYZ_maskZ = mask(alignmentZ, [[1], [1], [1], [0]])

    XYW_maskX = mask(alignmentX, [[1], [1], [0], [1]])
    XYW_maskY = mask(alignmentY, [[1], [1], [0], [1]])
    XYW_maskW = mask(alignmentW, [[1], [1], [0], [1]])

    XZW_maskX = mask(alignmentX, [[1], [0], [1], [1]])
    XZW_maskZ = mask(alignmentZ, [[1], [0], [1], [1]])
    XZW_maskW = mask(alignmentW, [[1], [0], [1], [1]])

    XYZW_maskX = mask(alignmentX, [[1], [1], [1], [1]])
    XYZW_maskY = mask(alignmentY, [[1], [1], [1], [1]])
    XYZW_maskZ = mask(alignmentZ, [[1], [1], [1], [1]])
    XYZW_maskW = mask(alignmentW, [[1], [1], [1], [1]])

    dataY = data(Y)
    dataZ = data(Z)
    dataW = data(W)

    # Initialise internal coordinates of X
    N = sequence_lengths(M_XYZW)[1]
    dataX = [similar(dataY[c], size(dataY[c], 1), N) for c ∈ 1:C]

    regimesX = regimes[X_mask]
    regimesY = regimes[Y_mask]
    regimesZ = regimes[Z_mask]
    regimesW = regimes[W_mask]

    for c ∈ 1:C, e ∈ 1:E
        p = processes(ξ)[c, e]

        # [1, 0, 0, 0] - sample from stationary distribution
        if count(X_maskX) > 0
            dataX1000 = @view dataX[c][:, X_maskX .& (regimesX .== e)]
            dataX1000 .= 0
            n1000 = size(dataX1000, 2)
            for i ∈ 1:size(dataX1000, 2)
                dataX1000[:, i] .= rand(statdist(p))
            end
        end


        # [1, 1, 0, 0] - observe Y, then sample X from Y
        if count(XY_maskX) > 0
            dataY1100 = @view dataY[c][:, XY_maskY .& (regimesY .== e)]
            dataX1100 = @view dataX[c][:, XY_maskX .& (regimesX .== e)]

            dataX1100 .= sample_anc_coords(p, [dataY1100], t)
        end


        # [1, 0, 1, 0] - observe Z, then sample X from Z
        if count(XZ_maskX) > 0
            dataZ1010 = @view dataZ[c][:, XZ_maskZ .& (regimesZ .== e)]
            dataX1010 = @view dataX[c][:, XZ_maskX .& (regimesX .== e)]

            dataX1010 .= sample_anc_coords(p, [dataZ1010], t)
        end

        # [1, 0, 0, 1] - observe W, then sample X from W
        if count(XW_maskX) > 0
            dataW1001 = @view dataW[c][:, XW_maskW .& (regimesW .== e)]
            dataX1001 = @view dataX[c][:, XW_maskX .& (regimesX .== e)]

            dataX1001 .= sample_anc_coords(p, [dataW1001], t)
        end


        # [1, 1, 1, 0] - observe Y, sample X from Y, then observe Z from X
        if count(XYZ_maskX) > 0
            dataY1110 = @view dataY[c][:, XYZ_maskY .& (regimesY .== e)]
            dataZ1110 = @view dataZ[c][:, XYZ_maskZ .& (regimesZ .== e)]
            dataX1110 = @view dataX[c][:, XYZ_maskX .& (regimesX .== e)]

            dataX1110 .= sample_anc_coords(p, [dataY1110, dataZ1110], t)
        end

        # [1, 1, 0, 1] - observe Y, sample X from Y, then observe W from X
        if count(XYW_maskX) > 0
            dataY1101 = @view dataY[c][:, XYW_maskY .& (regimesY .== e)]
            dataW1101 = @view dataW[c][:, XYW_maskW .& (regimesW .== e)]
            dataX1101 = @view dataX[c][:, XYW_maskX .& (regimesX .== e)]

            dataX1101 .= sample_anc_coords(p, [dataY1101, dataW1101], t)
        end

        # [1, 0, 1, 1] - observe Z, sample X from Z, then observe W from X
        if count(XZW_maskX) > 0
            dataZ1011 = @view dataZ[c][:, XZW_maskZ .& (regimesZ .== e)]
            dataW1011 = @view dataW[c][:, XZW_maskW .& (regimesW .== e)]
            dataX1011 = @view dataX[c][:, XZW_maskX .& (regimesX .== e)]

            dataX1011 .= sample_anc_coords(p, [dataZ1011, dataW1011], t)
        end

        # [1, 1, 1, 1] - observe Y, sample X from Y, then observe Z,W from X
        if count(XYZW_maskX) > 0
            dataY1111 = @view dataY[c][:, XYZW_maskY .& (regimesY .== e)]
            dataZ1111 = @view dataZ[c][:, XYZW_maskZ .& (regimesZ .== e)]
            dataW1111 = @view dataW[c][:, XYZW_maskW .& (regimesW .== e)]
            dataX1111 = @view dataX[c][:, XYZW_maskX .& (regimesX .== e)]

            dataX1111 .= sample_anc_coords(p, [dataY1111, dataZ1111, dataW1111], t)
        end

    end

    #hack - change later
    dataX[1] = Int.(dataX[1])
    X = ObservedChain(dataX)
    return X, M_XYZW
end


function ancestor_sampling(M_YZ::Alignment,
                           Y::ObservedChain, Z::ObservedChain,
                           t::Real, ξ::MixtureProductProcess, Λ)
    @info "reconstructing ancestor...\n"
    C = num_coords(ξ)
    E = num_regimes(ξ)

    # step 1 - sample XYZ alignment
    M_XYZ = sample_anc_alignment(M_YZ, Y, Z, t, ξ, Λ)

    # step 2 - sample coordinates of X
    alignment = M_XYZ
    X_mask = mask(alignment, [[1], [0,1], [0,1]])
    alignmentX = slice(alignment, X_mask)
    Y_mask = mask(alignment, [[0,1], [1], [0,1]])
    alignmentY = slice(alignment, Y_mask)
    Z_mask = mask(alignment, [[0,1], [0,1], [1]])
    alignmentZ = slice(alignment, Z_mask)

    M = length(alignment)
    regimes = ones(M)

    X_maskX = mask(alignmentX, [[1], [0], [0]])

    XY_maskX = mask(alignmentX, [[1], [1], [0]])
    XY_maskY = mask(alignmentY, [[1], [1], [0]])

    XZ_maskX = mask(alignmentX, [[1], [0], [1]])
    XZ_maskZ = mask(alignmentZ, [[1], [0], [1]])

    XYZ_maskX = mask(alignmentX, [[1], [1], [1]])
    XYZ_maskY = mask(alignmentY, [[1], [1], [1]])
    XYZ_maskZ = mask(alignmentZ, [[1], [1], [1]])

    dataY = data(Y)
    dataZ = data(Z)

    # Initialise internal coordinates of X
    N = sequence_lengths(M_XYZ)[1]
    dataX = [similar(dataY[c], size(dataY[c], 1), N) for c ∈ 1:C]

    regimesX = regimes[X_mask]
    regimesY = regimes[Y_mask]
    regimesZ = regimes[Z_mask]

    for c ∈ 1:C, e ∈ 1:E
        p = processes(ξ)[c, e]

        # [1, 0, 0] - sample from stationary distribution
        if count(X_maskX) > 0
            dataX100 = @view dataX[c][:, X_maskX .& (regimesX .== e)]
            dataX100 .= 0
            n100 = size(dataX100, 2)
            for i ∈ 1:size(dataX100, 2)
                dataX100[:, i] .= rand(statdist(p))
            end
        end


        # [1, 1, 0] - observe Y, then sample X from Y
        if count(XY_maskX) > 0
            dataY110 = @view dataY[c][:, XY_maskY .& (regimesY .== e)]
            dataX110 = @view dataX[c][:, XY_maskX .& (regimesX .== e)]

            dataX110 .= sample_anc_coords(p, [dataY110], t)
        end


        # [1, 0, 1] - observe Z, then sample X from Z
        if count(XZ_maskX) > 0
            dataZ101 = @view dataZ[c][:, XZ_maskZ .& (regimesZ .== e)]
            dataX101 = @view dataX[c][:, XZ_maskX .& (regimesX .== e)]

            dataX101 .= sample_anc_coords(p, [dataZ101], t)
        end


        # [1, 1, 1] - observe Y, sample X from Y, then observe Z from X
        if count(XYZ_maskX) > 0
            dataY111 = @view dataY[c][:, XYZ_maskY .& (regimesY .== e)]
            dataZ111 = @view dataZ[c][:, XYZ_maskZ .& (regimesZ .== e)]
            dataX111 = @view dataX[c][:, XYZ_maskX .& (regimesX .== e)]

            dataX111 .= sample_anc_coords(p, [dataY111, dataZ111], t)
        end
    end

    #hack - change later
    dataX[1] = Int.(dataX[1])
    X = ObservedChain(dataX)
    return X, M_XYZ
end

function ancestor_sampling(Y::ObservedChain, M_XY::Alignment,
                           t::Real, ξ::MixtureProductProcess, Λ)
    @info "reconstructing ancestor...\n"
    C = num_coords(ξ)
    E = num_regimes(ξ)

    # step 2 - sample coordinates of X
    alignment = M_XY
    X_mask = mask(alignment, [[1], [0,1]])
    alignmentX = slice(alignment, X_mask)
    Y_mask = mask(alignment, [[0,1], [1]])
    alignmentY = slice(alignment, Y_mask)

    M = length(alignment)
    regimes = ones(M)

    X_maskX = mask(alignmentX, [[1], [0]])

    XY_maskX = mask(alignmentX, [[1], [1]])
    XY_maskY = mask(alignmentY, [[1], [1]])

    dataY = data(Y)

    # Initialise internal coordinates of X
    N = sequence_lengths(M_XY)[1]
    dataX = [similar(dataY[c], size(dataY[c], 1), N) for c ∈ 1:C]

    regimesX = regimes[X_mask]
    regimesY = regimes[Y_mask]

    for c ∈ 1:C, e ∈ 1:E
        p = processes(ξ)[c, e]

        # [1, 0] - sample from stationary distribution
        if count(X_maskX) > 0
            dataX100 = @view dataX[c][:, X_maskX .& (regimesX .== e)]
            dataX100 .= 0
            n100 = size(dataX100, 2)
            for i ∈ 1:size(dataX100, 2)
                dataX100[:, i] .= rand(statdist(p))
            end
        end


        # [1, 1] - observe Y, then sample X from Y
        if count(XY_maskX) > 0
            dataY110 = @view dataY[c][:, XY_maskY .& (regimesY .== e)]
            dataX110 = @view dataX[c][:, XY_maskX .& (regimesX .== e)]

            dataX110 .= sample_anc_coords(p, [dataY110], t)
        end
    end

    #hack - change later
    dataX[1] = Int.(dataX[1])
    X = ObservedChain(dataX)
    return X, M_XY
end

function trajectory_reconstruction(M_YZ::Alignment, Y::ObservedChain, Z::ObservedChain,
                                   ξ::MixtureProductProcess, t::Real, Λ; levels=1)
    # ______________________________________________________________________________
    # Base case
    if levels == 0
        return [Y, Z], M_YZ
    end

    # ______________________________________________________________________________
    # Recursion

    # 1. Sample X midpoint of Y and Z, as well as the
    #    alignment M_XYZ, given the alignment M_YZ
    X, M_XYZ = ancestor_sampling(M_YZ, Y, Z, t, ξ, Λ)

    # 2. Reconstruct trajectories recursively on each branch
    M_YX  = subalignment(M_XYZ, [2, 1])
    traj_YX, M_Y_toX = trajectory_reconstruction(M_YX, Y, X, ξ, t/2, Λ; levels=levels-1)
    M_XZ = subalignment(M_XYZ, [1, 3])
    traj_XZ, M_X_toZ = trajectory_reconstruction(M_XZ, X, Z, ξ, t/2, Λ; levels=levels-1)

    # Combine the trajectories and alignments
    traj = [traj_YX; traj_XZ[2:end]]
    M = glue(M_Y_toX, M_X_toZ)
    return traj, M
end

function write_trajectory(chains::AbstractVector{ObservedChain},
                          M::Alignment,
                          poly_Y::Polypeptide,
                          poly_Z::Polypeptide,
                          name::String)
    N = length(chains)

    alignment = Alignment(data(M), collect(1:N))
    polypeptides = Vector{Polypeptide}(undef, N)
    polypeptides[1] = poly_Y
    polypeptides[N] = poly_Z
    for i ∈ 2:(N-1)
        X = chains[i]
        M_XYZ = subalignment(alignment, [i, 1, N])
        polypeptides[i] = from_triple_alignment(poly_Y, poly_Z, X, M_XYZ)
    end

    names = [name * "_$i" for i ∈ 1:N]
    to_file.(polypeptides, names)

    return polypeptides
end
