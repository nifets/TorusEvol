using BioSymbols
include("PairEvol.jl")

function ang_dist(ϕ_x, ϕ_y, ψ_x, ψ_y)
    d = sqrt(4 - 2*cos(ϕ_x - ϕ_y) - 2*cos(ψ_x - ψ_y))
    return d
end

function indel_angle_changes(chainX, chainY)
    X = vcat(reshape(sequence(chainX), 1, :), angles(chainX))
    X[isnan.(X)] .= 0.0
    Y = vcat(reshape(sequence(chainY), 1, :), angles(chainY))
    Y[isnan.(Y)] .= 0.0

    println("Alignment generated by affine gap combinatorial model")
    scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1);
    aln = alignment(pairalign(GlobalAlignment(), LongAA(chainX, standardselector),
                    LongAA(chainY, standardselector), scoremodel))

    my_aln = []
    for (x, y) in aln
        if x === AA_Gap
            push!(my_aln, INSERT)
        elseif y === AA_Gap
            push!(my_aln, DELETE)
        else
            push!(my_aln, MATCH)
        end
    end

    filled = filled_alignment(my_aln, X, Y)
    angle_dist = Array{Real}(undef, length(aln))
    indel_dist = similar(angle_dist)
    matches = zeros(length(angle_dist))
    angle_dist[1] = 0
    indel_dist[1] = 100000000
    angle_dist[end] = 0
    indel_dist[end] = 100000000
    for i in axes(filled, 2)[2:(end-1)]
        #indel
        if filled[1, i] == '-' || filled[4, i] == '-'
            angle_dist[i] = 0
            indel_dist[i] = 0
        else
            if filled[1, i] == filled[4, i]
                matches[i] = 1
            end
            angle_dist[i] = ang_dist(filled[2, i], filled[3, i], filled[5, i], filled[6,i])
            indel_dist[i] = indel_dist[i-1] + 1
        end
    end

    for i in reverse(axes(filled, 2))[2:(end-1)]
        #match
        if !(filled[1, i] == '-' || filled[4, i] == '-')
            indel_dist[i] = min(indel_dist[i], indel_dist[i+1] + 1)
        end
    end
    indel_dist = indel_dist[2:(end-1)]
    angle_dist = angle_dist[2:(end-1)]
    matches = matches[2:(end-1)]

    mask1 = filter(==(1), indel_dist)
    angle1 = mean(angle_dist[mask1])
    mask2 = filter(==(2), indel_dist)
    angle2 = mean(angle_dist[mask2])
    mask3 = filter(==(3), indel_dist)
    angle3 = mean(angle_dist[mask3])
    maskn = filter(x -> x ∉ [0,1,2,3], indel_dist)
    anglen = mean(angle_dist[maskn])

    maskmatch = filter(i -> indel_dist[i] >= 1 && matches[i] == 1, eachindex(matches))
    masksub = filter(i -> indel_dist[i] >= 1 && matches[i] == 0, eachindex(matches))
    anglematch = mean(angle_dist[maskmatch])
    anglesub = mean(angle_dist[masksub])

    return [angle1, angle2, angle3, anglen, anglematch, anglesub]
end

using Combinatorics, DataFrames, PrettyTables

function experiment()
    human = retrievepdb("1A3N", dir="data/pdb")
    hA, hB = human["A"], human["B"]
    cy = retrievepdb("1UMO", dir="data/pdb")
    my = retrievepdb("1MBN", dir="data/pdb")["A"]
    cyA, cyB = cy["A"], cy["B"]
    e = retrievepdb("4ZVA", dir="data/pdb")["A"]
    pairs = [[hA, hB], [hA, cyB], [hB, cyB], [cyA, hA], [hA, my], [hB, my], [my, cyA],
             [hA, e]]
    res = vcat(map(v-> indel_angle_changes(v...)', pairs)...)
    df = DataFrame(res, ["1 res from indel", "2 res from indel",
                         "3 res from indel", "far away from indels",
                         "match", "substitution"])

    println("Angular changes during insertion/deletion events")
    pretty_table(df, nosubheader=true)

end
