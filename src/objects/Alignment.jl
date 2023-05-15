using Base
import Base: show, size, IndexStyle, getindex, setindex!

const Domino = AbstractVector{<:Integer}

struct Alignment <: AbstractVector{Domino}
    ids::AbstractVector{Integer}
    row_indices::Dict{Integer, Integer}
    data::AbstractMatrix{Integer}
end

function Alignment(ids::Vector{<:Integer}, data::AbstractMatrix{<:Integer})
    row_indices = Dict([(ids[i],i) for i ∈ eachindex(ids)])
    # remove null columns
    nonzero_cols = (!).(iszero.(eachcol(data)))
    return Alignment(ids, row_indices, data[:, nonzero_cols])
end

function Alignment(data::AbstractMatrix{<:Integer})
    ids = collect(1:size(data, 1))
    return Alignment(ids, data)
end

ids(a::Alignment) = a.ids
data(a::Alignment) = a.data
# number of sequences is given by the number of rows
num_sequences(a::Alignment) = size(a.data, 1)

# length of alignment is given by the number of columns
Base.size(a::Alignment) = (size(a.data, 2),)

Base.IndexStyle(::Type{<:Alignment}) = IndexLinear()
Base.getindex(a::Alignment, i::Int) = a.data[:, i]
@views Base.setindex!(a::Alignment, v::Domino, i::Int) = a.data[:, i] .= v

row_index(a::Alignment, id::Int) = a.row_indices[id]

function mask(a::Alignment, allowed::AbstractArray{<:Domino})
    return BitVector([all(v .∈ allowed) for v ∈ a])
end

function slice(a::Alignment, mask::BitVector)
    Alignment(hcat(a[mask]...))
end

sequence_lengths(a::Alignment) = [count(x -> x==1, a.data[i, :]) for i ∈ 1:num_sequences(a)]

function subalignment(a::Alignment, ids::AbstractVector{<:Integer}) :: Alignment
    new_indices = getindex.(Ref(a.row_indices), ids)
    return Alignment(ids, a.data[new_indices, :])
end

Base.show(io::IO, a::Alignment) = print(io, "Alignment(" * string(a.data) * '\n')

# Combine two alignments using their common "parent" sequence to establish consensus
function combine(parent_id::Int, a1::Alignment, a2::Alignment) :: Alignment
    @assert intersect(a1.ids, a2.ids) == [parent_id] "The alignments to be combined should only have the parent sequence in common"

    # Which columns in each alignment contain the parent residue
    contains1 = data(a1)[row_index(a1, parent_id), :] .== 1
    contains2 = data(a2)[row_index(a2, parent_id), :] .== 1
    @assert count(contains1) == count(contains2) "Length of parent in alignments to be combined is not consistent"
    # Number of residues of parent sequence
    parent_length = count(contains1)

    # The alignment matrices excluding the parent
    data1 = data(a1)[1:end .!= row_index(a1, parent_id), :]
    data2 = data(a2)[1:end .!= row_index(a2, parent_id), :]

    empty1 = zeros(length(a1))
    empty2 = zeros(length(a2))

    n = length(a1)
    m = length(a2)
    i = 1; j = 1
    columns = []

    # Match columns which have a parent residue in common
    for _ in 1:parent_length
        while !contains1[i]
            push!(columns, [0; data1[:, i]; empty2])
            i += 1
        end

        while !contains2[j]
            push!(columns, [0; empty1; data2[:, j]])
            j += 1
        end

        push![1; data1[:, i]; data2[:, j]]
    end

    # Add remaining columns from each alignment
    for _ in i:n
        push!(columns, [0; data1[:, i]; empty2])
    end
    for _ in j:m
        push!(columns, [0; empty1; data2[:, j]])
    end

    new_ids = [parent_id; filter(!=(parent_id), a1.ids); filter(!=(parent_id), a2.ids)]

    return Alignment(new_ids, hcat(columns...))

end
