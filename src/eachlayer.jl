const Maybe{T} = Union{T,Nothing}

struct EachLayer{N}
    network::N
end

eachlayer(network::Network) = EachLayer(network)

hidden(iter::EachLayer) = (iter[i] for i in (firstindex(iter) + 1):(lastindex(iter) - 1))

skipinput(iter::EachLayer) = (iter[i] for i in (firstindex(iter) + 1):lastindex(iter))

skipoutput(iter::EachLayer) = (iter[i] for i in firstindex(iter):(lastindex(iter) - 1))

# See https://github.com/JuliaLang/julia/blob/1715110/base/strings/string.jl#L207-L213
function Base.iterate(iter::EachLayer, state=firstindex(iter))
    if state == firstindex(iter)
        return (first(iter.network.layers), nothing, nothing), state + 1
    elseif state > length(iter)
        return nothing
    else
        return (
            iter.network.layers[state],
            iter.network.weights[state - 1],  # Note the index here!
            iter.network.biases[state - 1],  # Note the index here!
        ),
        state + 1
    end
end

Base.eltype(::EachLayer) = (Int64, Maybe{Matrix{Float64}}, Maybe{Vector{Float64}})

Base.length(iter::EachLayer) = length(size(iter))

Base.size(iter::EachLayer) = iter.network.layers
Base.size(iter::EachLayer, dim) = size(iter)[dim]

function Base.getindex(X::EachLayer, i)  # Only works for integers!
    if i == firstindex(X)
        return first(X.network.layers), nothing, nothing
    else
        return X.network.layers[i], X.network.weights[i - 1], X.network.biases[i - 1]
    end
end

Base.firstindex(::EachLayer) = 1

Base.lastindex(X::EachLayer) = length(X)

Base.show(io::IO, iter::EachLayer) = print(io, summary(iter))
function Base.show(io::IO, ::MIME"text/plain", network::Network)
    print(io, "Network of size ", join(network.layers, "×"))
    return nothing
end
