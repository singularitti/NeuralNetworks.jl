using ComputedFieldTypes: @computed
using Statistics: mean

export Network, Estimator, feedforward, eachlayer, hidden, excludeinput

const Maybe{T} = Union{T,Nothing}

struct Example
    x::Vector{Float64}
    y::Vector{Float64}
end

@computed struct Network{N}
    layers::NTuple{N,Int64}
    weights::NTuple{N - 1,Matrix{Float64}}
    biases::NTuple{N - 1,Vector{Float64}}
end
function Network(layers)
    weights = Tuple(
        randn(nj, nk) for (nj, nk) in zip(layers[(begin + 1):end], layers[begin:(end - 1)])
    )  # Cannot use `undef` here!
    biases = Tuple(randn(nj) for nj in layers[(begin + 1):end])  # Cannot use `undef` here!
    return Network{length(layers)}(layers, weights, biases)
end
Network(layers::Integer...) = Network(layers)

(network::Network)(f, data::Example) = network(f, data.x, data.y)
(network::Network)(f, 𝘅) = feedforward(f, network.weights, network.biases, 𝘅)
function (network::Network)(f, 𝘅, 𝘆)
    𝘆̂ = network(f, 𝘅)
    return sum(abs2, 𝘆 .- 𝘆̂)
end

function feedforward(f, weights, biases, 𝗮)
    for (w, 𝗯) in zip(weights, biases)
        𝗮 = f.(w * 𝗮 .+ 𝗯)
    end
    return 𝗮
end

struct Estimator{F}
    network::Network
    f::F
end
function (estimator::Estimator)(data::AbstractVector{Example})
    hits =
        sum(
            argmax(estimator.network(estimator.f, example.x)) == argmax(example.y) for
            example in data
        ) / length(data)
    loss = mean(estimator.network(estimator.f, example) for example in data)
    return (hits=hits, loss=loss)
end

struct EachLayer{N}
    network::N
end

eachlayer(network::Network) = EachLayer(network)

hidden(iter::EachLayer) = (iter[i] for i in (firstindex(iter) + 1):(lastindex(iter) - 1))

excludeinput(iter::EachLayer) = (iter[i] for i in (firstindex(iter) + 1):lastindex(iter))

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

Base.show(io::IO, network::Network) = print(io, join(network.layers, "×"), " network")
Base.show(io::IO, iter::EachLayer) = print(io, summary(iter))
function Base.show(io::IO, ::MIME"text/plain", network::Network)
    print(io, "Network of size ", join(network.layers, "×"))
    return nothing
end
