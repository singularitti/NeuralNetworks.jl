using ComputedFieldTypes: @computed

export Network, feedforward

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

Base.show(io::IO, network::Network) = print(io, join(network.layers, "×"), " network")
