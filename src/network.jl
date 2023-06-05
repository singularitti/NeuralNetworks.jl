using ComputedFieldTypes: @computed

export Network, feedforward

@computed struct Network{N}
    layers::NTuple{N,Int64}
    weights::NTuple{N - 1,Matrix{Float64}}
    biases::NTuple{N - 1,Vector{Float64}}
end
function Network(layers)
    weights = Tuple(
        Matrix{Float64}(undef, nⱼ, nₖ) for
        (nⱼ, nₖ) in zip(layers[(begin + 1):end], layers[begin:(end - 1)])
    )
    biases = Tuple(Vector{Float64}(undef, nⱼ) for nⱼ in layers[(begin + 1):end])
    return Network{length(layers)}(layers, weights, biases)
end
Network(layers::Integer...) = Network(layers)

(network::Network)(f, 𝘅) = feedforward(f, network.weights, network.biases, 𝘅)

function feedforward(f, weights, biases, 𝗮)
    for (w, 𝗯) in zip(weights, biases)
        𝗮 = f.(w * 𝗮 .+ 𝗯)
    end
    return 𝗮
end

Base.show(io::IO, network::Network) = print(io, join(network.layers, "×"), " network")
