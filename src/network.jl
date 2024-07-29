using ComputedFieldTypes: @computed

export MultilayerPerceptron, feedforward
@computed struct MultilayerPerceptron{T,N}
    layers::NTuple{N,Int64}
    weights::NTuple{N - 1,Matrix{Float64}}
    biases::NTuple{N - 1,Vector{Float64}}
    activations::NTuple{N - 1,T}
end
function MultilayerPerceptron(layers, activations)
    weights = Tuple(
        zeros(nⱼ, nₖ) for (nⱼ, nₖ) in zip(layers[(begin + 1):end], layers[begin:(end - 1)])
    )  # Do not use `undef` here, as it will be added and subtracted from!
    biases = Tuple(zeros(nⱼ) for nⱼ in layers[(begin + 1):end])  # Do not use `undef` here!
    return MultilayerPerceptron{Base.promote_typeof(activations...),length(layers)}(
        Tuple(layers), weights, biases, Tuple(activations)
    )
end

(network::MultilayerPerceptron)(𝘅) =
    feedforward(network.weights, network.biases, network.activations, 𝘅)

function feedforward(weights, biases, activations, 𝗮)
    for (w, 𝗯, activation) in zip(weights, biases, activations)
        𝗮 = activation.(w * 𝗮 .+ 𝗯)
    end
    return 𝗮
end

Base.show(io::IO, network::MultilayerPerceptron) =
    print(io, join(network.layers, "×"), " network")
