using ComputedFieldTypes: @computed

export MultilayerPerceptron, feedforward

@computed struct MultilayerPerceptron{N}
    layers::NTuple{N,Int64}
    weights::NTuple{N - 1,Matrix{Float64}}
    biases::NTuple{N - 1,Vector{Float64}}
end
function MultilayerPerceptron(layers)
    weights = Tuple(
        zeros(nⱼ, nₖ) for (nⱼ, nₖ) in zip(layers[(begin + 1):end], layers[begin:(end - 1)])
    )  # Do not use `undef` here, as it will be added and subtracted from!
    biases = Tuple(zeros(nⱼ) for nⱼ in layers[(begin + 1):end])  # Do not use `undef` here!
    return MultilayerPerceptron{length(layers)}(layers, weights, biases)
end
MultilayerPerceptron(layers::Integer...) = MultilayerPerceptron(layers)

(network::MultilayerPerceptron)(f, 𝘅) = feedforward(f, network.weights, network.biases, 𝘅)

function feedforward(f, weights, biases, 𝗮)
    for (w, 𝗯) in zip(weights, biases)
        𝗮 = f.(w * 𝗮 .+ 𝗯)
    end
    return 𝗮
end

Base.show(io::IO, network::MultilayerPerceptron) =
    print(io, join(network.layers, "×"), " network")
