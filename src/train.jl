using ProgressMeter: @showprogress
using Random: shuffle

export train!

function train!(
    network::MultilayerPerceptron,
    data::AbstractVector{<:Example},
    batchsize::Integer,
    η,
    nepochs=1,
)
    @showprogress for _ in 1:nepochs
        data = shuffle(data)
        batches = Iterators.partition(data, batchsize)
        for batch in batches
            train!(network, batch, η)
        end
    end
    return network
end
function train!(network::MultilayerPerceptron, batch::AbstractVector{<:Example}, η)
    new_networks = collect(train(network, example, η / length(batch)) for example in batch)
    new_weights = (
        mean(new_network.weights[i] for new_network in new_networks) for
        i in 1:length(network.weights)
    )
    new_biases = (
        mean(new_network.biases[i] for new_network in new_networks) for
        i in 1:length(network.biases)
    )
    for (weight, bias, new_weight, new_bias) in
        zip(network.weights, network.biases, new_weights, new_biases)
        weight[:] = new_weight
        bias[:] = new_bias
    end
    return network
end
function train!(network::MultilayerPerceptron, example::Example, η)
    𝝯w, 𝝯𝗯 = backpropagate(network, example)
    for (w, 𝗯, ∇w, ∇𝗯) in zip(network.weights, network.biases, 𝝯w, 𝝯𝗯)
        w[:, :] .-= η * ∇w
        𝗯[:] .-= η * ∇𝗯
    end
    return network
end
function train(network::MultilayerPerceptron, example::Example, η)
    𝝯w, 𝝯𝗯 = backpropagate(network, example)
    new_network = deepcopy(network)
    for (w, 𝗯, ∇w, ∇𝗯) in zip(new_network.weights, new_network.biases, 𝝯w, 𝝯𝗯)
        w[:, :] .-= η * ∇w
        𝗯[:] .-= η * ∇𝗯
    end
    return new_network
end
