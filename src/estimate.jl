using Statistics: mean

export estimate

function estimate(network::Network, activation::Activation, data::AbstractVector{Example})
    f = functionof(activation)
    hits =
        sum(argmax(network(f, example.x)) == argmax(example.y) for example in data) /
        length(data)
    loss = mean(network(f, example) for example in data)
    return (hits=hits, loss=loss)
end
