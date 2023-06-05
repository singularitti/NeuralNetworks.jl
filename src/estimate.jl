using Statistics: mean

export estimate

abstract type LossFunction end
struct MeanSquaredError <: LossFunction end

function computeloss(f::Activation, network::Network, example::Example, ::MeanSquaredError)
    𝘅, 𝘆 = unwrap(example)
    𝘆̂ = network(f, 𝘅)
    return sum(abs2, 𝘆 .- 𝘆̂)
end

function estimate(
    f::Activation, network::Network, data::AbstractVector{Example}, l::LossFunction
)
    hits =
        sum(argmax(network(f, example.x)) == argmax(example.y) for example in data) /
        length(data)
    loss = mean(computeloss(f, network, example, l) for example in data)
    return (hits=hits, loss=loss)
end
