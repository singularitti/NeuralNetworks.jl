using Statistics: mean

export MeanSquaredError, estimate

abstract type Loss end
struct MeanSquaredError <: Loss end

struct Estimation
    hits::Float64
    loss::Float64
end

function computeloss(f::Activation, network::Network, example::Example, ::MeanSquaredError)
    𝘅, 𝘆 = unwrap(example)
    𝘆̂ = network(f, 𝘅)
    return mean(abs2, 𝘆 .- 𝘆̂) / 2
end

function estimate(f::Activation, network::Network, data::AbstractVector{<:Example}, l::Loss)
    hits = mean(argmax(network(f, example.x)) == argmax(example.y) for example in data)
    loss = mean(computeloss(f, network, example, l) for example in data)
    return Estimation(hits, loss)
end
