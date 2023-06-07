module MNIST

using MLDatasets: MLDatasets
if isdefined(Base, :get_extension)
    using NeuralNetworks
    using NeuralNetworks: Example
    import NeuralNetworks: loaddata
else
    using ..NeuralNetworks
    using ..NeuralNetworks: Example
    import ..NeuralNetworks: loaddata
end

export loaddata

function loaddata(type=:train)
    @assert type ∈ (:train, :test)
    𝗫, 𝘆 = MLDatasets.MNIST(type)[:]
    𝗫 = (vec(X) for X in eachslice(𝗫; dims=3))  # Each original data is 28x28 matrices
    𝘆 = ([index == y for index in 0:9] for y in 𝘆)  # Each original data is a number
    return collect(Example(X, y) for (X, y) in zip(𝗫, 𝘆))
end

end
