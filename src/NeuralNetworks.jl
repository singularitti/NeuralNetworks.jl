module NeuralNetworks

include("example.jl")
include("activation.jl")
include("network.jl")
include("eachlayer.jl")
include("initialization.jl")
include("estimate.jl")
include("backpropagate.jl")
include("train.jl")

function loaddata end

if !isdefined(Base, :get_extension)
    include("../ext/MNIST.jl")
end

end
