export GlorotNormal, GlorotUniform, XavierNormal, XavierUniform, HeNormal, HeUniform, init!

abstract type WeightInitialization end
abstract type UniformDistributionInitialization <: WeightInitialization end
abstract type NormalDistributionInitialization <: WeightInitialization end
struct GlorotNormal <: NormalDistributionInitialization end
struct GlorotUniform <: UniformDistributionInitialization end
struct HeNormal <: NormalDistributionInitialization end
struct HeUniform <: UniformDistributionInitialization end
const XavierNormal = GlorotNormal
const XavierUniform = GlorotUniform

function randweight(nₒᵤₜ, nᵢₙ, ::GlorotNormal)
    σ = sqrt(2 / (nₒᵤₜ + nᵢₙ))
    return randn(nₒᵤₜ, nᵢₙ) * σ
end
function randweight(nₒᵤₜ, nᵢₙ, ::GlorotUniform)
    r = sqrt(6 / (nₒᵤₜ + nᵢₙ))
    return rand(nₒᵤₜ, nᵢₙ) * 2r .- r
end

function init!(network::Network, scheme::WeightInitialization)
    layers = eachlayer(network)
    for ((nₒᵤₜ, weight, _), (nᵢₙ, _, _)) in zip(excludeinput(layers), excludeoutput(layers))
        weight[:] = randweight(nₒᵤₜ, nᵢₙ, scheme)
    end
    return network
end
