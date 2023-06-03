export LinearActivation,
    SigmoidActivation,
    ReLUActivation,
    HyperbolicTangent,
    LogisticActivation,
    instantiate,
    functionof,
    derivativeof

abstract type Activation end
struct LinearActivation <: Activation end
struct SigmoidActivation <: Activation end
struct ReLUActivation <: Activation end
struct HyperbolicTangent <: Activation end
const TanhActivation = HyperbolicTangent
const LogisticActivation = SigmoidActivation

instantiate(activation::Activation) = (functionof(activation), derivativeof(activation))

functionof(::SigmoidActivation) = z -> 1 / (1 + exp(-z))
functionof(::HyperbolicTangent) = tanh
functionof(::ReLUActivation) = z -> z >= 0 ? z : 0

function derivativeof(::SigmoidActivation)
    sigmoid = functionof(SigmoidActivation())
    return z -> sigmoid(z) * (1 - sigmoid(z))
end
derivativeof(::HyperbolicTangent) = z -> 1 - tanh(z)^2
derivativeof(::ReLUActivation) = z -> z >= 0 ? 1 : 0
