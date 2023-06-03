export LinearActivation,
    SigmoidActivation,
    ReLUActivation,
    LogisticActivation,
    instantiate,
    functionof,
    derivativeof

abstract type Activation end
struct LinearActivation <: Activation end
struct SigmoidActivation <: Activation end
struct ReLUActivation <: Activation end
const LogisticActivation = SigmoidActivation

instantiate(activation::Activation) = (functionof(activation), derivativeof(activation))

function functionof(::SigmoidActivation)
    sigmoid(z) = 1 / (1 + exp(-z))
    return sigmoid
end

function derivativeof(::SigmoidActivation)
    sigmoid = functionof(SigmoidActivation())
    sigmoid′(z) = sigmoid(z) * (1 - sigmoid(z))
    return sigmoid′
end
