export LinearActivation, SigmoidActivation, ReLUActivation, LogisticActivation, instantiate

abstract type Activation end
struct LinearActivation <: Activation end
struct SigmoidActivation <: Activation end
struct ReLUActivation <: Activation end
const LogisticActivation = SigmoidActivation

function instantiate(::SigmoidActivation)
    sigmoid(z) = 1 / (1 + exp(-z))
    sigmoid′(z) = sigmoid(z) * (1 - sigmoid(z))
    return (sigmoid, sigmoid′)
end
