export LinearActivation,
    SigmoidActivation, ReLUActivation, HyperbolicTangent, LogisticActivation, derivativeof

abstract type Activation end
struct LinearActivation <: Activation end
struct SigmoidActivation <: Activation end
struct ReLUActivation <: Activation end
struct HyperbolicTangent <: Activation end
const TanhActivation = HyperbolicTangent
const LogisticActivation = SigmoidActivation

(::SigmoidActivation)(z) = 1 / (1 + exp(-z))
(::HyperbolicTangent)(z) = tanh(z)
(::ReLUActivation)(z) = z >= 0 ? z : 0

derivativeof(sigmoid::SigmoidActivation) = z -> sigmoid(z) * (1 - sigmoid(z))
derivativeof(::HyperbolicTangent) = z -> 1 - tanh(z)^2
derivativeof(::ReLUActivation) = z -> z >= 0 ? 1 : 0
