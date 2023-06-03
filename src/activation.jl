export LinearActivation, SigmoidActivation, ReLUActivation, LogisticActivation

abstract type Activation end
struct LinearActivation <: Activation end
struct SigmoidActivation <: Activation end
struct ReLUActivation <: Activation end
const LogisticActivation = SigmoidActivation
