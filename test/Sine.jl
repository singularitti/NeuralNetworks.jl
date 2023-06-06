using Distributions: Uniform, Normal
using NeuralNetworks
using ProgressMeter: @showprogress
using Random: MersenneTwister

SEED = 42
N_SAMPLES = 200
LEARNING_RATE = 1
epochs = 1:30_000
rng = MersenneTwister(SEED)
x_samples = rand(rng, Uniform(0, 2pi), N_SAMPLES);
𝐱 = map(x -> [x], x_samples);
y_samples = sin.(x_samples) .+ rand(rng, Normal(0.0, 0.3), N_SAMPLES);
𝐲 = map(y -> [y], y_samples);
train_data = [Example(x, y) for (x, y) in zip(𝐱, 𝐲)];

network = Network(1, 10, 10, 10, 1)
init!(network, GlorotNormal())
train!(SigmoidActivation(), network, train_data, 10, LEARNING_RATE, length(epochs))
𝐲′ = [only(network(SigmoidActivation(), x)) for x in 𝐱]

default(;
    seriestype=:scatter,
    tick_direction=:out,
    legend=:none,
    grid=nothing,
    frame=:box,
    margins=(0, :mm),
)
scatter(x_samples, y_samples; color=:blue)
scatter!(x_samples, 𝐲′; color=:red)
xlims!(extrema(x_samples))
