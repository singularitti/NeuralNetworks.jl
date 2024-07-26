network = MultilayerPerceptron(784, 30, 10)
# init!(network, GlorotNormal())
init!(network, GlorotUniform())
train_data = loaddata(:train);
test_data = loaddata(:test);
epochs = 1:30
estimations = @showprogress map(epochs) do _
    train!(SigmoidActivation(), network, train_data, 10, 3, 1)
    estimate(SigmoidActivation(), network, test_data, MeanSquaredError())
end
hits = [estimation.hits for estimation in estimations]
loss = [estimation.loss for estimation in estimations]

default(;
    seriestype=:scatter,
    xlims=(1, 30),
    tick_direction=:out,
    grid=nothing,
    frame=:box,
    margins=(0, :mm),
)
p = plot(
    epochs,
    hits;
    color=:red,
    label="hits",
    xguide="epoch",
    yguide="hits ratio",
    legend=:topleft,
)
p = twinx(p)
plot!(
    p,
    epochs,
    loss;
    color=:blue,
    label="loss",
    legend=:right,
    xguide="",
    yguide="loss ratio",
    frame=:box,
)
