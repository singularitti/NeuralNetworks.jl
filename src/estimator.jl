using Statistics: mean

struct Estimator{F}
    network::Network
    f::F
end
function (estimator::Estimator)(data::AbstractVector{Example})
    hits =
        sum(
            argmax(estimator.network(estimator.f, example.x)) == argmax(example.y) for
            example in data
        ) / length(data)
    loss = mean(estimator.network(estimator.f, example) for example in data)
    return (hits=hits, loss=loss)
end
