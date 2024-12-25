export backpropagate

function backpropagate(network::MultilayerPerceptron, example::Example)
    𝘅, 𝘆 = unwrap(example)
    zs, activated = Vector{Float64}[], Vector{Float64}[𝘅]
    # Feed forward
    𝗮 = 𝘅
    for (_, wˡ, 𝗯ˡ, σ) in skipinput(eachlayer(network))
        𝘇ˡ = wˡ * 𝗮 .+ 𝗯ˡ
        push!(zs, 𝘇ˡ)
        𝗮 = σ.(𝘇ˡ)
        push!(activated, 𝗮)
    end
    𝘇ᴸ, 𝗮ᴸ = zs[end], activated[end]
    # Backward pass
    _, _, _, σᴸ = last(eachlayer(network))
    σ′ᴸ = derivativeof(σᴸ)
    𝝳 = (𝗮ᴸ .- 𝘆) .* σ′ᴸ.(𝘇ᴸ)  # 𝝳ᴸ
    𝝯w, 𝝯𝗯 = [kron(𝝳, activated[end - 1]')], [𝝳]  # 𝝯wᴸ, 𝝯𝗯ᴸ
    # Select `network` from layer L to 3, `zs` from layer L-1 to 2, `activations` from layer L-2 to 1
    for ((_, wˡ⁺¹, _, σ), 𝘇ˡ, 𝗮ˡ⁻¹) in zip(
        Iterators.reverse(skipinput(eachlayer(network))),
        zs[(end - 1):-1:begin],
        activated[(end - 2):-1:begin],
    )
        σ′ = derivativeof(σ)
        𝝳 = transpose(wˡ⁺¹) * 𝝳 .* σ′.(𝘇ˡ)
        push!(𝝯w, kron(𝝳, 𝗮ˡ⁻¹'))
        push!(𝝯𝗯, 𝝳)
    end
    return reverse(𝝯w), reverse(𝝯𝗯)
end
