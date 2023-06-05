export backpropagate

function backpropagate(network::Network, activation::Activation, example::Example)
    𝘅, 𝘆 = unwrap(example)
    f, f′ = instantiate(activation)
    # Feed forward
    zs, activations = Vector{Float64}[], Vector{Float64}[𝘅]
    𝗮 = 𝘅
    for (_, wˡ, 𝗯ˡ) in excludeinput(eachlayer(network))
        𝘇ˡ = wˡ * 𝗮 .+ 𝗯ˡ
        push!(zs, 𝘇ˡ)
        𝗮 = f.(𝘇ˡ)
        push!(activations, 𝗮)
    end
    𝘇ᴸ, 𝗮ᴸ = zs[end], activations[end]
    # Backward pass
    𝝳 = (𝗮ᴸ .- 𝘆) .* f′.(𝘇ᴸ)  # 𝝳ᴸ
    𝝯w, 𝝯𝗯 = [kron(𝝳, activations[end - 1]')], [𝝳]  # 𝝯wᴸ, 𝝯𝗯ᴸ
    # Select `network` from layer L to 3, `zs` from layer L-1 to 2, `activations` from layer L-2 to 1
    for ((_, wˡ⁺¹, _), 𝘇ˡ, 𝗮ˡ⁻¹) in zip(
        Iterators.reverse(excludeinput(eachlayer(network))),
        zs[(end - 1):-1:begin],
        activations[(end - 2):-1:begin],
    )
        𝝳 = transpose(wˡ⁺¹) * 𝝳 .* f′.(𝘇ˡ)
        push!(𝝯w, kron(𝝳, 𝗮ˡ⁻¹'))
        push!(𝝯𝗯, 𝝳)
    end
    return reverse(𝝯w), reverse(𝝯𝗯)
end
