export Example, unwrap

struct Example{X,Y}
    x::Vector{X}
    y::Vector{Y}
end

unwrap(example::Example) = example.x, example.y
