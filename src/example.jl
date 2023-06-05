export Example, unwrap

struct Example{X,Y}
    x::Vector{X}
    y::Vector{Y}
end
Example(x::AbstractVector{X}, y::AbstractVector{Y}) where {X,Y} = Example{X,Y}(x, y)

unwrap(example::Example) = example.x, example.y
