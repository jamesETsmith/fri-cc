using LinearAlgebra, Random, SparseArrays, Plots, Printf

ENV["GKSwstype"] = "100"

#  Implementing the old (aka "alt") 3.2
function sample_pivotal_alt(s, p)
        # println("$p $(sum(p))")
    S = []
    i = 1
    a = copy(p[1])
    
    for j = 2:length(p)
        a += p[j]
        rn = rand()
        if a < 1.
            if rn < p[j] / a
                i = copy(j)
            end
        else
            a -= 1.
            if rn < (1. - p[j]) / (1. - a)
                append!(S, [i])
                i = copy(j)
            else
                append!(S, [j])
            end
        end
        if length(S) == s
            break
        end
        if j == length(p)
            append!(S, [j])
        end
    end
    @assert s == length(S) "Wrong number of elements sampled $s != $(length(S))" 
    return S
end

"""
sample_pivotal(g, p)

Implementation of the latest SERIAL pivotal sampling algorithm (3.2) from the latest FRI paper.
"""
function sample_pivotal(g, p)

    # Line 2
    S = []
    l = 1
    f = 2
    b = copy(p[1])
    max_k = size(p)[1]
    p_norm = norm(p, 1)
    
    

    # 
    for j in 1:g
        prob = copy(b)
        s = copy(l)
        println("\nMacro iter $j: size of index list: $(size(S)[1])")
  
        # Get max s (Line 4)
        for i in l:max_k
            println("Micro iter $i: prob = $prob")
            if prob + p[i] > 1. || i == max_k
                println("Max s = $s \t prob + p[s+1] = $(prob + p[i])")
                break
            else
                s += 1
                prob += p[i]
            end
        end

        println("l = $l \t f = $f \t s = $s")

        # (Line 5)
        rn = rand()
        idx = append!([l], collect(f:s))
        c_sum = cumsum(append!([b], p[f:s]))
        println(idx)
        println(c_sum)
        @assert size(c_sum)[1] == size(idx)[1] "THINGS AREN'T THE SAME SIZE"
        # exit(0)
        
        h = 0
        for i in 1:size(c_sum)[1]
            if rn < c_sum[i] || i == size(c_sum)[1]
                println("Picking h $(idx[i])") 
                h = copy(idx[i])
                break
            end
        end

        # (Line 6 - 7)
        a = 1. - prob
        b = p[s + 1] - a
        println("a = $a \t b = $b")
        @assert b >= 0 && b <= 1 "b is out of the correct range"

        # Line 8
        cume_prob = 1 -  a / (1 - b)
        println("Cume Sum Prob $(cume_prob)")
        rn = rand()
        @assert (cume_prob) > 0 && cume_prob < 1 "BAD PROBABILITY $(cume_prob)"
        if rn < (cume_prob)
            push!(S, copy(h))
            l = s + 1
                  # Line 9
            f = s + 2
        else
            push!(S, s + 1)
            l = copy(h)
            f = l + 1
        end
    
  
    end # loop over j


    return S
    @assert size(S)[1] == g "Didn't sample the write number of indices!!" 
end # sample_pivotal

function find_d_largest(n_sample::Int, x::Vector{Float64})
    remaining_norm = norm(x,1)
    D = []
    sort_idx = sortperm(x, by=abs, rev=true)

    for i=1:n_sample
        idx = sort_idx[i]
        d = length(D)
        xi = abs(x[idx])
        if (n_sample-d)*xi >= remaining_norm - xi 
            append!(D,[idx])
            remaining_norm -= xi
            d -= 1
        else
            break
        end
    end
    return D, remaining_norm
end

function make_p(x, g, D, rn)
    p = zeros(Float64, length(x))
    indices = [x for x ∈ 1:length(x) if x ∉ D]
    Threads.@threads for i=1:length(indices)
        idx = indices[i]
        p[idx] = abs(x[idx]) * g/rn
    end
    return p
end


#
# Main
#

# User parameters
vec_size = 20
n_sample = 4

n_iter = 100000
n_points = 1000
step_size = convert(Int, n_iter / n_points)

# Choosing the vector we want to compress
# x = rand(Float64, vec_size)
# x = LinRange(1, 100,vec_size)
# x = exp10.(range(-1, stop=-6, length=vec_size))
x = rand(vec_size) .+ rand(vec_size) .* -1.0
# x = x / norm(x, 1)

# Error helpers
norm1 = norm(x, 1)
norm2 = norm(x, 2)
x_compare = zeros(Float64, vec_size)
instant_errors = zeros((2, n_iter))
avg_errors = zeros((2, n_iter))


# Make the vector of probabilities
D, remaining_norm = find_d_largest(n_sample, x)
D_vals = x[D]
println(length(D))
# Make the vector of probabilities
p = make_p(x, n_sample-length(D), D, remaining_norm)
println("Sum of p = $(sum(p))")
println(p)

@assert abs(sum(p) + length(D) - n_sample)/n_sample < 1e-14 "$(abs(sum(p) + length(D) - n_sample))"

# 
@time begin
    for i = 1:n_iter
    # Sample and collect average
    S = sample_pivotal(n_sample-length(D), p)
        exit(0)
        indices = append!(copy(D), S)
        vals = append!(copy(D_vals), x[S] ./ p[S] )
        sp_v = sparsevec(indices, vals, vec_size)
        global x_compare += sp_v

    # Keep track of errors
        instant_errors[1,i] = norm(x - sp_v, 1) / norm1
        instant_errors[2,i] = norm(x - sp_v, 2) / norm2
        avg_errors[1,i] = norm(x .- x_compare ./ i, 1) / norm1
        avg_errors[2,i] = norm(x .- x_compare ./ i, 2) / norm2

    # Log
        if i % step_size == 0
            Printf.@printf("Iter. %d:\tInstant L1: %.1e\tAvg L1:%.1e\tInstant L2: %.1e\tAvg L2: %.1e\n", 
        i, instant_errors[1,i], avg_errors[1,i], instant_errors[2,i], avg_errors[2,i])
        end
    end
end

#
# Plotting
#
iters = 1:step_size:n_iter
plot(iters, avg_errors[1,1:step_size:end], xscale=:log10, yscale=:log10, label="L1 Error")
plot!(iters, avg_errors[2,1:step_size:end], xscale=:log10, yscale=:log10, label="L2 Error")
plot!(iters, iters.^(-0.5), label="x^(-1/2)", xlabel="Iterations", ylabel="Relative Error")
savefig("pivotal.png")