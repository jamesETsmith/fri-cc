using LinearAlgebra
using Random
using SparseArrays 
using Plots
using Printf
using SharedArrays

ENV["GKSwstype"] = "100"

#  Implementing 3.2
function sample_pivotal(s::Int, p::Vector{Float64})
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

function sample_systematic(n_sample::Int, p::Vector{Float64})

    sample_r = rand()
    uk = LinRange(0, n_sample - 1, n_sample) .+ sample_r
    
    intervals = append!([0.], cumsum(p))

    last_interval = 1
    S = zeros(Int, n_sample)
    for i = 1:length(uk)
        for j = last_interval:length(intervals)-1
            if uk[i] >= intervals[j] && uk[i] < intervals[j + 1]
                S[i] = j
                last_interval = copy(j)
                break
            end
        end
    end


    return S
end

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



"""
divide_budget(g, q)

Perform step 2-9 in Alg. A.2 in latest FRI paper.
"""
function parallel_sampling(G::Int, q::Vector{Float64}, sample_method::Function=sample_pivotal)

    # Create our output set of indices to sample
    S = zeros(Int, G)


    # Threading shorthand
    n_threads = Threads.nthreads()
    step = length(q) ÷ n_threads
    rem = length(q) % n_threads
    # println("$step $rem")

    q_norm = norm(q, 1)
    t = zeros(Float64, n_threads)
    g = zeros(Int, n_threads)
    a = zeros(Float64, n_threads)
    qj_norms = zeros(Float64, n_threads)

    c = copy(G)

    for j in 1:n_threads
        ub = j * step 
        if j == n_threads
            ub = j * step + rem
        end
        # println("ub = $ub")
        qj = q[(j - 1) * step + 1:ub]
        qj_norms[j] = norm(qj, 1)

        a[j] = G * qj_norms[j] / q_norm
        t[j] = a[j] -  floor(a[j])
        g[j] = floor(a[j])
        c -= g[j]
    end

    # Make sure things ADD up
    # println("$c $(sum(g)) $(c + sum(g)) $G")
    @assert c + sum(g) == G "Per-process budget and remainder don't add to total!"

    s_prime = sample_method(c, t)
    #println(s_prime)

    for i = 1:length(s_prime)
        g[s_prime[i]] += 1
    end
    @assert sum(g) == G "Per-process budget doesn't add to total!"

    chunk_idx = cumsum(g) # Locations for where to "put" the sampled indices for each thread
    #println(chunk_idx)

    # Line 11 in A.2
    Threads.@threads for j = 1:n_threads
        # println("Hello! I'm processor $(Threads.threadid())")

        ub = j * step 
        if j == n_threads
            ub = j * step + rem
        end
        # println("ub = $ub")
        qj = copy(q[(j - 1) * step + 1:ub])
        sj = copy(qj_norms[j])
        #println("$(sum(qj)) ?= $(g[j])")

        if g[j] > a[j]
            #println("Increasing qj on $(Threads.threadid())")
            for i = 1:length(qj)
                yji = min(1., qj[i] / t[j])
                if yji == 1.
                    #println("yji == 1")
                    # qj[i] = 0
                end
                sj += yji - qj[i]
                qj[i] = copy(yji)
                if sj >= g[j]
                    qj[i] = yji + g[j] - sj
                    break
                end
            end
        elseif g[j] < a[j]
            #println("Decreasing qj on $(Threads.threadid())")

            for i = 1:length(qj)
                yji = max(0., (qj[i] - t[j]) / (1. - t[j]))
                sj += yji - qj[i]
                qj[i] = copy(yji)
                if sj <= g[j]
                    qj[i] = yji + g[j] - sj
                    break
                end
            end
        end 

        # Check sums are now g[j]
        # println("Thread = $(Threads.threadid()) g[j] = $(g[j]) sum(q[j]) = $(sum(qj))")
        @assert abs(g[j] - sum(qj)) < 1e-10 "Adjusted probabilities ($(sum(qj))) don't sum to g[j] ($(g[j])) \t $(abs(g[j] - sum(qj)))"

        lb = 1
        if j > 1
            lb = chunk_idx[j - 1] + 1
        end
        ub = chunk_idx[j]
        if j == n_threads
            @assert ub == length(S) "NOT THE RIGHT UPPER BOUND"
        end
        #println("$(Threads.threadid()) $lb  $ub")
        # NEED TO ADJUST INDICES here bc sample_method return them based on qj alone
        Sj = sample_method(g[j], qj) .+ ((j - 1) * step)
        # println("$(Threads.threadid()) $(length(lb:ub)) ?= $(length(Sj))")
        S[lb:ub] = Sj
        #println(S)

    end # End loop over threads
    return S
end # End of divide_budget_and_sample


#
# Main
#

# User parameters
vec_size = 1000
n_sample = 900

n_iter = 10000

n_points = 1000
step_size = convert(Int, n_iter / n_points)

# Choosing the vector we want to compress
x = rand(vec_size) .+ rand(vec_size) .* -1.0
# x = LinRange(1, 100,vec_size)
# x = exp10.(range(-1, stop=-6, length=vec_size))

# Error helpers
norm1 = norm(x, 1)
norm2 = norm(x, 2)
x_compare = zeros(Float64, vec_size)
instant_errors = zeros((2, n_iter))
avg_errors = zeros((2, n_iter))

D, remaining_norm = find_d_largest(n_sample, x)
D_vals = x[D]
println(length(D))
# Make the vector of probabilities
p = make_p(x, n_sample-length(D), D, remaining_norm)
println("Sum of p = $(sum(p))")

@assert abs(sum(p) + length(D) - n_sample)/n_sample < 1e-14 "$(abs(sum(p) + length(D) - n_sample))"


@time begin
    for i = 1:n_iter
    # Sample and collect average
        # S = sample_pivotal(n_sample, p)
        S = parallel_sampling(n_sample-length(D), p, sample_pivotal)
        #S = parallel_sampling(n_sample-length(D), p, sample_systematic)
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
savefig("parallel_pivotal.png")