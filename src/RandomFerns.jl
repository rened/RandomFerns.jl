VERSION >= v"0.4.0-dev+6521" && __precompile__()

module RandomFerns

using FunctionalData

export RandFern, getcell
export RandFerns, predict, predict!, votes!

type RandFern
    base::Array{Float32,2}
    splits::Array{Float32,2}
    cells::Array{Array{Float32,2},1}
    cellinds::Dict{Int,Int}
    C::Int
    proj::Array{Float32,2}
    zeros::Array{Float32,2}
end

function RandFern(data, targets; regression = false, nsplits = 10)
    assert(size(data,2)==size(targets,2))
    base = randn(nsplits,size(data,1))
    proj = base*data
    mi = minimum(proj,2)
    ma = maximum(proj,2)
    splits = col(mi + (ma - mi).*rand(nsplits,1))
    binary = proj .< vec(splits)
    cells = Any[]
    cellinds = Dict{Int,Int}()
    counts = Dict{Int,Int}()
    if regression
        C = size(targets,1)
    else
        C = round(Int,maximum(targets))
        labelcounts = [sum(targets.==i) for i in 1:C]
    end
    for i = 1:size(binary,2)
        ind = bin2ind(binary[:,i])
        if !haskey(cellinds,ind)
            push!(cells, zeros(Float32,C,1))
            cellinds[ind] = length(cells)
        end
        if regression
            cells[cellinds[ind]] += targets[:,i]
            counts[ind] = get(counts, ind, 0) + 1
        else
            c = round(Int,targets[i])
            cells[cellinds[ind]][c] += 1/labelcounts[c]
        end
    end
    if regression
        for i = keys(cellinds)
            cells[cellinds[i]] /= counts[i]
        end
    end
    RandFern(base, convert(Array{Float32,2},splits), cells, cellinds, C, zeros(Float32,nsplits,1), zeros(Float32,C,1))
end

function getcell(a::RandFern, data::Array{Float32,2})
    A_mul_B!(a.proj, a.base, data)
    ind = 1
    n = length(a.splits)
    for i = 1:n
        if a.proj[i] < a.splits[i]
            ind += 2^(n-i)
        end
    end
    haskey(a.cellinds,ind) ? (a.cells[a.cellinds[ind]]::Array{Float32,2},true) : (a.zeros, false)
end

bin2ind(a) = (local r = 1; for i = 1:length(a) r += 2^(length(a)-i)*a[i] end; r)

type RandFerns
    ferns::Array{RandFern}
    regression::Bool
    buf::Array{Float32,2}
end

function RandFerns(data, targets; nferns = 32, nsplits = 10, regression = false)
    ferns = [RandFern(data, targets; nsplits = nsplits, regression = regression) for i in 1:nferns]
    RandFerns(ferns, regression, zeros(Float32,ferns[1].C,1))
end

predict{T<:Real}(a::RandFerns, data::Array{T,2}) = @p map convert(Array{Float32,2},data) x->predict!(a,x)

function predict!(a::RandFerns, data::Array{Float32,2})
    assert(size(data,2) == 1)
    a.regression ? votes!(a,data) : indmax(votes!(a, data))
end

function votes!(a::RandFerns, data)
    nfound = 0
    a.buf[:] = 0
    for i = 1:length(a.ferns)
        temp, found = getcell(a.ferns[i], data)
        nfound += found
        for j = 1:length(a.buf)
            a.buf[j] += temp[j]
        end
    end
    if a.regression
        for i = 1:length(a.buf)
            a.buf[i] /= nfound
        end
    end
    a.buf
end
    
end

