println("\n\n\nRunning tests ...")

using RandomFerns
using Base.Test

using FunctionalData

D = 2
N = 100
C = 10
S = 10
data = @p map (1:C) (x->randn(D,N).+5x.+randn(D,1)) | unstack | flatten
labels = @p map (1:C) (x->x*ones(1,N)) | unstack | flatten

a = RandFerns(data,labels);
# testdata = @p meshgrid linspace(-15,15,200) linspace(-15,15,200)
r = predict(a, data)
@test sum(vec(r).!=vec(labels)) < 100

a = RandFerns(data,labels; regression = true);
r = predict(a, data)
@test mean(abs(vec(r)-vec(labels))) < 0.5

println("   done running tests!")
