using Impute
using Combinatorics
using DataFrames
using Distances
using LinearAlgebra
using RDatasets
using Statistics
using Test

function add_missings(X, ratio=0.1)
    result = Matrix{Union{Float64, Missing}}(X)

    for i in 1:floor(Int, length(X) * ratio)
        result[rand(1:length(X))] = missing
    end

    return result
end

@testset "Impute" begin
    a = Vector{Union{Float64, Missing}}(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)

    @testset "Drop" begin
        result = impute(a, :drop; limit=0.2)
        expected = copy(a)
        deleteat!(expected, [2, 3, 7])

        @test result == expected
    end

    @testset "Interpolate" begin
        result = impute(a, :interp; limit=0.2)
        @test result == collect(1.0:1.0:20)
        @test result == interp(a)
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = impute(a, :fill, fill_val; limit=0.2)
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
        end

        @testset "Mean" begin
            result = impute(a, :fill; limit=0.2)
            expected = copy(a)
            expected[[2, 3, 7]] .= mean(a[mask])

            @test result == expected
        end
    end

    @testset "LOCF" begin
        result = impute(a, :locf; limit=0.2)
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
    end

    @testset "NOCB" begin
        result = impute(a, :nocb; limit=0.2)
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
    end

    @testset "DataFrame" begin
        data = dataset("boot", "neuro")
        df = impute(data, :interp; limit=1.0)
    end

    @testset "Matrix" begin
        data = Matrix(dataset("boot", "neuro"))

        @testset "Drop" begin
            result = Iterators.drop(data)
            @test size(result, 1) == 4
        end

        @testset "Fill" begin
            result = impute(data, :fill, 0.0; limit=1.0)
            @test size(result) == size(data)
        end
    end

    @testset "Not enough data" begin
        @test_throws ImputeError impute(a, :drop)
    end

    @testset "Chain" begin
        data = Matrix(dataset("boot", "neuro"))
        result = chain(
            data,
            Impute.Interpolate(),
            Impute.LOCF(),
            Impute.NOCB();
            limit=1.0
        )

        @test size(result) == size(data)
        # Confirm that we don't have any more missing values
        @test !any(ismissing, result)
    end

    @testset "Alternate missing functions" begin
        data1 = dataset("boot", "neuro")                # Missing values with `missing`
        data2 = impute(data1, :fill, NaN; limit=1.0)     # Missing values with `NaN`

        @test impute(data1, :drop; limit=1.0) == dropmissing(data1)

        result1 = chain(data1, Impute.Interpolate(), Impute.Drop(); limit=1.0)
        result2 = chain(data2, isnan, Impute.Interpolate(), Impute.Drop(); limit=1.0)
        @test result1 == result2
    end

    @testset "SVD" begin
        # Test a case where we expect SVD to perform well (e.g., many variables, )
        @testset "Data match" begin
            data = mapreduce(hcat, 1:1000) do i
                seeds = [sin(i), cos(i), tan(i), atan(i)]
                mapreduce(vcat, combinations(seeds)) do args
                    [
                        +(args...),
                        *(args...),
                        +(args...) * 100,
                        +(abs.(args)...),
                        (+(args...) * 10) ^ 2,
                        (+(abs.(args)...) * 10) ^ 2,
                        log(+(abs.(args)...) * 100),
                        +(args...) * 100 + rand(-10:0.1:10),
                    ]
                end
            end

            println(svd(data').S)
            X = add_missings(data')

            svd_imputed = Impute.svd(X)
            mean_imputed = impute(copy(X), :fill; limit=1.0)

            # With sufficient correlation between the variables and enough observation we
            # expect the svd imputation to perform severl times better than mean imputation.
            @test nrmsd(svd_imputed, data') < nrmsd(mean_imputed, data') * 0.25
        end

        # Test a case where we know SVD imputation won't perform well
        # (e.g., only a few variables, only )
        @testset "Data mismatch - too few variables" begin
            data = Matrix(dataset("Ecdat", "Electricity"))
            X = add_missings(data)

            svd_imputed = Impute.svd(X)
            mean_imputed = impute(copy(X), :fill; limit=1.0)

            # If we don't have enough variables then SVD imputation will probably perform
            # about as well as mean imputation.
            @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
        end

        @testset "Data mismatch - poor low rank approximations" begin
            M = rand(100, 200)
            data = M * M'
            X = add_missings(data)

            svd_imputed = Impute.svd(X)
            mean_imputed = impute(copy(X), :fill; limit=1.0)

            # If most of the variance in the original data can't be explained by a small
            # subset of the eigen values in the svd decomposition then our low rank approximations
            # won't perform very well.
            @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
        end
    end
end
