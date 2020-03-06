using KernelAbstractions, Test, CUDAapi
if has_cuda_gpu()
    using CuArrays, CUDAdrv
    CuArrays.allowscalar(false)
end

function GPU_copy_test(M)

    A = CuArray(rand(Float64, M))
    B = CuArray(rand(Float64, M))

    a = Array{Float64}(undef, M)
    pin!(a)

    len = length(A)

    copystream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
    copyevent = recordevent(copystream)
    copyevent = async_copy!(CUDA(), pointer(A), pointer(B),
                            M, stream=copystream)

    wait(copyevent)

    copyevent = async_copy!(CUDA(), a, B, stream=copystream)
    wait(copyevent)
    copyevent = async_copy!(CUDA(), A, a, stream=copystream,
                            dependencies=copyevent)
    wait(copyevent)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end

function CPU_copy_test(M)
    A = Array(rand(Float64, M))
    B = Array(rand(Float64, M))

    a = Array{Float64}(undef, M)

    copyevent = async_copy!(CPU(), a, B)
    wait(copyevent)
    copyevent = async_copy!(CPU(), A, a)
    wait(copyevent)

    @test isapprox(a, A)
    @test isapprox(a, B)
end

M = 1024

if has_cuda_gpu()
    GPU_copy_test(M)
end
CPU_copy_test(M)
