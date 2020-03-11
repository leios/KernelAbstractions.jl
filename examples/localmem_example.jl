using KernelAbstractions, Test, CUDAapi
if CUDAapi.has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function localmem_check!(a, @Const(TDIM))
    T = eltype(a)
    #i = @index(Global)

    # Fails when waiting on event
    # tile = @localmem(T, (TDIM+1, TDIM))

    # succeeds
    tile = @localmem(Float32, (TDIM+1, TDIM))

end

# creating wrapper functions
function launch_localmem_check(a)
    TDIM = 32
    if isa(a, Array)
        kernel! = localmem_check!(CPU(),4)
    else
        kernel! = localmem_check!(CUDA(),256)
    end
    kernel!(a, TDIM, ndrange=(TDIM, TDIM))
end

function main()

    a = zeros(32, 32)
    ev = launch_localmem_check(a)
    wait(ev)

    if has_cuda_gpu()
        d_a = CuArray(a)

        launch_localmem_check(d_a)
    end

    return nothing
end

main()

