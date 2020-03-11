using KernelAbstractions, Test, CUDAapi
if CUDAapi.has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function copy_kernel!(a,b)
    I = @index(Global)
    @inbounds b[I] = a[I]
end

# this transpose kernel copies the input into an nxn shared memory tile which is
# then copied to the output, producing a transpose with coalesced reads and
# writes to the shared memory tile. Here, the NDRange needs to be
# (nx/TDIM, ny/TDIM), where TDIM is the Tile dimension.
@kernel function localmem_transpose!(a, b, @Const(TDIM), @Const(BLOCK_ROWS))
    T = eltype(a)

    #tile = @localmem(Float32, (TDIM+1, TDIM))
    tile = @localmem(T, (TDIM + 1,TDIM))
#=

    # Here, the NDRange has been set up to be (nx/TDIM, ny/TDIM)
    block_index = @index(Global, Cartesian)
    thread_index = @index(Local, Cartesian)

    i = (block_index[1] - 1) *TDIM + thread_index[1]
    j = (block_index[2] - 1) *TDIM + thread_index[2]

    for k = 0:BLOCK_ROWS:TDIM
        @inbounds tile[thread_index[2] + k, thread_index[1]] = in[i, j+k]
    end

    sync_threads()

    i = (block_index[2] - 1) * TDIM + thread_index[1]
    j = (block_index[1] - 1) * TDIM + thread_index[2]
  
    for k = 0:BLOCK_ROWS:TDIM-1
        @inbounds out[i, j + k] = tile[thread_index[1], thread_index[2] + k]
    end
=#

end

# creating wrapper functions
function launch_copy!(a, b)
    if size(a) != size(b)
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, Array)
        kernel! = copy_kernel!(CPU(),4)
    else
        kernel! = copy_kernel!(CUDA(),1024)
    end
    kernel!(a, b, ndrange=size(a))
end

# creating wrapper functions
function launch_localmem_transpose!(a, b)
    if size(a)[1] != size(b)[2] || size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    TDIM = 32
    BLOCK_ROWS = 8
    if isa(a, Array)
        kernel! = localmem_transpose!(CPU(),4)
    else
        kernel! = localmem_transpose!(CUDA(),256)
    end
    kernel!(a, b, TDIM, BLOCK_ROWS, ndrange=(cld.(size(b),TDIM)))
end

function main()

    # resolution of grid will be res*res
    res = 1024

    # creating initial arrays on CPU and GPU
    a = round.(rand(Float32, (res, res))*100)
    b = zeros(Float32, res, res)

    # beginning CPU tests
    ev = launch_copy!(a, b)
    wait(ev)

    ev = launch_localmem_transpose!(a,b)
    #wait(ev)

    println("CPU transpose time is:")
    println("Testing CPU transpose...")
    @test a == transpose(b)

    # beginning GPU tests
    if has_cuda_gpu()
        d_a = CuArray(a)
        d_b = CuArray(zeros(Float32, res, res))

        launch_copy!(d_a, d_b)

        launch_localmem_transpose!(d_a, d_b)

        a = Array(d_a)
        b = Array(d_b)

        println("Testing GPU transpose...")
        @test a == transpose(b)
    end

    return nothing
end

main()

