# =============================================================================
# Faster orbital-optimized UNRESTRICTED Hubbard BTS solver.
# GRADIENT :
#   * ETA block (P_eta params): ANALYTICAL row-spike gradient. 
# =============================================================================

using FermiPairing
using Combinatorics
#using BTS

using LinearAlgebra
using Optim
using Optim: converged, iterations, minimum, minimizer
using Printf
using Random

using ForwardDiff
using DiffResults

# -----------------------------------------------------------------------------
# 1. T matrix 
# -----------------------------------------------------------------------------
function build_T_matrix(m::Int, n::Int, boundary_condition::Symbol, t::Real=1.0)
    @assert m >= 1 && n >= 1 "Lattice dimensions must be positive integers."
    Lx, Ly = m, n
    N = m * n
    T = zeros(Float64, N, N)
    index_2D(x, y) = x + (y - 1) * Lx

    for y in 1:Ly, x in 1:Lx
        i = index_2D(x, y)
        if x < Lx
            j = index_2D(x + 1, y); T[i, j] = -t; T[j, i] = -t
        end
        if y < Ly
            j = index_2D(x, y + 1); T[i, j] = -t; T[j, i] = -t
        end
    end
    if boundary_condition == :PBC
        if Lx > 1
            for y in 1:Ly
                i = index_2D(Lx, y); j = index_2D(1, y); T[i, j] = -t; T[j, i] = -t
            end
        end
        if Ly > 1
            for x in 1:Lx
                i = index_2D(x, Ly); j = index_2D(x, 1); T[i, j] = -t; T[j, i] = -t
            end
        end
    end
    return T
end

# -----------------------------------------------------------------------------
# 2. U = exp(A),  A anti-Hermitian
# -----------------------------------------------------------------------------
_scal(x::Real) = float(x)
_scal(x::ForwardDiff.Dual) = float(ForwardDiff.value(x))   # strip dual for scaling count

function build_antiherm(N::Int, p::AbstractVector{T}) where {T}
    A = zeros(Complex{T}, N, N)
    idx = 1
    @inbounds for i in 1:N
        A[i, i] = Complex(zero(T), p[idx]); idx += 1
    end
    @inbounds for i in 1:N, j in (i + 1):N
        re = p[idx]; im = p[idx + 1]; idx += 2
        A[i, j] = Complex(re, im)
        A[j, i] = Complex(-re, im)          # = -conj(A[i,j])
    end
    return A
end

function expm_ss(A::AbstractMatrix{Tc}; order::Int = 12) where {Tc}
    N = size(A, 1)
    amax = 0.0
    @inbounds for x in A
        amax = max(amax, _scal(abs(x)))
    end
    nrm = N * amax
    s = nrm > 1.0 ? (ceil(Int, log2(nrm)) + 1) : 0     # ||A/2^s|| <~ 0.5
    As = A / (2.0^s)
    E = Matrix{Tc}(I, N, N) + As
    term = As
    @inbounds for k in 2:order
        term = (term * As) / k
        E = E + term
    end
    for _ in 1:s
        E = E * E
    end
    return E
end

unitary(N, p) = expm_ss(build_antiherm(N, p))

# -----------------------------------------------------------------------------
# 3. Unrestricted-Hubbard coefficient package
# -----------------------------------------------------------------------------
function uhub_coeffs(T::AbstractMatrix, U_up::AbstractMatrix, U_dn::AbstractMatrix)
    P  = U_up .* U_dn
    C  = P * P'                                  
    Qu = abs2.(U_up)
    Qd = abs2.(U_dn)
    D  = Qu * transpose(Qd)                       
    @inbounds for r in 1:size(D, 1)
        D[r, r] = zero(eltype(D))                 
    end
    Eu = real.(diag(U_up * T * U_up'))
    Ed = real.(diag(U_dn * T * U_dn'))
    return (Eu .+ Ed), C, D
end

# -----------------------------------------------------------------------------
# 4. Problem container
# -----------------------------------------------------------------------------
struct UHubProblem
    T::Matrix{Float64}
    U::Float64
    N::Int          # orbitals / sites = mb*nb
    m::Int          # BTS: total levels
    n::Int          # BTS: total pairs
    mb::Int
    nb::Int
    P_eta::Int
    P_A::Int        
    eta_pos::Vector{Tuple{Int,Int}}    
end


function _eta_positions(m, n, P)
    pos = Vector{Tuple{Int,Int}}(undef, P)
    for a in 1:P
        e = zeros(Float64, P); e[a] = 1.0
        M = FermiPairing.btpvectomat(e, m, n)
        c = argmax(abs.(M))
        pos[a] = (c[1], c[2])
    end
    return pos
end

function UHubProblem(m, n, mb, nb, U; bound::Symbol = :OBC)
    N = mb * nb
    P_eta = FermiPairing.btpdim(m, n)
    UHubProblem(build_T_matrix(mb, nb, bound, 1.0), float(U),
                N, m, n, mb, nb, P_eta, N * N, _eta_positions(m, n, P_eta))
end

# -----------------------------------------------------------------------------
# 5. DIAGONAL BTS matrix elements
# -----------------------------------------------------------------------------
@inline function bts_core(prob::UHubProblem, eta_vec::AbstractVector)
    eta_mat = FermiPairing.btpvectomat(eta_vec, prob.m, prob.n)
    ov  = FermiPairing.btsoverlap(eta_mat)
    N   = prob.N
    Z11 = [FermiPairing.btsz11(eta_mat, r)          for r in 1:N]
    Z02 = [FermiPairing.btsz02(eta_mat, (r, s))     for r in 1:N, s in 1:N]
    Z22 = [FermiPairing.btsz22(eta_mat, (r, s))     for r in 1:N, s in 1:N]
    return eta_mat, ov, Z11, Z02, Z22
end

@inline function _numerator(U, E_eff, C, D, Z11, Z02, Z22)
    return 0.5 * sum(E_eff .* Z11) + U * sum(C .* Z02) + (U / 4) * sum(D .* Z22)
end

# -----------------------------------------------------------------------------
# 6. Objective pieces
# -----------------------------------------------------------------------------
# (a) orbital-only energy with the BTS core FROZEN -> ForwardDiff target.
function orb_energy(p, prob::UHubProblem, Z11, Z02, Z22, ov; penalty::Bool = false)
    N, PA = prob.N, prob.P_A
    U_up = unitary(N, @view p[1:PA])
    U_dn = unitary(N, @view p[PA+1:2PA])
    E_eff, C, D = uhub_coeffs(prob.T, U_up, U_dn)
    val = _numerator(prob.U, E_eff, C, D, Z11, Z02, Z22) / ov
    return penalty ? real(val) + abs2(imag(val)) : real(val)
end

# (b) eta-only energy with coefficients FROZEN -> used by the FD eta gradient.
function eta_energy(prob::UHubProblem, eta_vec, E_eff, C, D; penalty::Bool = false)
    _, ov, Z11, Z02, Z22 = bts_core(prob, eta_vec)
    abs(ov) < 1e-12 && return 1000.0
    val = _numerator(prob.U, E_eff, C, D, Z11, Z02, Z22) / ov
    return penalty ? real(val) + abs2(imag(val)) : real(val)
end

# (c) plain full-energy evaluator (value only)
function energy_value(prob::UHubProblem, x; penalty::Bool = false)
    PA, PE, N = prob.P_A, prob.P_eta, prob.N
    _, ov, Z11, Z02, Z22 = bts_core(prob, @view x[1:PE])
    abs(ov) < 1e-12 && return 1000.0
    return orb_energy(x[PE+1:PE+2PA], prob, Z11, Z02, Z22, ov; penalty = penalty)
end

# -----------------------------------------------------------------------------
# 6.5  BTS TRANSITION matrix elements.
# -----------------------------------------------------------------------------
bts_ovlp_T(bra, ket)      = FermiPairing.btsbraket(bra, ket)
bts_z11_T(bra, ket, r)    = FermiPairing.btstz11(bra, ket, r)
bts_z02_T(bra, ket, rs)   = FermiPairing.btstz02(bra, ket, rs)
bts_z22_T(bra, ket, rs)   = FermiPairing.btstz22(bra, ket, rs)

# Analytical eta gradient (row-spike)
function eta_grad_analytic!(Geta, prob::UHubProblem, eta_vec, E_eff, C, D, ov, E)
    m, n, N, U = prob.m, prob.n, prob.N, prob.U
    eta_mat = FermiPairing.btpvectomat(eta_vec, m, n)
    ovr = real(ov)
    Tel = eltype(eta_mat)
    @inbounds for a in 1:prob.P_eta
        k, p = prob.eta_pos[a]
        ket = copy(eta_mat)
        for c in 1:size(ket, 2); ket[k, c] = zero(Tel); end
        ket[k, p] = one(Tel)                      

        S = bts_ovlp_T(eta_mat, ket)
        H = zero(ComplexF64)
        for r in 1:N
            H += 0.5 * E_eff[r] * bts_z11_T(eta_mat, ket, r)
        end
        for r in 1:N, s in 1:N
            H += U * C[r, s] * bts_z02_T(eta_mat, ket, (r, s))
            r != s && (H += U * (D[r, s] / 4) * bts_z22_T(eta_mat, ket, (r, s)))
        end
        Geta[a] = (2 / ovr) * real(H - E * S)
    end
    return Geta
end

# Central-difference eta gradient
function eta_grad_fd!(Geta, prob::UHubProblem, eta_vec, E_eff, C, D;
                      penalty::Bool = false, h::Float64 = 1e-6)
    ev = copy(eta_vec)
    @inbounds for a in 1:prob.P_eta
        ev[a] += h; fp = eta_energy(prob, ev, E_eff, C, D; penalty = penalty)
        ev[a] -= 2h; fm = eta_energy(prob, ev, E_eff, C, D; penalty = penalty)
        ev[a] += h; Geta[a] = (fp - fm) / (2h)
    end
    return Geta
end

# -----------------------------------------------------------------------------
# 7. Combined value + gradient.
#    orbital block: ForwardDiff. eta block: analytical (plain) or FD (penalty).
# -----------------------------------------------------------------------------
function make_fg(prob::UHubProblem; penalty::Bool = false, fd_step::Float64 = 1e-6,
                 eta_grad::Symbol = :analytic)
    PA, PE, N = prob.P_A, prob.P_eta, prob.N

    function fg!(F, G, x)
        eta_vec = x[1:PE]
        _, ov, Z11, Z02, Z22 = bts_core(prob, eta_vec)
        if abs(ov) < 1e-12
            G !== nothing && fill!(G, 0.0)
            return 1000.0
        end
        orb = x[PE+1:PE+2PA]

        if G === nothing
            return orb_energy(orb, prob, Z11, Z02, Z22, ov; penalty = penalty)
        end

        # orbital gradient 
        res  = DiffResults.GradientResult(orb)
        gfun = q -> orb_energy(q, prob, Z11, Z02, Z22, ov; penalty = penalty)
        res  = ForwardDiff.gradient!(res, gfun, orb)
        val  = DiffResults.value(res)
        @views G[PE+1:PE+2PA] .= DiffResults.gradient(res)

        # eta gradient
        U_up = unitary(N, @view orb[1:PA])
        U_dn = unitary(N, @view orb[PA+1:2PA])
        E_eff, C, D = uhub_coeffs(prob.T, U_up, U_dn)
        Geta = @view G[1:PE]
        if eta_grad === :analytic && !penalty
            eta_grad_analytic!(Geta, prob, eta_vec, E_eff, C, D, ov, val)
        else
            eta_grad_fd!(Geta, prob, eta_vec, E_eff, C, D; penalty = penalty, h = fd_step)
        end
        return val
    end
    return fg!
end


function make_obj(prob::UHubProblem; penalty::Bool = false, fd_step::Float64 = 1e-6,
                  eta_grad::Symbol = :analytic)
    fg! = make_fg(prob; penalty = penalty, fd_step = fd_step, eta_grad = eta_grad)
    cache_x = Float64[]
    cache_f = Ref(0.0)
    cache_g = Float64[]
    function ensure!(x)
        if length(cache_x) != length(x) || cache_x != x
            g = similar(x)
            f = fg!(true, g, x)
            cache_x = copy(x); cache_f[] = f; cache_g = g
        end
        return nothing
    end
    fval(x)     = (ensure!(x); cache_f[])
    gval!(G, x) = (ensure!(x); copyto!(G, cache_g); G)
    return fval, gval!
end

# -----------------------------------------------------------------------------
# 8. Final complex energy + matrices for reporting
# -----------------------------------------------------------------------------
function final_report(prob::UHubProblem, x)
    PA, PE, N = prob.P_A, prob.P_eta, prob.N
    eta_vec = x[1:PE]
    U_up = unitary(N, x[PE+1:PE+PA])
    U_dn = unitary(N, x[PE+PA+1:PE+2PA])
    E_eff, C, D = uhub_coeffs(prob.T, U_up, U_dn)
    eta_mat, ov, Z11, Z02, Z22 = bts_core(prob, eta_vec)
    E_complex = _numerator(prob.U, E_eff, C, D, Z11, Z02, Z22) / ov
    return E_complex, eta_mat, U_up, U_dn
end

# -----------------------------------------------------------------------------
# 9. Solver
# -----------------------------------------------------------------------------
function btsminhubbard_unrestricted_corrected(m::Int, n::Int, mb::Int, nb::Int,
                                    hamp::Float64, x_init::AbstractVector;
                                    niters::Int = 3000, bound::Symbol = :OBC,
                                    penalty_polish::Bool = true,
                                    fd_step::Float64 = 1e-6, seed::Int = 0,
                                    eta_grad::Symbol = :auto,
                                    n_starts::Int = 4,
                                    amp_orb::Float64 = 0.4, amp_eta::Float64 = 0.05)
    rng  = MersenneTwister(seed)
    prob = UHubProblem(m, n, mb, nb, hamp; bound = bound)
    PE, PA = prob.P_eta, prob.P_A

    if eta_grad === :auto
        eta_grad = hasmethod(FermiPairing.btstz02, Tuple{AbstractMatrix, AbstractMatrix, Tuple}) ?
                   :analytic : :fd
        @info "eta-gradient mode" eta_grad
    end

    f_plain, g_plain = make_obj(prob; penalty = false, fd_step = fd_step, eta_grad = eta_grad)
    f_pen,   g_pen   = make_obj(prob; penalty = true,  fd_step = fd_step, eta_grad = eta_grad)

    lbfgs = LBFGS(m = 20, linesearch = Optim.LineSearches.BackTracking())
    opts(it; g = 1e-8) = Optim.Options(iterations = it, g_tol = g,
                                       allow_f_increases = true, show_trace = false)
    Eplain(res) = energy_value(prob, Optim.minimizer(res))

    @printf("----------------\nBTS solve  U = %.4f\n", hamp)

    # Phase A -- warm start
    best   = optimize(f_plain, g_plain, collect(float.(x_init)), lbfgs, opts(2 * niters))
    Ebest  = Eplain(best)
    eta0   = collect(float.(x_init[1:PE]))
    coarse = clamp(niters ÷ 20, 200, 2000)

    for s in 1:n_starts
        scale = amp_orb * (0.25 + 1.75 * (s - 1) / max(1, n_starts - 1))
        x0 = zeros(PE + 2PA)
        x0[1:PE]          .= eta0 .+ amp_eta .* randn(rng, PE)
        x0[PE+1:PE+PA]    .= scale .* randn(rng, PA)      # A_up
        x0[PE+PA+1:end]   .= scale .* randn(rng, PA)      # A_dn (independent => spin broken)
        r = optimize(f_plain, g_plain, x0, lbfgs, opts(coarse))
        Eplain(r) < Ebest - 1e-12 && (best = r; Ebest = Eplain(best))
    end

    # Phase B -- polish 
    r = optimize(f_plain, g_plain, copy(Optim.minimizer(best)), lbfgs, opts(4 * niters; g = 1e-11))
    Eplain(r) < Ebest - 1e-13 && (best = r; Ebest = Eplain(best))

    # Phase C -- polish
    if penalty_polish
        rp = optimize(f_pen, g_pen, copy(Optim.minimizer(best)), lbfgs, opts(niters))
        Eplain(rp) < Ebest - 1e-13 && (best = rp; Ebest = Eplain(best))
    end

    xf = copy(Optim.minimizer(best))
    E_complex, eta_mat, U_up, U_dn = final_report(prob, xf)
    @printf("Final energy = %.10f   (imag leakage %.1e)\n\n",
            real(E_complex), abs(imag(E_complex)))
    return real(E_complex), eta_mat, xf, U_up, U_dn
end

# -----------------------------------------------------------------------------
# 10. VALIDATION GATES
# -----------------------------------------------------------------------------
function reference_energy_loops(prob::UHubProblem, x; penalty::Bool = false)
    PA, PE, N = prob.P_A, prob.P_eta, prob.N
    eta_vec = x[1:PE]
    U_up = unitary(N, x[PE+1:PE+PA]); U_dn = unitary(N, x[PE+PA+1:PE+2PA])
    Eu = real.(diag(U_up * prob.T * U_up')); Ed = real.(diag(U_dn * prob.T * U_dn'))
    Eeff = Eu .+ Ed
    C = zeros(ComplexF64, N, N); D = zeros(ComplexF64, N, N)
    for r in 1:N, s in 1:N
        sc = 0.0 + 0im; sd = 0.0 + 0im
        for i in 1:N
            por = U_up[r, i] * U_dn[r, i]; pos = U_up[s, i] * U_dn[s, i]
            sc += por * conj(pos)
            r != s && (sd += abs2(U_up[r, i]) * abs2(U_dn[s, i]))
        end
        C[r, s] = sc; D[r, s] = sd
    end
    eta_mat = FermiPairing.btpvectomat(eta_vec, prob.m, prob.n)
    ov = FermiPairing.btsoverlap(eta_mat)
    H = 0.0 + 0im
    for r in 1:N
        H += Eeff[r] * 0.5 * FermiPairing.btsz11(eta_mat, r)
    end
    for r in 1:N, s in 1:N
        H += prob.U * C[r, s] * FermiPairing.btsz02(eta_mat, (r, s))
        r != s && (H += prob.U * (D[r, s] / 4) * FermiPairing.btsz22(eta_mat, (r, s)))
    end
    val = H / ov
    return penalty ? real(val) + abs2(imag(val)) : real(val)
end


function check_eta_gradient(prob::UHubProblem, x; h::Float64 = 1e-6)
    PE, PA, N = prob.P_eta, prob.P_A, prob.N
    eta_vec = x[1:PE]
    U_up = unitary(N, x[PE+1:PE+PA]); U_dn = unitary(N, x[PE+PA+1:PE+2PA])
    E_eff, C, D = uhub_coeffs(prob.T, U_up, U_dn)
    _, ov, Z11, Z02, Z22 = bts_core(prob, eta_vec)
    E = real(_numerator(prob.U, E_eff, C, D, Z11, Z02, Z22) / ov)
    ga = zeros(PE); gf = zeros(PE)
    eta_grad_analytic!(ga, prob, eta_vec, E_eff, C, D, ov, E)
    eta_grad_fd!(gf, prob, eta_vec, E_eff, C, D; h = h)
    return maximum(abs, ga .- gf), ga, gf
end


function check_full_gradient(prob::UHubProblem, x; h::Float64 = 1e-6)
    fg! = make_fg(prob; eta_grad = :analytic)
    g = zeros(length(x)); fg!(true, g, x)
    gfd = zeros(length(x))
    for i in eachindex(x)
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        gfd[i] = (energy_value(prob, xp) - energy_value(prob, xm)) / (2h)
    end
    return maximum(abs, g .- gfd), g, gfd
end


# =============================================================================
# 11. Driver: unrestricted Hubbard U-sweep
# =============================================================================
function run_scan(; m = 16, n = 8, mb = 4, nb = 4, bc = :OBC,
                    niters = 100000, hpinit = 0.0, dhp = 0.5, nP = 21,
                    n_starts = 4, outdir = ".")
    println("Start Unrestricted BTS Scan")
    N_sites    = mb * nb
    P_eta      = FermiPairing.btpdim(m, n)
    P_A_single = N_sites * N_sites

    ebts = zeros(Float64, nP)
    Gval = float(hpinit)

    eta_guess_agp = ones(Float64, N_sites)
    for p in 2:2:N_sites; eta_guess_agp[p] *= -1; end     # staggered (Neel-like)
    bts_eta_guess = FermiPairing.btsfromagp(eta_guess_agp, n)
    A_up_guess = zeros(Float64, P_A_single) .+ 0.001
    A_dn_guess = zeros(Float64, P_A_single) .- 0.001
    current_minimizer = [bts_eta_guess; A_up_guess; A_dn_guess]

    f_out   = open(joinpath(outdir, "EUnHubBTS16.txt"),   "w")
    eta_out = open(joinpath(outdir, "EtaUnHubBTS16.txt"), "w")
    U_out   = open(joinpath(outdir, "UmatrixUnHubBTS16.txt"), "w")
    println(f_out, "Hubbard Unrestricted, m: $m, n: $n"); println(f_out, "-------------------------")

    for j in 1:nP
        E, btsmat, next_x, U_up, U_dn =
            btsminhubbard_unrestricted_corrected(m, n, mb, nb, Gval, current_minimizer;
                                       niters = niters, bound = bc, n_starts = n_starts)
        ebts[j] = E
        println("GorD: $Gval")
        println(f_out,   "Energy: $(ebts[j])")
        println(eta_out, "Step $j, Gval $Gval:"); println(eta_out, next_x)
        println(U_out,   "Step $j, Gval $Gval:"); println(U_out, "U_up: $U_up"); println(U_out, "U_down: $U_dn")
        flush(f_out); flush(eta_out); flush(U_out)

        if j < nP
            Gval += dhp
            current_minimizer = copy(next_x)
            current_minimizer[(P_eta + 1):end] .+= 1e-5 .* randn(2 * P_A_single)
        end
    end
    close(f_out); close(eta_out); close(U_out)
    println("Check output files.")
    return ebts
end

# =============================================================================
# 12. Bidirectional sweep 
# =============================================================================
function run_scan_bidirectional(; m = 8, n = 4, mb = 8, nb = 1, bc = :OBC,
                                  niters = 100000, Umin = 0.0, Umax = 10.0, nP = 21,
                                  n_starts = 4, break_amp = 0.7, outdir = ".")
    N_sites = mb * nb
    P_eta   = FermiPairing.btpdim(m, n)
    P_A     = N_sites * N_sites
    Us      = collect(range(Umin, Umax, length = nP))

    agp = ones(Float64, N_sites); for p in 2:2:N_sites; agp[p] *= -1; end
    eta0 = FermiPairing.btsfromagp(agp, n)
    seedx(strength) = [eta0; strength .* randn(2 * P_A)]
    solve(U, x) = btsminhubbard_unrestricted_corrected(m, n, mb, nb, U, x;
                        niters = niters, bound = bc, n_starts = n_starts)

    E_up = zeros(nP)
    x = seedx(0.01)
    for j in 1:nP
        E, _, xf, _, _ = solve(Us[j], x); E_up[j] = E
        x = copy(xf); x[P_eta+1:end] .+= 1e-4 .* randn(2 * P_A)
    end

    E_dn = zeros(nP)
    x = seedx(break_amp)
    for j in nP:-1:1
        E, _, xf, _, _ = solve(Us[j], x); E_dn[j] = E
        x = copy(xf); x[P_eta+1:end] .+= 1e-4 .* randn(2 * P_A)
    end

    E_best = min.(E_up, E_dn)
    open(joinpath(outdir, "EUnHubBTS_bidir.txt"), "w") do io
        println(io, "U    E_best    E_up    E_down")
        for j in 1:nP
            @printf(io, "%.4f  %.10f  %.10f  %.10f\n", Us[j], E_best[j], E_up[j], E_dn[j])
        end
    end
    return Us, E_best, E_up, E_dn
end


# =============================================================================
# 13. ROBUST 1D NEEL-PAIRED INITIALIZATION
# =============================================================================

# Pack an explicitly supplied anti-Hermitian matrix into the N^2 parameter
function antiherm_to_params(A::AbstractMatrix)
    N = size(A, 1)
    @assert size(A, 2) == N
    p = zeros(Float64, N * N)
    idx = 1
    @inbounds for i in 1:N
        p[idx] = imag(A[i, i]); idx += 1
    end
    @inbounds for i in 1:N, j in (i + 1):N
        p[idx]     = real(A[i, j])
        p[idx + 1] = imag(A[i, j])
        idx += 2
    end
    return p
end

function nn_swap_params(N::Int; theta::Float64 = pi / 2)
    @assert iseven(N) "The half-filled Neel seed below assumes even N."
    A = zeros(ComplexF64, N, N)
    for i in 1:2:(N - 1)
        A[i, i + 1] = theta
        A[i + 1, i] = -theta
    end
    return antiherm_to_params(A)
end

function neel_eta_seed(m::Int, n::Int; parity::Symbol = :odd,
                       background::Float64 = 1e-3)
    @assert m == 2n "This helper is intended for half-filled even chains."
    P = FermiPairing.btpdim(m, n)
    v = fill(background, P)
    M = FermiPairing.btpvectomat(v, m, n)
    targets = parity === :odd ? collect(1:2:(m - 1)) :
              parity === :even ? collect(2:2:m) :
              error("parity must be :odd or :even")
    @assert length(targets) == n
    for j in 1:n
        M[j, targets[j]] = 1.0
    end
    return FermiPairing.btpmattovec(M)
end

# Build one strong-coupling seed
function neel_pair_seed_1d(N::Int, n::Int;
                           parity::Symbol = :odd,
                           swap_spin::Symbol = :down,
                           theta::Float64 = pi / 2,
                           eta_background::Float64 = 1e-3)
    eta = neel_eta_seed(N, n; parity = parity,
                       background = eta_background)
    z = zeros(Float64, N * N)
    sw = nn_swap_params(N; theta = theta)
    if swap_spin === :down
        return [eta; z; sw]
    elseif swap_spin === :up
        return [eta; sw; z]
    else
        error("swap_spin must be :up or :down")
    end
end


function best_of_ubts_seeds(m::Int, n::Int, mb::Int, nb::Int, U::Float64,
                            seeds::Vector{Vector{Float64}};
                            niters::Int = 100000,
                            bc::Symbol = :OBC,
                            n_starts::Int = 6,
                            amp_orb::Float64 = 0.8,
                            amp_eta::Float64 = 0.08,
                            rng_seed::Int = 1000)
    bestE = Inf
    bestmat = nothing
    bestx = nothing
    bestUu = nothing
    bestUd = nothing

    for (k, x0) in enumerate(seeds)
        E, btsmat, xf, Uu, Ud = btsminhubbard_unrestricted_corrected(
            m, n, mb, nb, U, x0;
            niters = niters,
            bound = bc,
            n_starts = n_starts,
            amp_orb = amp_orb,
            amp_eta = amp_eta,
            seed = rng_seed + k,
        )
        @printf("seed %d at U=%7.3f -> %.12f\n", k, U, E)
        if E < bestE
            bestE, bestmat, bestx, bestUu, bestUd = E, btsmat, copy(xf), Uu, Ud
        end
    end
    return bestE, bestmat, bestx, bestUu, bestUd
end

# Repolish 
function repolish_continuation_point(m::Int, n::Int, mb::Int, nb::Int,
                                     U::Float64, xwarm::Vector{Float64};
                                     niters::Int = 100000,
                                     bc::Symbol = :OBC,
                                     n_starts::Int = 4,
                                     n_local::Int = 4,
                                     rng_seed::Int = 2000)
    N = mb * nb
    PE = FermiPairing.btpdim(m, n)
    PA = N * N
    rng = MersenneTwister(rng_seed)

    seeds = Vector{Vector{Float64}}()
    push!(seeds, copy(xwarm))

    # stronger perturbations
    amps_orb = (1e-5, 1e-4, 1e-3, 5e-3)
    amps_eta = (1e-6, 1e-5, 1e-4, 5e-4)
    for k in 1:min(n_local, length(amps_orb))
        x = copy(xwarm)
        x[1:PE] .+= amps_eta[k] .* randn(rng, PE)
        x[PE+1:PE+2PA] .+= amps_orb[k] .* randn(rng, 2PA)
        push!(seeds, x)
    end

    return best_of_ubts_seeds(m, n, mb, nb, U, seeds;
        niters = niters, bc = bc, n_starts = n_starts,
        amp_orb = 0.8, amp_eta = 0.08, rng_seed = rng_seed + 100)
end


# =============================================================================
# 14. Attractive strong-coupling anchor seeds
# =============================================================================
function attractive_pair_seeds_1d(N::Int, n::Int;
                                  nseed::Int = 8,
                                  rng_seed::Int = 271828)
    @assert nseed >= 1
    P_eta = FermiPairing.btpdim(N, n)
    P_A = N * N
    rng = MersenneTwister(rng_seed)

    agp_plus = ones(Float64, N)
    eta_plus = FermiPairing.btsfromagp(agp_plus, n)

    agp_alt = ones(Float64, N)
    for p in 2:2:N
        agp_alt[p] *= -1
    end
    eta_alt = FermiPairing.btsfromagp(agp_alt, n)

    seeds = Vector{Vector{Float64}}()

    # Two clean restricted seeds
    z = zeros(Float64, P_A)
    push!(seeds, [eta_plus; z; z])
    nseed == 1 && return seeds
    push!(seeds, [eta_alt; z; z])
    nseed == 2 && return seeds

    strengths = (0.01, 0.03, 0.08, 0.15, 0.30, 0.60)
    for k in 1:(nseed - 2)
        s = strengths[min(k, length(strengths))]
        A = s .* randn(rng, P_A)
        eta0 = isodd(k) ? copy(eta_plus) : copy(eta_alt)
        eta0 .+= 1e-3 .* randn(rng, P_eta)
        push!(seeds, [eta0; A; A])
    end

    return seeds
end

# =============================================================================
# 15. Output -- energy, eta, U matrices, and exact restart vectors
# =============================================================================
function write_ubts_branch(rows, N::Int, n::Int, bc::Symbol,
                           branch_name::AbstractString,
                           continuation_note::AbstractString,
                           outdir::AbstractString)
    P_eta = FermiPairing.btpdim(N, n)
    bcstr = String(bc)
    tag = "3x2_$(bcstr)_$(branch_name)"

    energy_file = joinpath(outdir, "EUnHubBTS_$(tag).txt")
    eta_file = joinpath(outdir, "EtaUnHubBTS_$(tag).txt")
    umat_file = joinpath(outdir, "UmatrixUnHubBTS_$(tag).txt")
    restart_file = joinpath(outdir, "RestartUnHubBTS_$(tag).txt")

    open(energy_file, "w") do io
        println(io, "# unrestricted BTS, 6x1 $(bcstr), half-filled (5 pairs), t=1")
        println(io, "# $continuation_note")
        println(io, "# U/t                 E_UBTS/t")
        for r in rows
            @printf(io, "%6.1f    %.12f\n", r.U, r.E)
        end
    end

    open(eta_file, "w") do io
        for (j, r) in enumerate(rows)
            println(io, "Step $j, Gval $(r.U):")
            println(io, "Energy = $(r.E)")
            println(io, "eta_vector:")
            println(io, r.x[1:P_eta])
            println(io, "eta_matrix:")
            show(io, "text/plain", r.eta)
            println(io, "\n")
        end
    end

    # unrestricted orbital matrices
    open(umat_file, "w") do io
        for (j, r) in enumerate(rows)
            println(io, "Step $j, Gval $(r.U):")
            println(io, "Energy = $(r.E)")
            println(io, "U_up:")
            show(io, "text/plain", r.Uu)
            println(io)
            println(io, "U_down:")
            show(io, "text/plain", r.Ud)
            println(io, "\n")
        end
    end

    open(restart_file, "w") do io
        for (j, r) in enumerate(rows)
            println(io, "Step $j, Gval $(r.U):")
            println(io, "Energy = $(r.E)")
            println(io, "x = ", r.x)
            println(io, "U_up = ", r.Uu)
            println(io, "U_down = ", r.Ud)
            println(io)
        end
    end

    println("Wrote branch files:")
    println("  ", energy_file)
    println("  ", eta_file)
    println("  ", umat_file)
    println("  ", restart_file)

    return energy_file, eta_file, umat_file, restart_file
end

# =============================================================================
# 16. Corrected repulsive branch: U = +10 -> 0
# =============================================================================
function run_repulsive_corrected(; Umax::Float64 = 10.0,
                                        Umin::Float64 = 0.0,
                                        dU::Float64 = 1.0,
                                        niters::Int = 100000,
                                        anchor_starts::Int = 8,
                                        point_starts::Int = 4,
                                        n_local::Int = 4,
                                        bc::Symbol = :OBC,
                                        outdir::AbstractString = ".")
    N  = 16
    n  = 8
    mb = 4
    nb = 4

    mkpath(outdir)
    Us = collect(Umax:-dU:Umin)

    # Eight strong-coupling Neel-compatible seeds
    anchor_seeds = Vector{Vector{Float64}}()
    for parity in (:odd, :even), swap_spin in (:down, :up), sgn in (1.0, -1.0)
        push!(anchor_seeds,
              neel_pair_seed_1d(N, n;
                  parity = parity,
                  swap_spin = swap_spin,
                  theta = sgn * pi / 2,
                  eta_background = 1e-3))
    end

    println("============================================================")
    println("UBTS  $(bc) half-filled REPULSIVE scan")
    println("Continuation: U/t = $Umax -> $Umin in steps of $dU")
    println("Eight Neel-compatible anchor seeds at U/t = $Umax")
    println("============================================================")

    E, btsmat, xbest, Uu, Ud = best_of_ubts_seeds(
        N, n, mb, nb, Umax, anchor_seeds;
        niters = niters, bc = bc, n_starts = anchor_starts,
        amp_orb = 1.0, amp_eta = 0.10, rng_seed = 314159)

    rows = Vector{NamedTuple}()
    push!(rows, (U = Umax, E = E, eta = copy(btsmat),
                 x = copy(xbest), Uu = copy(Uu), Ud = copy(Ud)))

    xwarm = copy(xbest)
    for (istep, U) in enumerate(Us[2:end])
        println("\n============================================================")
        @printf("Repulsive continuation point U/t = %.3f\n", U)
        println("============================================================")

        E, btsmat, xbest, Uu, Ud = repolish_continuation_point(
            N, n, mb, nb, U, xwarm;
            niters = niters, bc = bc,
            n_starts = point_starts,
            n_local = n_local,
            rng_seed = 400000 + istep * 100)

        push!(rows, (U = U, E = E, eta = copy(btsmat),
                     x = copy(xbest), Uu = copy(Uu), Ud = copy(Ud)))
        xwarm = copy(xbest)
    end

    write_ubts_branch(rows, N, n, bc,
                      "repulsive",
                      "Strong-U Neel-compatible anchor; continuation U=+10 -> 0",
                      outdir)
    return rows
end

# =============================================================================
# 17. Corrected attractive branch: U = -10 -> 0
# =============================================================================
function run_attractive_corrected(; Umin::Float64 = -10.0,
                                         Umax::Float64 = 0.0,
                                         dU::Float64 = 1.0,
                                         niters::Int = 100000,
                                         anchor_starts::Int = 8,
                                         point_starts::Int = 4,
                                         n_local::Int = 4,
                                         bc::Symbol = :OBC,
                                         outdir::AbstractString = ".")
    N  = 16
    n  = 8
    mb = 4
    nb = 4

    mkpath(outdir)
    Us = collect(Umin:dU:Umax)

    # Eight paired/restricted seeds
    anchor_seeds = attractive_pair_seeds_1d(N, n;
                                             nseed = 8,
                                             rng_seed = 271828)

    println("\n\n============================================================")
    println("UBTS $(bc) half-filled ATTRACTIVE scan")
    println("Continuation: U/t = $Umin -> $Umax in steps of $dU")
    println("Eight paired/restricted anchor seeds at U/t = $Umin")
    println("============================================================")

    E, btsmat, xbest, Uu, Ud = best_of_ubts_seeds(
        N, n, mb, nb, Umin, anchor_seeds;
        niters = niters, bc = bc, n_starts = anchor_starts,
        amp_orb = 0.6, amp_eta = 0.08, rng_seed = 161803)

    rows = Vector{NamedTuple}()
    push!(rows, (U = Umin, E = E, eta = copy(btsmat),
                 x = copy(xbest), Uu = copy(Uu), Ud = copy(Ud)))

    xwarm = copy(xbest)
    for (istep, U) in enumerate(Us[2:end])
        println("\n============================================================")
        @printf("Attractive continuation point U/t = %.3f\n", U)
        println("============================================================")

        E, btsmat, xbest, Uu, Ud = repolish_continuation_point(
            N, n, mb, nb, U, xwarm;
            niters = niters, bc = bc,
            n_starts = point_starts,
            n_local = n_local,
            rng_seed = 700000 + istep * 100)

        push!(rows, (U = U, E = E, eta = copy(btsmat),
                     x = copy(xbest), Uu = copy(Uu), Ud = copy(Ud)))
        xwarm = copy(xbest)
    end

    write_ubts_branch(rows, N, n, bc,
                      "attractive",
                      "Strong-attractive paired anchor; continuation U=-10 -> 0",
                      outdir)
    return rows
end

# =============================================================================
# 18. Run BOTH independent branches
# =============================================================================
function run_both_branches(; bc::Symbol = :OBC,
                                  niters::Int = 100000,
                                  anchor_starts::Int = 8,
                                  point_starts::Int = 4,
                                  n_local::Int = 4,
                                  outdir::AbstractString = ".")
    rep = run_repulsive_corrected(
        bc = bc, niters = niters,
        anchor_starts = anchor_starts,
        point_starts = point_starts,
        n_local = n_local,
        outdir = outdir)

    att = run_attractive_corrected(
        bc = bc, niters = niters,
        anchor_starts = anchor_starts,
        point_starts = point_starts,
        n_local = n_local,
        outdir = outdir)

    return rep, att
end

#run_both_branches()
