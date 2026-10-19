using LinearAlgebra
using SparseArrays
using Statistics
using Printf
import LinearAlgebra.BLAS: BlasFloat
using Optim
using Optim: converged, iterations, minimum, minimizer
using Random

# ==============================================================================
# SECTION 1 — sumESP core algorithm
# ==============================================================================

function sumESP(X::AbstractVector{T}, N::Int) where T<:Number
    N == 0 && return one(T)
    N < 0  && return zero(T)
    M = length(X)
    N > M && return zero(T)
    S = zeros(T, N + 1)
    S[1] = one(T)
    count = 0
    @inbounds for xi in X
        count += 1
        jmax = min(count, N)
        for j in jmax:-1:1
            S[j + 1] += xi * S[j]
        end
    end
    return S[N + 1]
end

function sumESP_skip1(X::AbstractVector{T}, N::Int, skip::Int) where T<:Number
    N == 0 && return one(T)
    N < 0  && return zero(T)
    M = length(X) - 1
    N > M && return zero(T)
    S = zeros(T, N + 1)
    S[1] = one(T)
    count = 0
    @inbounds for i in eachindex(X)
        i == skip && continue
        count += 1
        jmax = min(count, N)
        for j in jmax:-1:1
            S[j + 1] += X[i] * S[j]
        end
    end
    return S[N + 1]
end

function sumESP_skip2(X::AbstractVector{T}, N::Int, skip1::Int, skip2::Int) where T<:Number
    N == 0 && return one(T)
    N < 0  && return zero(T)
    skip1 == skip2 && return sumESP_skip1(X, N, skip1)
    M = length(X) - 2
    N > M && return zero(T)
    S = zeros(T, N + 1)
    S[1] = one(T)
    count = 0
    @inbounds for i in eachindex(X)
        (i == skip1 || i == skip2) && continue
        count += 1
        jmax = min(count, N)
        for j in jmax:-1:1
            S[j + 1] += X[i] * S[j]
        end
    end
    return S[N + 1]
end

# ==============================================================================
# SECTION 2 — AGP helpers
# ==============================================================================

function rescale_eta(η::AbstractVector{T}) where T<:Real
    ηs = copy(η)
    s  = maximum(abs.(ηs))
    if !isfinite(s) || s < eps(T)
        ηs .= one(T)
        return ηs
    end
    ηs ./= s
    k = argmax(abs.(ηs))
    if ηs[k] < zero(T)
        ηs .*= -one(T)
    end
    return ηs
end

function build_agp_cache(η::AbstractVector{T}, N::Int) where T<:Real
    M    = length(η)
    eta2 = η .^ 2
    ov   = sumESP(eta2, N)
    esp1     = zeros(T, M)
    esp2_Nm1 = zeros(T, M, M)
    esp2_Nm2 = zeros(T, M, M)
    @inbounds for r in 1:M
        esp1[r] = sumESP_skip1(eta2, N - 1, r)
    end
    @inbounds for r in 1:M
        for s in (r + 1):M
            v1 = sumESP_skip2(eta2, N - 1, r, s)
            v2 = sumESP_skip2(eta2, N - 2, r, s)
            esp2_Nm1[r, s] = v1;  esp2_Nm1[s, r] = v1
            esp2_Nm2[r, s] = v2;  esp2_Nm2[s, r] = v2
        end
    end
    return eta2, ov, esp1, esp2_Nm1, esp2_Nm2
end

# ==============================================================================
# SECTION 3 — Hubbard Hamiltonian overlap
# ==============================================================================

function agp_ham_overlap_hubbard_cached(η            ::AbstractVector{<:Real},
                                        eta2         ::AbstractVector{<:Real},
                                        U_interaction::Real,
                                        E_effective  ::AbstractVector,
                                        C_total      ::AbstractMatrix,
                                        D_total      ::AbstractMatrix{<:Real},
                                        esp1         ::AbstractVector{<:Real},
                                        esp2_Nm1     ::AbstractMatrix{<:Real},
                                        esp2_Nm2     ::AbstractMatrix{<:Real})
    M = length(η)
    H = zero(ComplexF64)
    @inbounds for r in 1:M
        H += E_effective[r] * eta2[r] * esp1[r]
    end
    @inbounds for r in 1:M
        H += U_interaction * C_total[r, r] * eta2[r] * esp1[r]
        for s in 1:M
            r == s && continue
            H += U_interaction * C_total[r, s] * η[r] * η[s] * esp2_Nm1[r, s]
            H += U_interaction * D_total[r, s]  * eta2[r] * eta2[s] * esp2_Nm2[r, s]
        end
    end
    return H
end

# ==============================================================================
# SECTION 4 — T-matrix and unrestricted unitary construction
# ==============================================================================

function build_T_matrix(m::Int, n::Int, boundary_condition::Symbol, t::Real = 1.0)
    @assert m >= 1 && n >= 1
    N  = m * n
    Lx = m;  Ly = n
    T_mat = spzeros(Float64, N, N)
    index_2D(x, y) = x + (y - 1) * Lx
    for y in 1:Ly, x in 1:Lx
        i = index_2D(x, y)
        if x < Lx
            j = index_2D(x + 1, y);  T_mat[i,j] = -t;  T_mat[j,i] = -t
        end
        if y < Ly
            j = index_2D(x, y + 1);  T_mat[i,j] = -t;  T_mat[j,i] = -t
        end
    end
    if boundary_condition == :PBC
        if Lx > 1
            for y in 1:Ly
                i = index_2D(Lx, y);  j = index_2D(1, y);  T_mat[i,j] = -t;  T_mat[j,i] = -t
            end
        end
        if Ly > 1
            for x in 1:Lx
                i = index_2D(x, Ly);  j = index_2D(x, 1);  T_mat[i,j] = -t;  T_mat[j,i] = -t
            end
        end
    end
    return Matrix(T_mat)
end

function matrix_to_params(A::AbstractMatrix)
    N = size(A, 1)
    params = zeros(Float64, N * N)
    idx = 1
    for i in 1:N
        params[idx] = imag(A[i, i]);  idx += 1
    end
    for i in 1:N
        for j in (i + 1):N
            params[idx]     = real(A[i, j])
            params[idx + 1] = imag(A[i, j])
            idx += 2
        end
    end
    return params
end

function construct_unitary_complex(N::Int, params::AbstractVector)
    A   = zeros(ComplexF64, N, N)
    idx = 1
    diag_params = Float64.(params[idx:(idx + N - 1)])
    diag_params .-= mean(diag_params)
    idx += N
    @inbounds for i in 1:N
        A[i, i] = 1im * diag_params[i]
    end
    @inbounds for i in 1:N
        for j in (i + 1):N
            A[i, j] =  params[idx] + 1im * params[idx + 1]
            A[j, i] = -conj(A[i, j])
            idx += 2
        end
    end
    return exp(A)
end

function calculate_unrestricted_quantities(T_matrix   ::AbstractMatrix,
                                           params_up  ::AbstractVector,
                                           params_down::AbstractVector)
    N    = size(T_matrix, 1)
    U_up = construct_unitary_complex(N, params_up)
    U_dn = construct_unitary_complex(N, params_down)
    T_up_rot = U_up * T_matrix * U_up'
    T_dn_rot = U_dn * T_matrix * U_dn'
    E_effective = real(diag(T_up_rot)) .+ real(diag(T_dn_rot))
    C_total = zeros(ComplexF64, N, N)
    D_total = zeros(Float64,    N, N)
    @inbounds for r in 1:N
        for s in 1:N
            sum_C = zero(ComplexF64);  sum_D = 0.0
            for i in 1:N
                u_up_ri = U_up[r, i];  u_dn_ri = U_dn[r, i]
                u_up_si = U_up[s, i];  u_dn_si = U_dn[s, i]
                sum_C += u_up_ri * u_dn_ri * conj(u_up_si * u_dn_si)
                r != s && (sum_D += abs2(u_up_ri) * abs2(u_dn_si))
            end
            C_total[r, s] = sum_C
            D_total[r, s] = sum_D
        end
    end
    return E_effective, C_total, D_total, U_up, U_dn
end

# ==============================================================================
# SECTION 4.5 — UHF Initializer
# ==============================================================================

function solve_uhf(T_mat::AbstractMatrix, U_int::Real, N_pairs::Int, mb::Int, nb::Int; max_iter=2000, tol=1e-8, mix_alpha=0.3)
    N = mb * nb
    
    P_up = zeros(Float64, N, N)
    P_dn = zeros(Float64, N, N)
    
    # 1. Correct 2D Néel (checkerboard) Guess
    for y in 1:nb
        for x in 1:mb
            i = x + (y - 1) * mb
            if (x + y) % 2 == 0
                P_up[i,i] = 1.0
                P_dn[i,i] = 0.0
            else
                P_up[i,i] = 0.0
                P_dn[i,i] = 1.0
            end
        end
    end
    
    U_up = zeros(Float64, N, N)
    U_dn = zeros(Float64, N, N)
    E_uhf = 0.0
    
    # 2. SCF Iteration Loop with Density Mixing
    for iter in 1:max_iter
        F_up = T_mat + U_int * Diagonal(diag(P_dn))
        F_dn = T_mat + U_int * Diagonal(diag(P_up))
        
        eig_up = eigen(Symmetric(F_up))
        eig_dn = eigen(Symmetric(F_dn))
        
        U_up = eig_up.vectors'
        U_dn = eig_dn.vectors'
        
        C_up_occ = U_up[1:N_pairs, :]
        C_dn_occ = U_dn[1:N_pairs, :]
        
        P_up_new = C_up_occ' * C_up_occ
        P_dn_new = C_dn_occ' * C_dn_occ
        
        diff = norm(P_up_new - P_up) + norm(P_dn_new - P_dn)
        
        # Linear mixing to damp oscillations at high U
        P_up = mix_alpha * P_up_new + (1.0 - mix_alpha) * P_up
        P_dn = mix_alpha * P_dn_new + (1.0 - mix_alpha) * P_dn
        
        if diff < tol
            E_uhf = sum(T_mat .* P_up_new) + sum(T_mat .* P_dn_new) + U_int * sum(diag(P_up_new) .* diag(P_dn_new))
            break
        end
    end
    
    # Clean up determinant signs
    det(U_up) < 0 && (U_up[1, :] .*= -1)
    det(U_dn) < 0 && (U_dn[1, :] .*= -1)
    
    return U_up, U_dn, E_uhf
end

# ==============================================================================
# SECTION 5 — Optimiser
# ==============================================================================

function agpmin_hubbard_unrestricted(
        M_sites ::Int,
        N_pairs ::Int,
        mb      ::Int, nb::Int,
        hamp    ::Float64,
        x_init  ::AbstractVector{T};
        niters  ::Int    = 3000,
        bound   ::Symbol = :OBC) where T<:BlasFloat

    N_sites    = mb * nb
    P_eta      = M_sites
    P_A_single = N_sites * N_sites
    @assert M_sites == N_sites

    T_matrix = build_T_matrix(mb, nb, bound, 1.0)

    function f_noimag(xvec::AbstractVector)
        eta_raw = xvec[1:P_eta]
        A_up = @view xvec[(P_eta + 1):(P_eta + P_A_single)]
        A_dn = @view xvec[(P_eta + P_A_single + 1):(P_eta + 2 * P_A_single)]
        nrm = sqrt(sum(abs2, eta_raw))
        nrm < 1e-12 && (nrm = 1e-12)
        eta = eta_raw ./ nrm
        eta2, ov, esp1, esp2_Nm1, esp2_Nm2 = build_agp_cache(eta, N_pairs)
        (!isfinite(real(ov)) || abs(ov) < 1e-14) && return 1000.0
        E_eff, C, D, _, _ = calculate_unrestricted_quantities(T_matrix, A_up, A_dn)
        en  = agp_ham_overlap_hubbard_cached(eta, eta2, hamp, E_eff, C, D,
                                              esp1, esp2_Nm1, esp2_Nm2)
        val = en / ov
        !isfinite(real(val)) && return 1000.0
        return real(val)
    end

    function f_full(xvec::AbstractVector)
        eta_raw = xvec[1:P_eta]
        A_up = @view xvec[(P_eta + 1):(P_eta + P_A_single)]
        A_dn = @view xvec[(P_eta + P_A_single + 1):(P_eta + 2 * P_A_single)]
        nrm = sqrt(sum(abs2, eta_raw))
        nrm < 1e-12 && (nrm = 1e-12)
        eta = eta_raw ./ nrm
        eta2, ov, esp1, esp2_Nm1, esp2_Nm2 = build_agp_cache(eta, N_pairs)
        (!isfinite(real(ov)) || abs(ov) < 1e-14) && return 1000.0
        E_eff, C, D, _, _ = calculate_unrestricted_quantities(T_matrix, A_up, A_dn)
        en  = agp_ham_overlap_hubbard_cached(eta, eta2, hamp, E_eff, C, D,
                                              esp1, esp2_Nm1, esp2_Nm2)
        val = en / ov
        !isfinite(real(val)) && return 1000.0
        return real(val) + 1.0 * abs2(imag(val))
    end

    @printf("----------------\nAGP solve  U = %.4f\n\n", hamp)

    lbfgs_solver        = LBFGS(m = 20, linesearch = Optim.LineSearches.BackTracking())
    lbfgs_solver_polish = LBFGS(m = 20, linesearch = Optim.LineSearches.BackTracking())

    lbfgs_opts(iters)        = Optim.Options(iterations=iters, g_tol=1e-8,
                                              show_trace=false, allow_f_increases=true)
    lbfgs_opts_polish(iters) = Optim.Options(iterations=iters, g_tol=1e-8,
                                              f_reltol=1e-10,
                                              show_trace=false, allow_f_increases=true)

    res  = optimize(f_noimag, x_init, lbfgs_solver, lbfgs_opts(2 * niters))
    best = minimum(res)

    for _ in 1:4
        x_restart = copy(minimizer(res))
        x_restart[(P_eta + 1):end] .+= 5e-3 .* randn(2 * P_A_single)
        r = optimize(f_noimag, x_restart, lbfgs_solver, lbfgs_opts(niters))
        if minimum(r) < best
            best = minimum(r);  res = r
        end
    end

    if Optim.g_residual(res) > 1.0
        for _ in 1:6
            x_restart = copy(x_init) 
            x_restart[(P_eta + 1):end] .+= 0.3 .* randn(2 * P_A_single)
            x_restart[1:P_eta]         .+= 0.1 .* randn(P_eta)
            
            nrm_res = sqrt(sum(abs2, x_restart[1:P_eta]))
            x_restart[1:P_eta] ./= max(nrm_res, 1e-12)
            
            r = optimize(f_noimag, x_restart, lbfgs_solver, lbfgs_opts(niters))
            if minimum(r) < best
                best = minimum(r);  res = r
            end
        end
    end

    if minimum(res) < 500.0
        res_polish = optimize(f_full, minimizer(res), lbfgs_solver_polish, lbfgs_opts_polish(niters))
        if minimum(res_polish) < minimum(res) && Optim.g_residual(res_polish) < 1e6
            res = res_polish
        end
    end

    x_final = copy(minimizer(res))
    nrm_f   = sqrt(sum(abs2, x_final[1:P_eta]));  nrm_f < 1e-12 && (nrm_f = 1.0)
    x_final[1:P_eta] ./= nrm_f

    eta_final = x_final[1:P_eta]
    A_up_f    = @view x_final[(P_eta + 1):(P_eta + P_A_single)]
    A_dn_f    = @view x_final[(P_eta + P_A_single + 1):end]

    eta2, ov, esp1, esp2_Nm1, esp2_Nm2 = build_agp_cache(eta_final, N_pairs)
    E_eff, C, D, U_up, U_dn = calculate_unrestricted_quantities(T_matrix, A_up_f, A_dn_f)
    E_complex = agp_ham_overlap_hubbard_cached(eta_final, eta2, hamp, E_eff, C, D,
                                               esp1, esp2_Nm1, esp2_Nm2) / ov

    display(res)
    @printf("Final energy = %.10f\nImag leakage = %.2e\n\n",
            real(E_complex), abs(imag(E_complex)))

    return real(E_complex), eta_final, x_final, U_up, U_dn
end

# ==============================================================================
# SECTION 6 — Main scan loop
# ==============================================================================

let
    Random.seed!(1234)

    m  = 16;  n  = 8
    mb = 16;  nb = 1
    N_sites    = mb * nb
    P_eta      = N_sites
    P_A_single = N_sites * N_sites
    bc         = :OBC
    niters     = 3000

    U_schedule = reverse(vcat(collect(0.0:-0.05:-1.0), collect(-1.2:-0.2:-2.0), collect(-2.5:-0.5:-11.0)))
    nP = length(U_schedule)
    ebts = zeros(Float64, nP)

    T_mat_init = build_T_matrix(mb, nb, bc, 1.0)
    
    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------
    U_first = Float64(U_schedule[1])
    @printf("Solving initial UHF for U = %.4f...\n", U_first)
    
    U_eigen_up, U_eigen_dn, E_uhf_init = solve_uhf(T_mat_init, U_first, n, mb, nb)
    
    @printf("Converged UHF Energy = %.10f\n\n", E_uhf_init)

    function safe_skew_log(U::AbstractMatrix)
        A = log(ComplexF64.(U))
        return (A .- A') ./ 2
    end

    A_up_guess = matrix_to_params(safe_skew_log(U_eigen_up))
    A_dn_guess = matrix_to_params(safe_skew_log(U_eigen_dn))

    η_guess = zeros(Float64, N_sites)
    η_guess[1:n] .= 0.99
    η_guess[n+1:end] .= 0.01
    nrm0 = sqrt(sum(abs2, η_guess)); η_guess ./= nrm0

    current_minimizer = [η_guess; A_up_guess; A_dn_guess]

    out_dir = joinpath(pwd(), "ResultsHub")
    isdir(out_dir) || mkpath(out_dir)
    f_out    = open(joinpath(out_dir, "E_Unrestricted_AGP_16.txt"), "w")
    eta_file = open(joinpath(out_dir, "eta_values_AGP_16.txt"),     "w")
    U_file   = open(joinpath(out_dir, "U_values_AGP_16.txt"),       "w")

    println(f_out, "Hubbard Unrestricted AGP — reverse scan")
    println(f_out, "m=$m  n=$n  lattice=$(mb)x$(nb)  bc=$bc")
    println(f_out, "U schedule: $U_schedule\n-------------------------")

    for (j, Gval) in enumerate(U_schedule)
        min_energy, eta_final, next_x, U_up, U_dn =
            agpmin_hubbard_unrestricted(m, n, mb, nb, Float64(Gval),
                                        current_minimizer;
                                        niters = niters, bound = bc)
        ebts[j] = min_energy
        @printf("U = %.4f  ->  E = %.10f\n", Gval, min_energy)
        println(f_out,    "U = $Gval   E = $min_energy")
        println(eta_file, "Step $j, U=$Gval:  eta = $eta_final")
        println(U_file,   "Step $j, U=$Gval:  U_up = $U_up")
        println(U_file,   "Step $j, U=$Gval:  U_dn = $U_dn")

        if j < nP
            current_minimizer = copy(next_x)
            nrm_s = sqrt(sum(abs2, current_minimizer[1:P_eta]))
            nrm_s < 1e-12 && (nrm_s = 1.0)
            current_minimizer[1:P_eta] ./= nrm_s
            
            current_minimizer[(P_eta + 1):end] .+= 1e-5 .* randn(2 * P_A_single)
        end
    end

    close(f_out);  close(eta_file);  close(U_file)
    println("Scan complete.")
    nothing
end