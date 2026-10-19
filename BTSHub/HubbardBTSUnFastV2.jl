using FermiPairing
using Combinatorics 
#using BTS 

using LinearAlgebra
using SparseArrays
import LinearAlgebra.BLAS: BlasFloat
using Optim
using Optim: converged, iterations, minimum, minimizer
using Expokit
import Expokit.exp
using Printf
using LineSearches

# --- 1. T Matrix Construction ---
"""
    build_T_matrix(m::Int, n::Int, boundary_condition::Symbol, t::Real=1.0)
"""
function build_T_matrix(m::Int, n::Int, boundary_condition::Symbol, t::Real=1.0)
    @assert m >= 1 && n >= 1 "Lattice dimensions must be positive integers."
    
    N = m * n # Total number of sites
    Lx = m    
    Ly = n    

    T = spzeros(Float64, N, N)

    index_2D(x, y) = x + (y - 1) * Lx
    
    for y in 1:Ly
        for x in 1:Lx
            i = index_2D(x, y)
            
            # X-direction
            if x < Lx
                j_right = index_2D(x + 1, y)
                T[i, j_right] = -t
                T[j_right, i] = -t
            end
            
            # Y-direction 2D
            if y < Ly
                j_down = index_2D(x, y + 1)
                T[i, j_down] = -t
                T[j_down, i] = -t
            end
        end
    end
    
    # Periodic Boundary Conditions (PBC)
    if boundary_condition == :PBC
        # X Wrap-around
        if Lx > 1
            for y in 1:Ly
                i_last_col = index_2D(Lx, y)
                j_first_col = index_2D(1, y)
                T[i_last_col, j_first_col] = -t
                T[j_first_col, i_last_col] = -t
            end
        end
        # Y Wrap-around 2D
        if Ly > 1
            for x in 1:Lx
                i_last_row = index_2D(x, Ly)
                j_first_row = index_2D(x, 1)
                T[i_last_row, j_first_row] = -t
                T[j_first_row, i_last_row] = -t
            end
        end
    end
    
    return Matrix(T)
end

# --- 2. Unrestricted Unitary Construction ---

"""
    construct_unitary_complex(N, params)
    Constructs a Complex Anti-Hermitian matrix A from parameters and returns U = exp(A).
    Params size: N^2
"""
function construct_unitary_complex(N::Int, params::AbstractVector)
    A = zeros(ComplexF64, N, N)
    idx = 1
    
    # 1. Diagonals
    for i in 1:N
        A[i, i] = 0.0 + 1im * params[idx]
        idx += 1
    end
    
    # 2. Off-Diagonals
    for i in 1:N
        for j in (i + 1):N
            real_part = params[idx]
            imag_part = params[idx + 1]
            
            A[i, j] = real_part + 1im * imag_part
            A[j, i] = -conj(A[i, j]) 
            
            idx += 2
        end
    end
    
    return exp(A)
end

# --- 3. Unrestricted Coefficients & Energy ---


function calculate_unrestricted_quantities(T_matrix::AbstractMatrix, params_up::AbstractVector, params_down::AbstractVector)
    N = size(T_matrix, 1)
    
    # 1. Build Two Separate Unitaries
    U_up = construct_unitary_complex(N, params_up)
    U_dn = construct_unitary_complex(N, params_down)
    
    # 2. Rotate Kinetic Energy separately
    T_up_rot = U_up * T_matrix * U_up'
    T_dn_rot = U_dn * T_matrix * U_dn'
    
    E_up_vals = real(diag(T_up_rot))
    E_dn_vals = real(diag(T_dn_rot))

    # Effective Kinetic Energy for the PAIR at index k
    E_effective = E_up_vals .+ E_dn_vals
    
    # 3. Calculate Interaction Coefficients (C and D)
    C_total = zeros(ComplexF64, N, N)
    D_total = zeros(ComplexF64, N, N)

    for r in 1:N 
        for s in 1:N
            sum_C = 0.0 + 0.0im 
            sum_D = 0.0 + 0.0im 
            
            for i in 1:N 
                
                # Retrieve coefficients
                u_up_ri = U_up[r, i]
                u_dn_ri = U_dn[r, i]
                
                u_up_si = U_up[s, i]
                u_dn_si = U_dn[s, i]
                
                # --- C Coefficient (Pair Transfer) ---
                
                pair_overlap_r = u_up_ri * u_dn_ri
                pair_overlap_s = u_up_si * u_dn_si
                
                C_term = pair_overlap_r * conj(pair_overlap_s)
                sum_C += C_term
                
                # --- D Coefficient (Density Density) ---
                
                prob_up_r = abs2(u_up_ri) 
                prob_dn_s = abs2(u_dn_si)
                
                D_term = prob_up_r * prob_dn_s
                
                if r != s
                    sum_D += D_term
                end
            end
            
            C_total[r, s] = sum_C
            D_total[r, s] = sum_D
        end
    end
    
    return E_effective, C_total, D_total, U_up, U_dn
end

# --- 4. Overlap Function ---

"""
    btshamoverlapHubbard_Unrestricted
"""
function btshamoverlapHubbard_Unrestricted(eta::AbstractMatrix{T},
                                   U_interaction::Real,
                                   E_effective::AbstractVector, 
                                   C_total::AbstractMatrix,
                                   D_total::AbstractMatrix) where T<:BlasFloat
    
    N = length(E_effective)
    R = promote_type(T, eltype(C_total)) 
    H_overlap = zero(R)

    # 1. Kinetic Term
    for r in 1:N
        H_overlap += E_effective[r] * 0.5 * FermiPairing.btsz11(eta, r) 
    end

    # 2. Interaction Term
    for r in 1:N
        for s in 1:N
            # Pairing Term
            C_rs = C_total[r, s] 
            H_overlap += U_interaction * C_rs * FermiPairing.btsz02(eta, (r, s,)) 
            
            # Density-Density Term
            if r != s
                D_rs = D_total[r, s]
                H_overlap += U_interaction * (D_rs / 4.0) * FermiPairing.btsz22(eta, (r, s,)) 
            end
        end
    end

    return H_overlap
end

# ==============================================================================
# SECTION 5 — Optimiser 
# ==============================================================================

function btsminhubbard_unrestricted(m::Int, n::Int, mb::Int, nb::Int, hamp::Float64,
                                    x_init::AbstractVector{T};
                                    niters::Int    = 3000,
                                    bound ::Symbol = :OBC) where T<:BlasFloat

    N_sites    = mb * nb
    P_eta      = FermiPairing.btpdim(m, n)
    P_A_single = N_sites * N_sites
    T_matrix   = build_T_matrix(mb, nb, bound, 1.0)

    function f_noimag(xvec::AbstractVector)
        eta_vec  = view(xvec, 1:P_eta)
        A_up_vec = view(xvec, (P_eta + 1):(P_eta + P_A_single))
        A_dn_vec = view(xvec, (P_eta + P_A_single + 1):(P_eta + 2 * P_A_single))

        E_eff, C, D, _, _ = calculate_unrestricted_quantities(T_matrix, A_up_vec, A_dn_vec)
        eta_mat = FermiPairing.btpvectomat(eta_vec, m, n)
        ov = FermiPairing.btsoverlap(eta_mat)
        abs(ov) < 1e-12 && return 1000.0

        en = btshamoverlapHubbard_Unrestricted(eta_mat, hamp, E_eff, C, D)
        val = en / ov
        !isfinite(real(val)) && return 1000.0
        return real(val)          # no imag penalty yet — let L-BFGS find the basin first
    end

    # Polish
    function f_full(xvec::AbstractVector)
        val = f_noimag(xvec)
        val >= 999.0 && return val

        eta_vec  = view(xvec, 1:P_eta)
        A_up_vec = view(xvec, (P_eta + 1):(P_eta + P_A_single))
        A_dn_vec = view(xvec, (P_eta + P_A_single + 1):(P_eta + 2 * P_A_single))
        E_eff, C, D, _, _ = calculate_unrestricted_quantities(T_matrix, A_up_vec, A_dn_vec)
        eta_mat  = FermiPairing.btpvectomat(eta_vec, m, n)
        ov       = FermiPairing.btsoverlap(eta_mat)
        en       = btshamoverlapHubbard_Unrestricted(eta_mat, hamp, E_eff, C, D)
        en_ov    = en / ov
        return real(en_ov) + 1.0 * abs2(imag(en_ov))
    end

    @printf("----------------\nBTS solve  U = %.4f\n\n", hamp)

    lbfgs         = LBFGS(m = 20, linesearch = Optim.LineSearches.BackTracking())
    lbfgs_polish  = LBFGS(m = 20, linesearch = Optim.LineSearches.BackTracking())

    opts(iters)        = Optim.Options(iterations = iters, g_tol = 1e-8,
                                       show_trace = false, allow_f_increases = true)
    opts_polish(iters) = Optim.Options(iterations = iters, g_tol = 1e-8,
                                       f_reltol   = 1e-10,
                                       show_trace = false, allow_f_increases = true)

    res  = optimize(f_noimag, x_init, lbfgs, opts(2 * niters))
    best = Optim.minimum(res)


    for _ in 1:4
        x_r = copy(Optim.minimizer(res))
        x_r[(P_eta + 1):end] .+= 5e-3 .* randn(2 * P_A_single)
        r = optimize(f_noimag, x_r, lbfgs, opts(niters))
        Optim.minimum(r) < best && (best = Optim.minimum(r); res = r)
    end

    # Second restart
    if Optim.g_residual(res) > 1.0
        for _ in 1:6
            x_r = copy(x_init)
            x_r[(P_eta + 1):end] .+= 0.3  .* randn(2 * P_A_single)
            x_r[1:P_eta]         .+= 0.05 .* randn(P_eta)
            r = optimize(f_noimag, x_r, lbfgs, opts(niters))
            Optim.minimum(r) < best && (best = Optim.minimum(r); res = r)
        end
    end

    # Polish 
    if Optim.minimum(res) < 500.0
        res_p = optimize(f_full, Optim.minimizer(res), lbfgs_polish, opts_polish(niters))
        if Optim.minimum(res_p) < Optim.minimum(res) && Optim.g_residual(res_p) < 1e6
            res = res_p
        end
    end

    x_final   = copy(Optim.minimizer(res))
    eta_final = view(x_final, 1:P_eta)
    A_up_f    = view(x_final, (P_eta + 1):(P_eta + P_A_single))
    A_dn_f    = view(x_final, (P_eta + P_A_single + 1):length(x_final))

    eta_mat              = FermiPairing.btpvectomat(eta_final, m, n)
    ov                   = FermiPairing.btsoverlap(eta_mat)
    E_eff, C, D, U_up, U_dn = calculate_unrestricted_quantities(T_matrix, A_up_f, A_dn_f)
    en                   = btshamoverlapHubbard_Unrestricted(eta_mat, hamp, E_eff, C, D)
    E_complex            = en / ov

    display(res)
    @printf("Final energy = %.10f\nImag leakage = %.2e\n\n",
            real(E_complex), abs(imag(E_complex)))

    return real(E_complex), eta_mat, x_final, U_up, U_dn
end 

# ==============================================================================
# SECTION 6 — Start Unrestricted BTS Scan
# ==============================================================================

let
    println("Start Unrestricted BTS Scan")

    # Inputs
    m = 16
    n = 8 
    mb = 4
    nb = 4
    
    N_sites = mb * nb 
    
    P_eta = FermiPairing.btpdim(m, n) 
    P_A_single = N_sites * N_sites
    
    # Total Vector Size: Eta + A_up + A_down
    total_params = P_eta + 2 * P_A_single

    bc = :OBC
    niters = 100000
    hpinit = 0.0 
    dhp = -0.5
    nP = 21

    # Initialize Storage
    ebts = zeros(Float64, nP)
    Gval = copy(hpinit)

    # --- INITIAL GUESS GENERATION ---
    
    # 1. Eta Guess
    eta_guess_agp = ones(Float64, N_sites) 
    for p = 2:2:N_sites; eta_guess_agp[p] *= -1; end 
    bts_eta_guess = FermiPairing.btsfromagp(eta_guess_agp, n)

    # 2. A Parameters Guess (Unrestricted)
    A_up_guess = zeros(Float64, P_A_single) .+ 0.001
    A_dn_guess = zeros(Float64, P_A_single) .- 0.001 
    
    current_minimizer = [bts_eta_guess; A_up_guess; A_dn_guess]
    
    # Open output files
    f_out = open("/ResultsHub/EUnHubBTS16.txt", "w")
    eta_file_out = open("/ResultsHub/EtaUnHubBTS16.txt", "w")
    U_file_out = open("ResultsHub/UmatrixUnHubBTS16.txt", "w")
    
    println(f_out, "Hubbard Unrestricted, m: $m, n: $n")
    println(f_out, "-------------------------")
    println(f_out, "GorD: 10 - 0")
    
    for j = 1:nP
    
        min_energy, btsmat, next_x, U_up, U_dn = 
            btsminhubbard_unrestricted(m, n, mb, nb, Gval, current_minimizer; niters=niters, bound=bc)

        ebts[j] = min_energy

        # --------------------------------------------------------
        # Save eta and U to file for each step
        println(eta_file_out, "Step $j, Gval $Gval:")
        println(U_file_out, "Step $j, Gval $Gval:")
        println("GorD: $Gval")
        #println(f_out, "GorD: $Gval")
        println(f_out, "Energy: $(ebts[j])")
        println(eta_file_out, next_x)
        println(U_file_out, "U_up: $U_up")
        println(U_file_out, "U_down: $U_dn")
        
        flush(f_out)
        flush(eta_file_out)
        flush(U_file_out)

        if j < nP
            Gval += dhp
            current_minimizer = copy(next_x)
            current_minimizer[(P_eta + 1):end] .+= 1e-5 .* randn(2 * P_A_single)
        end
    end

    close(f_out)
    close(eta_file_out)
    close(U_file_out)
    println("Check output files.")
    nothing    
end