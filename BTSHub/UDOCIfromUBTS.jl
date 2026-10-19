using LinearAlgebra
using Printf

# ============================================================
#  UDOCI from BTS Orbitals
# ============================================================

# ─────────────────────────────────────────────────────────────
# Containers
# ─────────────────────────────────────────────────────────────

struct PairBasis
    masks::Vector{UInt64}
    occs ::Vector{Vector{Int}}
    index::Dict{UInt64,Int}
end

# ─────────────────────────────────────────────────────────────
# Lattice builder
# ─────────────────────────────────────────────────────────────

function hubbard_2d_hopping(
    Lx::Int, Ly::Int, t::Real;
    periodic_x::Bool=false, periodic_y::Bool=false,
)
    n  = Lx * Ly
    h  = zeros(Float64, n, n)
    tt = -float(t)
    site(x, y) = (x - 1) * Ly + y
    for x in 1:Lx, y in 1:Ly
        i = site(x, y)
        if x < Lx
            j = site(x+1, y); h[i,j] = tt; h[j,i] = tt
        elseif periodic_x && Lx > 2
            j = site(1, y);   h[i,j] = tt; h[j,i] = tt
        end
        if y < Ly
            j = site(x, y+1); h[i,j] = tt; h[j,i] = tt
        elseif periodic_y && Ly > 2
            j = site(x, 1);   h[i,j] = tt; h[j,i] = tt
        end
    end
    return h
end

function hubbard_1d_hopping(L::Int, t::Real; periodic::Bool=false)
    h  = zeros(Float64, L, L)
    tt = -float(t)
    for i in 1:(L-1); h[i,i+1] = tt; h[i+1,i] = tt; end
    if periodic && L > 2; h[1,L] = tt; h[L,1] = tt; end
    return h
end

# ─────────────────────────────────────────────────────────────
# Pair basis
# ─────────────────────────────────────────────────────────────

@inline isocc(mask::UInt64, p::Int) = ((mask >> (p-1)) & UInt64(1)) == UInt64(1)

function occ_list(mask::UInt64, M::Int)
    occ = Int[]
    for p in 1:M; isocc(mask, p) && push!(occ, p); end
    return occ
end

function build_pair_basis(M::Int, npairs::Int)
    M > 63 && error("Pair basis uses UInt64, require M ≤ 63")
    masks = UInt64[]
    function rec(start, left, mask)
        if left == 0; push!(masks, mask); return; end
        for p in start:(M - left + 1)
            rec(p+1, left-1, mask | (UInt64(1) << (p-1)))
        end
    end
    rec(1, npairs, UInt64(0))
    occs  = [occ_list(m, M) for m in masks]
    index = Dict{UInt64,Int}(m => i for (i,m) in enumerate(masks))
    return PairBasis(masks, occs, index)
end

# ─────────────────────────────────────────────────────────────
# UDOCI tensors
# ─────────────────────────────────────────────────────────────

function hubbard_udoci_tensors(
    h::AbstractMatrix{<:Real}, U::Real,
    Cα::AbstractMatrix{<:Real}, Cβ::AbstractMatrix{<:Real},
)
    hα   = Matrix(Cα' * Matrix(h) * Cα)
    hβ   = Matrix(Cβ' * Matrix(h) * Cβ)
    epsα = Vector(diag(hα))
    epsβ = Vector(diag(hβ))
    A2 = Cα .^ 2;  B2 = Cβ .^ 2;  AB = Cα .* Cβ
    J  = U .* (A2' * B2)
    L  = U .* (AB' * AB)
    K  = copy(L)
    return epsα, epsβ, J, K, L
end

# ─────────────────────────────────────────────────────────────
# Hamiltonian
# ─────────────────────────────────────────────────────────────

function apply_udoci_H!(
    Hv::AbstractVector{Float64},
    v ::AbstractVector{Float64},
    basis  ::PairBasis,
    onepair::Vector{Float64},
    J::Matrix{Float64},
    L::Matrix{Float64},
    M::Int,
)
    nb = length(v)
    fill!(Hv, 0.0)
    @inbounds for I in 1:nb
        mask = basis.masks[I]
        occ  = basis.occs[I]
        np   = length(occ)
        vi   = v[I]

        e = 0.0
        for k in 1:np; e += onepair[occ[k]]; end
        for k1 in 1:np
            p = occ[k1]
            for k2 in (k1+1):np
                q = occ[k2]
                e += J[p,q] + J[q,p]
            end
        end
        Hv[I] += e * vi

        iszero(vi) && continue

        for k in 1:np
            q    = occ[k]
            qbit = UInt64(1) << (q-1)
            rmvd = mask ⊻ qbit
            for p in 1:M
                (p == q || isocc(mask, p)) && continue
                newmask = rmvd | (UInt64(1) << (p-1))
                @inbounds Hv[basis.index[newmask]] += L[p,q] * vi
            end
        end
    end
    return Hv
end

# ─────────────────────────────────────────────────────────────
# Lanczos solver
# ─────────────────────────────────────────────────────────────

function lanczos_ground_state(
    Afun!::Function, n::Int;
    maxiter::Int    = min(n, 300),
    tol    ::Float64 = 1e-12,
    v_init           = nothing,
)
    k = min(maxiter, n)
    Q     = Matrix{Float64}(undef, n, k)
    α_vec = Vector{Float64}(undef, k)
    β_vec = Vector{Float64}(undef, k-1)

    v = v_init === nothing ? randn(Float64, n) : Vector{Float64}(v_init)
    normalize!(v)
    Q[:,1] .= v

    w      = Vector{Float64}(undef, n)
    Eprev  = Inf
    mfinal = k

    @inbounds for j in 1:k
        Afun!(w, @view Q[:,j])
        αj = dot(@view(Q[:,j]), w)
        α_vec[j] = αj
        w .-= αj .* @view(Q[:,j])
        j > 1 && (w .-= β_vec[j-1] .* @view(Q[:,j-1]))
        for l in 1:j
            c = dot(@view(Q[:,l]), w)
            w .-= c .* @view(Q[:,l])
        end
        βj = norm(w)
        if j >= 5
            T_β  = j > 1 ? β_vec[1:j-1] : Float64[]
            Enew = eigmin(SymTridiagonal(α_vec[1:j], T_β))
            if abs(Enew - Eprev) < tol; mfinal = j; break; end
            Eprev = Enew
        end
        if βj < 1e-13 || j == k; mfinal = j; break; end
        β_vec[j]   = βj
        Q[:,j+1] .= w ./ βj
    end

    T_β = mfinal > 1 ? β_vec[1:mfinal-1] : Float64[]
    F   = eigen(SymTridiagonal(α_vec[1:mfinal], T_β))
    idx = argmin(F.values)
    E0  = F.values[idx]
    y   = F.vectors[:,idx]
    v0  = zeros(Float64, n)
    for l in 1:mfinal; @views v0 .+= y[l] .* Q[:,l]; end
    normalize!(v0)
    return E0, v0
end

# ─────────────────────────────────────────────────────────────
# Fixed-orbital UDOCI solve
# ─────────────────────────────────────────────────────────────

function solve_udoci_fixed_orbitals(
    h::AbstractMatrix{<:Real}, U::Real,
    Cα::AbstractMatrix{<:Real}, Cβ::AbstractMatrix{<:Real},
    basis::PairBasis;
    v_init=nothing,
    lanczos_tol::Float64=1e-12,
)
    epsα, epsβ, J, K, L = hubbard_udoci_tensors(h, U, Cα, Cβ)
    M       = length(epsα)
    nb      = length(basis.masks)
    onepair = epsα .+ epsβ .+ diag(L)
    Afun!   = (y, x) -> apply_udoci_H!(y, x, basis, onepair, J, L, M)
    E0, v0  = lanczos_ground_state(Afun!, nb; tol=lanczos_tol, v_init=v_init)
    Hv      = similar(v0)
    Afun!(Hv, v0)
    residual = norm(Hv .- E0 .* v0)
    return E0, v0, epsα, epsβ, J, K, L, residual
end

# ─────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────

function pair_occupations(coeffs::Vector{Float64}, basis::PairBasis, M::Int)
    occ = zeros(Float64, M)
    for I in eachindex(basis.masks)
        w = coeffs[I]^2
        for p in basis.occs[I]; occ[p] += w; end
    end
    return occ
end

function site_densities_from_udoci(
    coeffs::Vector{Float64}, basis::PairBasis,
    Cα::AbstractMatrix{<:Real}, Cβ::AbstractMatrix{<:Real},
)
    M   = size(Cα, 2)
    occ = pair_occupations(coeffs, basis, M)
    nα  = vec(sum((Cα .^ 2) .* permutedims(occ), dims=2))
    nβ  = vec(sum((Cβ .^ 2) .* permutedims(occ), dims=2))
    return nα, nβ
end

function udoci_site_magnetization(
    coeffs::Vector{Float64}, basis::PairBasis,
    Cα::AbstractMatrix{<:Real}, Cβ::AbstractMatrix{<:Real},
)
    nα, nβ = site_densities_from_udoci(coeffs, basis, Cα, Cβ)
    m = nα .- nβ
    return m, maximum(abs.(m))
end

function orbital_spin_distance(Cα::AbstractMatrix{<:Real}, Cβ::AbstractMatrix{<:Real})
    return norm(Matrix(Cα) - Matrix(Cβ))
end

function tokens_to_complex(tokens::Vector{<:AbstractString})
    entries = ComplexF64[]
    i = 1
    while i <= length(tokens)
        tok = tokens[i]

        if i+2 <= length(tokens) && (tokens[i+1] == "+" || tokens[i+1] == "-")
            im_tok = tokens[i+2]
            if endswith(im_tok, "im")
                re  = parse(Float64, tok)
                sgn = tokens[i+1] == "+" ? 1.0 : -1.0
                im  = sgn * parse(Float64, im_tok[1:end-2])
                push!(entries, ComplexF64(re, im))
                i += 3
                continue
            end
        end

        if endswith(tok, "im")
            body = tok[1:end-2]
            split_pos = 0
            for k in length(body):-1:2
                if body[k] == '+' || body[k] == '-'
                    split_pos = k; break
                end
            end
            if split_pos > 0
                re = parse(Float64, body[1:split_pos-1])
                im = parse(Float64, body[split_pos:end])
                push!(entries, ComplexF64(re, im))
            else
                # pure imaginary
                push!(entries, ComplexF64(0.0, parse(Float64, body)))
            end
            i += 1
            continue
        end

        # Pure real token
        push!(entries, ComplexF64(parse(Float64, tok), 0.0))
        i += 1
    end
    return entries
end


function parse_julia_complex_matrix(s::AbstractString, N::Int)
    s = strip(s)

    # Strip type prefix "ComplexF64[" and trailing "]"
    s = replace(s, r"^[A-Za-z0-9{},]+\[" => "")
    s = replace(s, r"\]$" => "")
    s = strip(s)

    row_strings = split(s, r"\s*;\s*")

    entries = ComplexF64[]
    for row_str in row_strings
        toks = filter(!isempty, split(strip(row_str)))
        append!(entries, tokens_to_complex(toks))
    end

    got = length(entries)
    got == N * N || error(
        "Expected $(N*N) entries for $(N)×$(N) matrix, got $got.\n" *
        "Check that N matches your lattice size and that the file is not truncated.")

    return reshape(entries, N, N)'
end

"""
    load_bts_U_matrices
"""
function load_bts_U_matrices(
    filename::AbstractString;
    N::Int,
    transpose_U::Bool = true,
)
    isfile(filename) || error("File not found: $filename")
    lines = readlines(filename)

    results = NamedTuple[]
    i = 1
    while i <= length(lines)
        line = strip(lines[i])

        # Look for a header line
        hdr = match(r"^Step\s+(\d+),\s*Gval\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
        if hdr === nothing
            i += 1; continue
        end

        step    = parse(Int,     hdr[1])
        U_value = parse(Float64, hdr[2])

        # Next non-empty lines should be U_up: ... and U_down: ...
        up_line   = ""
        down_line = ""
        j = i + 1
        while j <= length(lines) && (isempty(up_line) || isempty(down_line))
            l = strip(lines[j])
            if startswith(l, "U_up:")
                up_line = l[length("U_up:")+1:end]
            elseif startswith(l, "U_down:")
                down_line = l[length("U_down:")+1:end]
            end
            j += 1
        end

        if isempty(up_line) || isempty(down_line)
            @warn "Could not find U_up/U_down for step $step (Gval=$U_value), skipping."
            i = j; continue
        end

        U_up = parse_julia_complex_matrix(strip(up_line),   N)
        U_dn = parse_julia_complex_matrix(strip(down_line), N)

        # BTS convention: U[orbital, site]  →  UDOCI C[site, orbital]
        Cα = transpose_U ? real(U_up') : real(U_up)
        Cβ = transpose_U ? real(U_dn') : real(U_dn)

        Cα = Matrix(qr(Cα).Q)
        Cβ = Matrix(qr(Cβ).Q)

        push!(results, (step=step, U_value=U_value, Cα=Cα, Cβ=Cβ))
        i = j
    end

    isempty(results) && error("Parsed 0 valid BTS steps from $filename — check file format.")
    println("Loaded $(length(results)) BTS steps from $(basename(filename))")
    return results
end

# ============================================================
#  Main driver
# ============================================================
function udoci_from_bts_file(
    U_file   ::AbstractString,
    h        ::AbstractMatrix{<:Real},
    npairs   ::Int;
    output_file ::AbstractString = "udoci_from_bts.txt",
    transpose_U ::Bool           = true,
    lanczos_tol ::Float64        = 1e-12,
)
    N = size(h, 1)
    size(h, 1) == size(h, 2) || error("h must be square")

    bts_steps = load_bts_U_matrices(U_file; N=N, transpose_U=transpose_U)

    println("Building pair basis (M=$N, npairs=$npairs)…")
    basis = build_pair_basis(N, npairs)
    println("  Basis dimension = $(length(basis.masks))")

    v0 = nothing   # warm-start Lanczos vector across steps

    open(output_file, "w") do io
        println(io, "# UDOCI energies computed from BTS orbitals (no orbital optimization)")
        println(io, "# Source: $U_file")
        println(io, "# N=$N  npairs=$npairs  lanczos_tol=$lanczos_tol")
        println(io, "# STEP\tU_VALUE\tE_UDOCI\tCI_RESIDUAL\tMAX_ABS_M\t||Cα-Cβ||")

        for st in bts_steps
            step    = st.step
            U_val   = st.U_value
            Cα      = st.Cα
            Cβ      = st.Cβ

            E, coeffs, epsα, epsβ, J, K, L, resid =
                solve_udoci_fixed_orbitals(h, U_val, Cα, Cβ, basis;
                                           v_init=v0, lanczos_tol=lanczos_tol)

            v0 = copy(coeffs)  

            _, max_m    = udoci_site_magnetization(coeffs, basis, Cα, Cβ)
            coeffdiff   = orbital_spin_distance(Cα, Cβ)

            line = "$(step)\t$(U_val)\t$(E)\t$(resid)\t$(max_m)\t$(coeffdiff)"
            println(io, line)
            flush(io)

            @printf("Step %3d | U = %6.3f | E_UDOCI = %14.8f | resid = %.2e | max|m| = %.4f | ||Cα-Cβ|| = %.4f\n",
                    step, U_val, E, resid, max_m, coeffdiff)
        end
    end

    println("\nResults written to: $output_file")
end

# ============================================================
#  Usage 
# ============================================================

Lx     = 16
Ly     = 1
t      = 1.0
npairs = 8          
bc_x   = false     
bc_y   = false

h = hubbard_2d_hopping(Lx, Ly, t; periodic_x=bc_x, periodic_y=bc_y)

U_file = "/ResultsHub/UmatrixUnHubBTS162.txt"

udoci_from_bts_file(
    U_file, h, npairs;
    output_file  = "UDOCI_from_BTS_16.txt",
    transpose_U  = true,
    lanczos_tol  = 1e-12,
)
