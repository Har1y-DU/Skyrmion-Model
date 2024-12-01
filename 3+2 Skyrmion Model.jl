using Random, PyCall, LinearAlgebra, StaticArrays, Printf, HDF5
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.mplot3d as mplot3d

# Parameters
Nx = 50  # Lattice size in x-direction
Ny = 50  # Lattice size in y-direction
eqSteps = 100000  # Number of equilibration steps
T = [0.3]  # List of temperatures to simulate
nt = length(T)   # Number of temperature points
gamma = 0.005  # Step size for spin perturbation
J = 1.0  # Heisenberg interaction strength
D = 0.4
B = 0.1  # Magnetic field along z-axis

function initial_state(Nx, Ny; seed=100)
# Initialize spin lattice with random 3D unit vectors on a 2D surface
    Random.seed!(seed)
    lattice = Array{MVector{3, Float64}, 2}(undef, Nx, Ny)
    for i in 1:Nx
        for j in 1:Ny
            θ = 2 * pi * rand()
            φ = acos(2 * rand() - 1)
            lattice[i, j] = [sin(φ) * cos(θ), sin(φ) * sin(θ), cos(φ)]  # Random 3D unit vector
        end
    end
    return lattice
end

# Define neighbors (periodic boundary conditions in 2D)
function get_neighbors(i, j, Nx, Ny, lattice)
    nb_right = lattice[mod1(i+1, Nx), j]
    nb_left  = lattice[mod1(i-1, Nx), j]
    nb_up    = lattice[i, mod1(j+1, Ny)]
    nb_down  = lattice[i, mod1(j-1, Ny)]
    return nb_right, nb_left, nb_up, nb_down
end

# Heisenberg exchange and Dzyaloshinskii-Moriya interaction calculation without dot and cross
function E_heisenburg(S, neighbors)
    nb_left, nb_right, nb_up, nb_down = neighbors

    # Heisenberg exchange interaction (S · neighbor) expanded component-wise
    Energy = -J * (
        S[1] * nb_right[1] + S[2] * nb_right[2] + S[3] * nb_right[3] +
        S[1] * nb_left[1] + S[2] * nb_left[2] + S[3] * nb_left[3] +
        S[1] * nb_up[1] + S[2] * nb_up[2] + S[3] * nb_up[3] +
        S[1] * nb_down[1] + S[2] * nb_down[2] + S[3] * nb_down[3]
    ) + D * (
        (S[2] * nb_right[3] - S[3] * nb_right[2]) - 
        (S[2] * nb_left[3] - S[3] * nb_left[2]) + 
        (S[3] * nb_up[1] - S[1] * nb_up[3]) - 
        (S[3] * nb_down[1] - S[1] * nb_down[3]) 
    ) - B * S[3]

    return Energy
end

# Energy calculation using the 3+2 Hamiltonian: Heisenberg, DM, and magnetic field terms
function calc_energy(lattice, Nx, Ny)
    E = 0.0
    for i in 1:Nx
        for j in 1:Ny
            S = lattice[i, j]
            neighbors = get_neighbors(i, j, Nx, Ny, lattice)
            E += E_heisenburg(S, neighbors)
        end
    end
    return E * 0.5
end

# Magnetization calculation in x, y, z directions
function calc_magnetization(lattice, Nx, Ny)
    Mx, My, Mz = 0.0, 0.0, 0.0
    for i in 1:Nx, j in 1:Ny
        Mx += lattice[i, j][1]
        My += lattice[i, j][2]
        Mz += lattice[i, j][3]
    end
    return Mx , My , Mz 
end

function mcmove(lattice, beta, Nx, Ny)
    for _ in 1:(Nx * Ny)
        i = rand(1:Nx)
        j = rand(1:Ny)
        # Store the old spin configuration
        lattice_old = lattice[i, j]

        # Perturb the spin to create the new spin configuration
        nb_right = lattice[mod1(i+1, Nx), j]
        nb_left  = lattice[mod1(i-1, Nx), j]
        nb_up    = lattice[i, mod1(j+1, Ny)]
        nb_down  = lattice[i, mod1(j-1, Ny)]

        dx = gamma * (-J * (nb_right[1] + nb_left[1] + nb_up[1] + nb_down[1]) + D * (nb_down[3] - nb_up[3]))
        dy = gamma * (-J * (nb_right[2] + nb_left[2] + nb_up[2] + nb_down[2]) + D * (nb_right[3] - nb_left[3]))
        dz = gamma * (-J * (nb_right[3] + nb_left[3] + nb_up[3] + nb_down[3]) + D * (nb_left[2] - nb_right[2] + nb_up[1] - nb_down[1]) - B)
        
        # New spin configuration
        lattice_new = lattice_old - [dx, dy, dz]

        norm = sqrt(lattice_new[1]^2 + lattice_new[2]^2 + lattice_new[3]^2)
        lattice_new[1] /= norm
        lattice_new[2] /= norm
        lattice_new[3] /= norm

        # Calculate energy for the old configuration
        E_old = -J * (
            lattice_old[1] * (nb_right[1] + nb_left[1] + nb_up[1] + nb_down[1]) +
            lattice_old[2] * (nb_right[2] + nb_left[2] + nb_up[2] + nb_down[2]) +
            lattice_old[3] * (nb_right[3] + nb_left[3] + nb_up[3] + nb_down[3])
        ) + D * (
            (lattice_old[2] * nb_right[3] - lattice_old[3] * nb_right[2]) -
            (lattice_old[2] * nb_left[3] - lattice_old[3] * nb_left[2]) +
            (lattice_old[3] * nb_up[1] - lattice_old[1] * nb_up[3]) -
            (lattice_old[3] * nb_down[1] - lattice_old[1] * nb_down[3])
        ) - B * lattice_old[3]

        # Calculate energy for the new configuration
        E_new = -J * (
            lattice_new[1] * (nb_right[1] + nb_left[1] + nb_up[1] + nb_down[1]) +
            lattice_new[2] * (nb_right[2] + nb_left[2] + nb_up[2] + nb_down[2]) +
            lattice_new[3] * (nb_right[3] + nb_left[3] + nb_up[3] + nb_down[3])
        ) + D * (
            (lattice_new[2] * nb_right[3] - lattice_new[3] * nb_right[2]) -
            (lattice_new[2] * nb_left[3] - lattice_new[3] * nb_left[2]) +
            (lattice_new[3] * nb_up[1] - lattice_new[1] * nb_up[3]) -
            (lattice_new[3] * nb_down[1] - lattice_new[1] * nb_down[3])
        ) - B * lattice_new[3]

        # Calculate energy difference
        dE = E_new - E_old

        # Metropolis criterion
        if dE < 0 
            lattice[i, j] = lattice_new  # Accept move
        elseif rand() < exp(-dE * beta)
            lattice[i, j] = lattice_new  # Accept move
        end
    end
    return lattice
end

function topological(nx_p::Array, ny_p::Array, nz_p::Array, Nx, Ny)
    topo = 0.0
    for i in 1:Nx
        for j in 1:Ny
            # Periodic boundary conditions
            iu = mod1(i+1, Nx); id = mod1(i-1, Nx)
            ju = mod1(j+1, Ny); jd = mod1(j-1, Ny)
            
            # Current spin components
            nx = nx_p[i, j]
            ny = ny_p[i, j]
            nz = nz_p[i, j]
            
            # Neighboring spin components
            nx_iu = nx_p[iu, j]
            ny_iu = ny_p[iu, j]
            nz_iu = nz_p[iu, j]
            
            nx_id = nx_p[id, j]
            ny_id = ny_p[id, j]
            nz_id = nz_p[id, j]
            
            nx_ju = nx_p[i, ju]
            ny_ju = ny_p[i, ju]
            nz_ju = nz_p[i, ju]
            
            nx_jd = nx_p[i, jd]
            ny_jd = ny_p[i, jd]
            nz_jd = nz_p[i, jd]
            
            # partial derivatives
            d_xx = (nx_iu - nx_id)
            d_yx = (ny_iu - ny_id)
            d_xy = (nx_ju - nx_jd)
            d_yy = (ny_ju - ny_jd)
            d_zx = (nz_iu - nz_id)
            d_zy = (nz_ju - nz_jd)

            # Manually calculate cross products for neighboring spins
            topo += nx * (d_yx*d_zy - d_yy*d_zx) + ny * (d_xy*d_zx - d_xx*d_zy) * nz * (d_xx*d_yy - d_xy*d_yx)
        end
    end
    
    # Normalize by the total number of spins and a factor of 4π
    return topo / (4 * π)
end

# Updated function to plot 2D slices using the quiver plot
function plotxy(nx_m, ny_m, nz_m, Nx, Ny)
    X, Y = range(1, stop=Nx, length=Nx), range(1, stop=Ny, length=Ny)
    
    U = transpose(nx_m)
    V = transpose(ny_m)
    C = transpose(nz_m)

    plt.figure(figsize=(10, 10), dpi=100)
    
    # Create quiver plot
    quiver_plot = plt.quiver(X, Y, U, V, C, angles="xy", scale_units="xy", scale=1, width=0.003, pivot="mid")
    
    # Add color bar with labeled values at -1, 0.25, 0.50, 0.75, and +1
    cbar = plt.colorbar(quiver_plot)
    cbar.set_label("Z-Component")
    # Set the ticks at -1, 0.25, 0.50, 0.75, and +1
    cbar.set_ticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    # Set the labels for each tick
    cbar.ax.set_yticklabels(["-1.0", "-0.75", "-0.5", "-0.25", "0.0", "0.25", "0.5", "0.75", "1.0"])
    
    plt.gca().set_aspect("equal", adjustable="box")

    # Label the axes and set ticks at 10, 20, 30, etc.
    plt.xlim(0, Nx+1)
    plt.ylim(0, Ny+1)
    plt.xticks(1:9:Nx, fontsize=25)  # Label positions 1, 10, 20, etc.
    plt.yticks(1:9:Ny, fontsize=25)
    plt.xlabel("X Position", fontsize=25)
    plt.ylabel("Y Position", fontsize=25)

    # Title and display the plot
    plt.title("XY Plane", fontsize=25)
    plt.show()
end

# Function to plot 2D slices using the quiver plot (from your previous code)
function visualize_slices(nx_m, ny_m, nz_m, Nx, Ny, T_val)
    # Visualization for XY plane
    plotxy(nx_m, ny_m, nz_m, Nx, Ny)
    println("Final configuration visualized for T = $T_val.")
end

# Modified function to extract spin components and call visualization
function visualize_lattice(lattice, T_val, Nx, Ny)
    nx_m = Array{Float64}(undef, Nx, Ny)  # x-components of spins
    ny_m = Array{Float64}(undef, Nx, Ny)  # y-components of spins
    nz_m = Array{Float64}(undef, Nx, Ny)  # z-components of spins

    for i in 1:Nx, j in 1:Ny
        nx_m[i, j] = lattice[i, j][1]
        ny_m[i, j] = lattice[i, j][2]
        nz_m[i, j] = lattice[i, j][3]
    end

    # Call the slice visualization method
    visualize_slices(nx_m, ny_m, nz_m, Nx, Ny, T_val)
end

function save_data_hdf5(nx_p, ny_p, nz_p, Nx, Ny, T_val, filename)
    h5open(filename, "w") do file
        file["Nx"] = Nx
        file["Ny"] = Ny
        file["T_val"] = T_val
        file["nx_p"] = nx_p
        file["ny_p"] = ny_p
        file["nz_p"] = nz_p
    end
end

# Main simulation loop combined with the model simulation
function main()
    for ti in 1:nt
        lattice = initial_state(Nx, Ny)
        T_val = T[ti]
        println("Simulating for T = $T_val")
        
        beta = 1.0 / T_val
        E_vals = []
        Mx_vals, My_vals, Mz_vals = [], [], []
        Q_vals = []
        sweep_counts = Int[]

        nx_p = Array{Float64}(undef, Nx, Ny)
        ny_p = Array{Float64}(undef, Nx, Ny)
        nz_p = Array{Float64}(undef, Nx, Ny)

        # Equilibration phase
        for i in 1:eqSteps
            lattice = mcmove(lattice, beta, Nx, Ny)
            energy = calc_energy(lattice, Nx, Ny)
            Mx, My, Mz = calc_magnetization(lattice, Nx, Ny)

            # Extract spin components for topological charge calculation
            nx_p = [lattice[i, j][1] for i in 1:Nx, j in 1:Ny]
            ny_p = [lattice[i, j][2] for i in 1:Nx, j in 1:Ny]
            nz_p = [lattice[i, j][3] for i in 1:Nx, j in 1:Ny]
            Q = topological(nx_p, ny_p, nz_p, Nx, Ny)

            # Store the results
            push!(E_vals, energy)
            push!(Mx_vals, Mx)
            push!(My_vals, My)
            push!(Mz_vals, Mz)
            push!(Q_vals, Q)
            push!(sweep_counts, i)
        end

        # Plot energy vs number of sweeps with log scale on the y-axis
        function negative_scientific_formatter(x, _)
            if x != 0
                exponent = Int(log10(x))
                return Printf.@sprintf("-10^%d", exponent)
            else
                return "0"
            end
        end

        plt.figure(figsize=(8, 6))
        plt.plot(sweep_counts, abs.(E_vals), marker="o", label="Energy", markersize=2, color="blue")
        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(negative_scientific_formatter))

        plt.xlabel("Number of Sweeps")
        plt.ylabel("Energy (log scale)")
        plt.title("Energy vs Sweeps (Equilibration) at T = $T_val")
        plt.legend()

        # Plot Mx, My, Mz vs number of sweeps
        plt.figure(figsize=(10, 6))
        plt.plot(sweep_counts, Mx_vals, label="Mx", color="r")
        plt.plot(sweep_counts, My_vals, label="My", color="g")
        plt.plot(sweep_counts, Mz_vals, label="Mz", color="b")
        plt.xlabel("Number of Sweeps")
        plt.ylabel("Magnetization (M)")
        plt.title("Magnetization Components vs Sweeps at T = $T_val")
        plt.legend()

        # Plot topological charge vs number of sweeps
        plt.figure(figsize=(8, 6))
        plt.scatter(sweep_counts, Q_vals, marker="o", label="Q vs Sweeps", s=2)
        plt.title("Topological Charge (Q) vs Number of Sweeps")
        plt.xlabel("Number of Sweeps")
        plt.ylabel("Topological Charge (Q)")
        plt.legend()
        plt.show()

        # Visualize final 3D spin configuration after equilibration
        visualize_lattice(lattice, T_val, Nx, Ny)
    end
end

main()

