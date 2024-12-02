# Skyrmion-Model
This is a general 3+2 model for simulating 3D spin lattice on a 2D plane, essentially represent a single layer of crystal and metal. 

The Hamiltonian of such a system can be expressed as:
![image](https://github.com/user-attachments/assets/fe623254-d660-443a-a49f-7917da2257d5)
- J: Heisenburg exchange interation coefficient
- $D_{ij}$ : DM interation
- B: Magnetic field in z-direction
- g: Landé g-factor
- $\mu_B$: Bohr moment

Here is this code, g $\mu_B$ is treated as 1 for simplicity.

# Method
This model uses Monte Carlo method and Metroplis algorithm for the simulation, and periodic boundary condition is applied. 

We chose to build a 3+2 model rather than 3+3 model, because the model in 3D space depends on the specific physics problem that is working on. Once the extra interation term and the appropriate type of boundry conditions are determined for the problem, the Hamiltonian can be adjusted and the model can be updated to 3 dimensional easily.

# Results
The configuration of the lattice is visualised, and the skyrmions can be observed clearly:
![Figure_2](https://github.com/user-attachments/assets/3fac9dc2-5825-4ed0-89ef-38adcd454a9e)

The energy decreases as the Monte Carlo sweep increases, the value of energy is shown as logarithm in the plot:
![Figure_1](https://github.com/user-attachments/assets/784d11a2-c0c2-43bb-9119-17e009167483)

We can observe different states of lattice as well, by adjusting the hyperparameters: Temperature, DM interation, and Magnetic filed. 

Here is the configuration at Helix state:
![image](https://github.com/user-attachments/assets/ef51ca01-0f69-4626-a1b6-e53679b3a51b)

Here is the configuration at Skyrmion-Helix state:
![image](https://github.com/user-attachments/assets/cec70389-0f29-4f13-8ffa-f9d9a3c889c3)




