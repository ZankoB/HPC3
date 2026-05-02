#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA headers
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <cuda.h>


#include "gifenc.h"
#include "lennard-jones.h"

#define NUM_THREADS 256

// plotting functions
#if GENERATE_GIF
uint8_t palette[] = {
                             0, 0, 0,
                             255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }
    size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = index;
}


void render_frame_gif(ge_GIF *gif, const Particle *particles, unsigned int n, double box_size) {

    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);

    for (unsigned int i = 0; i < n; ++i) {

        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;

        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy) {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx) {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS) {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
                }
            }
        }
    }
}
#endif
double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

__global__ void compute_ke_kernel(const Particle *particles, double *partial_ke, unsigned int n) {
    // Cache za delne vsote kinetične energije v bloku
    extern __shared__ double ke_cache[];

    // Thread idx
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    double ke = 0.0;
    // While je zato, če se zgodi, da je en thread slučajno zadolžen za več partiklov
    while(i < n) {
        const Particle *p = &particles[i];
        // Izračun kinetične energije
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
        // Prestavi se na naslednji particle, ki ga ta thread obdeluje, če je potrebno (Prestavi se za stevilo threadov v celotnem gridu)
        i += blockDim.x * gridDim.x;
    }
    // Zapiše delno vsoto kinetične energije za ta thread v cache
    ke_cache[tid] = ke;

    // Počak da vsi threadi končajo z izračunom ke, predn se začne sum
    __syncthreads();

    // Paralelno seštej vrednost celotnega ke v bloku
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ke_cache[tid] += ke_cache[tid + s];
        }
        // Sync pred naslednim korakom
        __syncthreads();
    }

    // Zapiši delno vsoto kinetične energije za ta blok
    if (tid == 0) {
        partial_ke[blockIdx.x] = ke_cache[0];
    }
}

__global__ void sum_array_kernel(const double* input, double* output, unsigned int n) {
    // Cache za delne vsote v bloku
    extern __shared__ double sum_cache[];

    // Thread idx
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    double sum = 0.0;
    // While je zato, če se zgodi, da je en thread slučajno zadolžen za več subsumov
    while (i < n) {
        sum += input[i];
        i += gridDim.x * blockDim.x;
    }
    sum_cache[tid] = sum;

    // Počak da vsi threadi končajo z izračunom delnih vsot, predn se začne sum
    __syncthreads();

    // Paralelno seštej delne vsote v manjše delne vsote
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_cache[tid] += sum_cache[tid + s];
        }
        __syncthreads();
    }

    // Zapiši končni rezultat celotnega sistema
    if (tid == 0) {
        output[blockIdx.x] = sum_cache[0];
    }
}

double compute_keGPU(const Particle *particles, unsigned int n) {
    size_t shared_mem_size = NUM_THREADS * sizeof(double);

    int num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    double *d_partial_ke;
    cudaMalloc(&d_partial_ke, num_blocks * sizeof(double));

    compute_ke_kernel<<<num_blocks, NUM_THREADS, shared_mem_size>>>(particles, d_partial_ke, n);

    // Zdaj imamo delne vsote v d_partial_ke, potrebujemo še en kernel za seštevanje teh delnih vsot
    int num_blocks_sum = (num_blocks + NUM_THREADS - 1) / NUM_THREADS;
    
    while(num_blocks_sum > 1) {
        int blocks = (num_blocks_sum + NUM_THREADS - 1) / NUM_THREADS;
        double *d_output;
        cudaMalloc(&d_output, blocks * sizeof(double));

        sum_array_kernel<<<blocks, NUM_THREADS, shared_mem_size>>>(d_partial_ke, d_output, num_blocks_sum);
        checkCudaErrors(cudaGetLastError());

        cudaFree(d_partial_ke);
        d_partial_ke = d_output;
        num_blocks_sum = blocks;
    }

    double ke;
    cudaMemcpy(&ke, d_partial_ke, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_partial_ke);

    return ke;
}

// compute kinetic energy of the system
double compute_keCPU(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    #pragma omp parallel for reduction(+:ke)
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    // place particles int he middle of the grid with some random jitter and assign random velocities
    #pragma omp parallel for reduction(+:mean_vx, mean_vy)
    for (unsigned int k = 0; k < n; k++) {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        
        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    #pragma omp parallel for reduction(+:ke)
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (
            particles[k].vx * particles[k].vx +
            particles[k].vy * particles[k].vy
        );
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    #pragma omp parallel for
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

// apply periodic boundary conditions to ensure particles stay within the simulation box
void wrap_positionsCPU(Particle *particles, unsigned int n, double box_size) {
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        p->x = wx;
        p->y = wy;
    }
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
__host__ __device__ double compute_v_shift(void) {
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}

__global__ void compute_forces_kernel(Particle *particles, double *potential_energy, unsigned int n, double box_size) {
    // Thread idx
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // While je zato, če se zgodi, da je en thread slučajno zadolžen za več partiklov
    while(i < n) {
        // Pozicija partikla i
        double pi_x = particles[i].x;
        double pi_y = particles[i].y;

        // Inicializiraj sile in potencialno energijo za partikla i
        double fx = 0.0;
        double fy = 0.0;
        double pe = 0.0;

        double v_shift = compute_v_shift();

        for (unsigned int j = 0; j < n; j++) {
            // particle ne vpliva sama nase
            if(j == i) {
                continue;
            }

            // Pozicija partikla j
            double pj_x = particles[j].x;
            double pj_y = particles[j].y;

            // Izračun razdalje med partiklom i in j ob upoštevanju boxa
            double dx = pi_x - pj_x;
            double dy = pi_y - pj_y;
            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            double r = sqrt(dx * dx + dy * dy);

            // Izračunaj sile in potencialne energije, če sta partikla znotraj cutoff razdalje
            if (r < R_CUT && r > 0.0) {
                // LJ sile
                double sr = SIGMA / r;
                double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
                fx += fij * dx / r;
                fy += fij * dy / r;

                // LJ potencialna energija
                double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
                pe += 0.5 * vij;
            }
        }
        // Posodobi sile in potencialno energijo za partikla i
        particles[i].fx = fx;
        particles[i].fy = fy;
        potential_energy[i] = pe;

        // Prestavi se na naslednji particle, ki ga ta thread obdeluje, če je potrebno
        i += gridDim.x * blockDim.x;
    }
}

double compute_forcesGPU(Particle *particles, unsigned int n, double box_size) {
    int num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    size_t shared_mem_size = NUM_THREADS * sizeof(double);

    double *d_potential_energy;
    cudaMalloc(&d_potential_energy, n * sizeof(double));

    compute_forces_kernel<<<num_blocks, NUM_THREADS, shared_mem_size>>>(particles, d_potential_energy, n, box_size);
    checkCudaErrors(cudaGetLastError());

    int num_blocks_sum = (n + NUM_THREADS - 1) / NUM_THREADS;
    while(num_blocks_sum > 1) {
        int blocks = (num_blocks_sum + NUM_THREADS - 1) / NUM_THREADS;
        double *d_output;
        cudaMalloc(&d_output, blocks * sizeof(double));

        sum_array_kernel<<<blocks, NUM_THREADS, shared_mem_size>>>(d_potential_energy, d_output, num_blocks_sum);
        checkCudaErrors(cudaGetLastError());

        cudaFree(d_potential_energy);
        d_potential_energy = d_output;
        num_blocks_sum = blocks;
    }

    double pe;
    cudaMemcpy(&pe, d_potential_energy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_potential_energy);

    return pe;
}

double compute_forcesCPU(Particle *particles, unsigned int n, double box_size) {

    #pragma omp parallel for
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    double pe = 0.0;
    double v_shift = compute_v_shift();
    #pragma omp parallel for reduction(+:pe)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            Particle *pi = &particles[i];
            Particle *pj = &particles[j];
            
            // compute distance between particles with periodic boundary conditions
            double dx = pi->x - pj->x;
            double dy = pi->y - pj->y;

            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
            double r = sqrt(dx * dx + dy * dy);
            if (r >= R_CUT || r == 0.0) {
                continue;
            }
            double sr = SIGMA / r;

            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            double fx = fij * dx / r;
            double fy = fij * dy / r;

            pi->fx += fx;
            pi->fy += fy;

            double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
            pe += 0.5 * vij;
        }
    }

    return pe;
}

__global__ void leapfrog_step1_kernel(Particle *particles, unsigned int n, double box_size) {
    // Thread idx
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // While je zato, če se zgodi, da je en thread slučajno zadolžen za več partiklov
    while(i < n) {
        Particle *p = &particles[i];
        // Updatei hitrosti za polovico časovnega koraka
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;

        // Updatei pozicije za cel časovni korak
        p->x += DT * p->vx;
        p->y += DT * p->vy;

        // Periodično zavijanje pozicij nazaj v box, da ostanejo znotraj meja
        // to je ubistvu wrap_positionsGPU, sam je tuki da ni treba dodatnega kernelja
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);
        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        // Posodobi pozicije
        p->x = wx;
        p->y = wy;

        i += gridDim.x * blockDim.x;
    }
}

__global__ void leapfrog_step2_kernel(Particle *particles, unsigned int n) {
    // Thread idx
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // While je zato, če se zgodi, da je en thread slučajno zadolžen za več partiklov
    while (i < n) {
        Particle *p = &particles[i];
        // Updatei hitrosti za drugo polovico časovnega koraka z novimi silami
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        
        i += gridDim.x * blockDim.x;
    }
}

double leapfrog_stepGPU(Particle *d_particles, unsigned int n, double box_size) {
    int num_threads = NUM_THREADS;
    int num_blocks = (n + num_threads - 1) / num_threads;

    // Nared prvi del za polovico koraka
    leapfrog_step1_kernel<<<num_blocks, num_threads>>>(d_particles, n, box_size);
    checkCudaErrors(cudaGetLastError());

    // Izračunaj sile in potencialno energijo z novimi pozicijami
    double pe = compute_forcesGPU(d_particles, n, box_size);

    // Nared drugi del za drugo polovico koraka
    leapfrog_step2_kernel<<<num_blocks, num_threads>>>(d_particles, n);
    checkCudaErrors(cudaGetLastError());

    return pe;
}

double leapfrog_stepCPU(Particle *particles, unsigned int n, double box_size) {
    // update velocities by half a time step, then update positions by a full time step, 
    //and finally update velocities by another half time step to complete the leapfrog integration step
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;

        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }

    wrap_positionsCPU(particles, n, box_size);

    double pe = compute_forcesCPU(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    return pe;
}

SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
  
    Particle* d_particles = NULL; // This will be our device pointer
    Particle* h_particles_pinned = NULL; // This will be our pinned host pointer

    // Allocate pinned host memory and get a device pointer to it.
    checkCudaErrors(cudaHostAlloc(&h_particles_pinned, n * sizeof(Particle), cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer(&d_particles, h_particles_pinned, 0));

    // Copy the initial data into the new pinned memory block
    memcpy(h_particles_pinned, particles, n * sizeof(Particle));

    SimulationResult out;
    out.start_potential= compute_forcesGPU(d_particles, n, box_size);
    out.start_kinetic = compute_keGPU(d_particles, n);
    out.start_total = out.start_kinetic + out.start_potential;

    
#if GENERATE_GIF
    ge_GIF *gif = NULL;
    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        // Render the first frame using the pinned host pointer
        render_frame_gif(gif, h_particles_pinned, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_stepGPU(d_particles, n, box_size);
        out.final_kinetic = compute_keGPU(d_particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps) {
            printf(
                "step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                step,
                out.final_kinetic,
                out.final_potential,
                out.final_total
            );
        }
    
#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            // We need to ensure the GPU is done computing before the CPU reads the memory.
            checkCudaErrors(cudaDeviceSynchronize()); 
            // No cudaMemcpy! Just render directly from the pinned host pointer.
            render_frame_gif(gif, h_particles_pinned, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    // Ensure all GPU work is done before we copy the final results and clean up.
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy the final results from pinned memory back to the original array.
    memcpy(particles, h_particles_pinned, n * sizeof(Particle));

    // Free the pinned host memory. This also invalidates the device pointer.
    checkCudaErrors(cudaFreeHost(h_particles_pinned));

    out.n = n;
    out.particles = particles; // Return the original, now updated, host array
    return out;
}