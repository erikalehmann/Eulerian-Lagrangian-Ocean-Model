# --- Biological parameters ---
#coral = 0.05–0.15 per day
#urchin = 0.02–0.08 per day
m_min_R = 0.003
c_R = 0.05
T_opt_R = 25
T_br_R = 2.5

#coral = 0.001–0.01 day⁻¹
#urchin = 0.001–0.005 per day
m_min_A = 0.0003
c_A = 0.005
T_opt_A = 25
T_br_A = 2.5

# --- Growth & Settlement parameters ---
D_A = 0.001
gamma = 0.003 #growth rate 0.01–0.05 day⁻¹
f = 1 #number of particles per adult

a_set = 100 #max recruits per particle
b_set = 2.0 #lagran recruit decay rate
larvae_threshold = 0  # particles will only spawn if there are this many adults in a cell
min_settle_steps = 40  # minimum particle age (in Lagrangian steps/days) to allow settlement

eta_A = 0.01 #scale of dense mortality
p_K = 1 #power of dens mortality
K_A = 100000 #coral = 500–5000, urchin = 100–2000

# --- Eulerian grid ---
N = 50
domain_km = 2000
cell_width_km = domain_km/N
shelf_width_km =20

input_cell = [10,11,12,13,14,15,16,17,18,19]

# --- Temperature forcing ---
t_tempmax = 45
a_T = 2.5
b_T = 20
Delta_cc_T = 0.00
Delta_y_T = 7

refugia = None #None or [cells]
refugia_warming = 1.0 # 0=no warming, 1=same climate change warming
refugia_delta = -1.0

# --- Lagrangian parameters ---
nx = 100
ny = domain_km
dx = 1.0
dy = 1.0
dt_L = 1.0
steps = 100
sigma = 1.8
bias_strength=1.5
v_mod = -1

# --- Simulation time ---
years = 100
dt = 1
t_max = 365 * years

particles_to_plot=30
plot_every_years=10