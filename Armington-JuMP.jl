# Toy model to test syntax required for JuMP. This example is based on an Armington model. Ultimately, we want use the JuMP package to solve the Eaton-Kortum and Caliendo et al. models.
using JuMP, Ipopt
# Define parameters:
N = 2  # number of countries
V = 1000  # number of varieties
w = ones(N)  # wages in each country
L = ones(N)  # labor endowment in each country
A = ones(N)  # aggregate productivity in each country
z = exp.(randn(N, V)) # idiosyncratic productivities for each variety
sigma = 5.0  # elasticity of substitution between varieties
tau = ones(N, N)  # trade costs between countries; tau[i,j] denotes iceberg trade cost from i to j

# Create a JuMP model
model = Model(Ipopt.Optimizer)
# set_silent(model)
@variable(model, w[1:N] >= 0)

# Optimal price function (c.i.f. prices):
function price_function(w, tau, A, z, sigma)
    return (w .* tau) ./ A .* z .* sigma / (sigma - 1)
end
# Register the function with JuMP:
@NLfunction(model, price_func(w, tau, A, z, sigma), price_function(w, tau, A, z, sigma))

# Price index for each country:
function price_index_function(w, tau, A, z, sigma)
    P = zeros(N)
    for j in 1:N # indexes prices by destination (of final use)
        sum_term = 0.0
        for i in 1:N # indexes prices by origin
            for v in 1:V
                p_ijv = price_function(w[i], tau[i, j], A[i], z[i, v], sigma)
                sum_term += p_ijv^(1 - sigma)
            end
        end
        P[j] = sum_term^(1 / (1 - sigma))
    end
    return P
end
# Register the function with JuMP:
@NLfunction(model, price_index_func(w, tau, A, sigma), price_index_function(w, tau, A, sigma))

# Demand function:
function demand_function(a, w, L, A, tau, sigma)
    q = a .* price_function(w, tau, A, sigma).^(-sigma) .* price_index_function(w, tau, A, sigma).^(sigma - 1) .* w .* L
    return q
end
# Register the function with JuMP:
@NLfunction(model, demand_func(a, w, L, A, tau, sigma), demand_function(a, w, L, A, tau, sigma))

# Trade balance condition:
# function trade_balance_function(w, L, A, tau, sigma)
#     TB = zeros(N)
#     for j in 1:N-1
    
#     for i in 1:N
#         p_ij = price_function(w[i], tau[i, j], A[i], sigma)
#         q_ij = demand_function(a[i,j],w[j], L[j], A, tau, sigma)
#         exports[j] += p_ij 