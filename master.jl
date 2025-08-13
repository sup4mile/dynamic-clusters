# 05/08/2025 
# Indices updated, not tested

using Random, Distributions, Optim, NLsolve, SpecialFunctions

# Exogenous parameters:

N = 2 # Number of countries
periods = 5
J = 3 # Number of goods (needs to be big number and an integer type)
θ = fill(4.0, J) # Frechet shape parameter (governs comparative advantage)

v = 1.0 # migration elasticity
β = .99 # discount factor
α = ones(J) * (1/J) # final good expenditure share (currently set to equal shares)

κ = ones(N, N, J) # iceberg trade costs (currently set to 1, i.e., no trade costs)

# Initial conditions (state variables):

# Labor force (endogenous):
Lt = ones(N, J, periods) # size of labor force in each country at time
Ldot = ones(N, J, periods) # growth rate of labor force in each country between 'periods-1' and 'periods'

# Productivities (exogenous):
At0 = ones(N, J) # initial productivities
Adot = ones(N, J, periods)
# Trade costs (exogenous):
kdot = ones(N, N, J, periods)

# Initial guesses for control variables:

# Wages:
wt = ones(N, J, periods)
wt0 = ones(N, J)
wdot = ones(N, J, periods)
# Trade shares:
tradesharest0 = ones(N, N, J) * (1 / N) # symmetric for the time being

pdotArray = ones(N, J, periods)
d1wdot = ones(N, J)

μtminus1 = zeros(N, N, J, J) #value of μ at time t = -1
πt0 = ones(N, N, J) #initial trade shares
μt = ones(N, N, J, J, periods)*(1/(N*J))

# a guess for path of udot:
global udotPathGuess = ones(N, J, periods)
udotPathUpdate = zeros(N, J, periods)
errormax = 1.0 #the maximum log difference between guesses and updates for appendix D algorithm


function pdot(n, j, d1wdot, kdot, Adot, time, tradesharest0) #pdot(nj) from equation (12)
    (sum(tradesharest0[n, i, j] * (d1wdot[i, j] * kdot[n, i, j, time])^-θ[j] * Adot[i, j, time]^θ[j] for i in 1:N))^(-1 / θ[j])
end 

function tradeSharest0(n, i, j, wt0, At0, κ) #trade shares to nj from ij, equation (7)
    (wt0[i, j] * κ[n, i, j])^-θ[j] * At0[i, j] ^θ[j] / (sum((wt0[m, j] *κ[n, m, j])^-θ[j] * At0[m, j]^θ[j] for m in 1:N)) 
end

function tradeSharest1(n, i, j, d1wdot, Ldot, Adot, kdot, time, tradesharest0) #trade shares to nj from ij, equation (13)
    tradesharest0[n, i, j] * ((d1wdot[i, j] * kdot[n, i, j, time]) / pdot(n, j, d1wdot, kdot, Adot, time, tradesharest0))^-θ[j] * Adot[i, j, time]^θ[j]
end

function incomet0(n, j, wt0, Lt) # total income of country n, sector j in time t=0 given wage and labor  
    wt0[n, j] * Lt[n, j, 1] 
end

function incomet1(n, j, d1wdot, time, Lt, wt) # total income of country n, sector j in time t+1
    wt[n, j, time]*(d1wdot[n, j])*Lt[n, j, time]*(Ldot[n, j, time])
end

function Xt0(n, j, α, wt0, Lt) # expenditure on sector good j in region n, from equation (8)
    α[j] * sum(wt0[n, k] * Lt[n, k, 1] for k in 1:J) 
end

function Xt1(n, j, α, d1wdot, Lt, wt, Ldot, time) # expenditure on sector good j in region n, from equation (14)
    α[j] * sum(d1wdot[n, k] * Ldot[n, k, time] * wt[n, k, time] * Lt[n, k, time] for k in 1:J) #from equation (14) 
end


while errormax > .01 
    for time in 1:periods-1
        # Update μt step 2 of appendix D
        for n in 1:N               # destination country
            for i in 1:N           # source country
                for j in 1:J       # destination sector
                    for k in 1:J   # source sector
                        num = μt[n, i, j, k, time] *
                            (udotPathGuess[i, k, time+1])^(β/v)

                        # denominator: sum over all (m, h)
                        denom = sum(μt[n, m, j, h, time] *(udotPathGuess[m, h, time+1])^(β/v) for m in 1:N, h in 1:J)
                        μt[n, i, j, k, time+1] = num / denom
                    end
                end
            end
        end

        # Update Lt step 3 of appendix D
        for n in 1:N
            for j in 1:J
                Lt[n, j, time+1] = sum(μt[i, n, k, j, time] * Lt[i, k, time] for i in 1:N, k in 1:J)
            end
        end
        # Update Ldots from Lts
        Ldot[:, :, time] .= Lt[:, :, time+1] ./ Lt[:, :, time]
    end

    function g!(G, wt0)
        for n in 1:N
            for j in 1:J
                G[n, j] = incomet0(n, j, wt0, Lt) - sum(Xt0(i, j, α, wt0, Lt) * tradeSharest0(n, i, j, wt0, At0, κ) for i in 1:N) #market clearing for wt0
            end
        end
    end
    initial = [1.0  1.1  1.2;
           1.15 1.22 1.21]    # size: 2×3 if N=2, J=3

    results = nlsolve(g!, initial) #solve for wages

    # extract the wage solution as an N×J matrix
    # wt0 = results.zero
    
    wt[:,:, 1] = results.zero

    for n in 1:N
        for i in 1:N
            for j in 1:J
                tradesharest0[n, i , j] = tradeSharest0(n, i, j, wt0, At0, κ)
            end
        end
    end

    println(wt0)  #end solving for wt0

    for time in 1:periods-1
        function f!(F, d1wdot)
            for n in 1:N
                for j in 1:J
                    F[n,j] = (incomet1(n, j, d1wdot, time, Lt, wt) - sum(tradeSharest1(n, i, j, d1wdot, Ldot, Adot, kdot, time, tradesharest0)
                    * Xt1(i, j, α, d1wdot, Lt, wt, Ldot, time) for i in 1:N)) # equation (15)
                end
            end
        end
        println("checkpoint")
        #initial = [1.0  1.1  1.2;
        #        1.15 1.22 1.21]    # size: 2×3 if N=2, J=3
        #results = nlsolve(f!, initial) #solving for wages with country 1 set at 1.0
        initial_rand = rand(N,J)
        nlsolve(f!, initial_rand)
        println("Random guesses:", initial_rand)
    
        wdot[:, :, time] .= results.zero[:,:] # updates wdot with solution to the system of equations
        d1wdot[:,:] .= wdot[:,:,time] 

        for n in 1:N, j in 1:J
            println(wdot[n, j, time])
        end

        println("checkpoint")
        ################### not edited past this point
        for n in 1:N, j in 1:J
            pdotArray[n,j,time] = pdot(n, j, d1wdot, kdot, Adot,time, tradesharest0)
        end
    
        tradesharest0[:,:,:] .= [tradeSharest1(n,i,j,d1wdot,Ldot,Adot,kdot,time, tradesharest0) for n in 1:N, i in 1:N, j in 1:J]

        wt[:, :, time + 1] .= wdot[:, :, time] .* wt[:, :, time]

    end

    for time in 1:periods-1
        for n in 1:N, j in 1:J
            udotPathUpdate[n,j,time] = wdot[n,j,time]*(sum(μt[n,i,j,k,time]*(udotPathGuess[i,k,time+1])^(β/v) for i in 1:N, k in 1:J))^(v) ##needs verification but is equation 17 as specified by step 5
        end## Note: trying to figure out how to make sure the time+1 does not end the program with infs or NaN in the error
    end
    # Take the log difference of the guess and updated udots to get the error
    logudotPathGuess = log.(udotPathGuess) 

    logudotPathUpdate = log.(udotPathUpdate)
    logdifference = abs.(logudotPathGuess - logudotPathUpdate)
    global errormax
    errormax = maximum(logdifference[:,:,1:5])

    global udotPathGuess = udotPathUpdate
    println("WHILE LOOP")
    display(errormax)

end
