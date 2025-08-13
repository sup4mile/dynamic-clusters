# 08/06/2025
# G (initial equilibrium equations) has norm but not F, probably code that will be used going forward unless JuMP is better.

using Random, Distributions, Optim, NLsolve, SpecialFunctions
N = 2 # Number of countries
periods = 6
J = 3 # Number of goods (needs to be big number and an integer type)
θ = fill(4.0, J) # Frechet shape parameter (governs comparative advantage)

v = 1.0 # migration elasticity
β = .99 # discount factor
α = ones(J) * (1/J) # final good expenditure share

τ = ones(N, N, J) # Iceberg trade costs 
Lt = ones(N, J, periods) # Size of labor force in each country at time 
Lt[1,1,1] = 1.1

Ldot = ones(N, J, periods)

At0 = ones(N, J)#initial productivities

wt = ones(N, J, periods)
wt0 = ones(N, J)

wdot = ones(N, J, periods)
tradesharest0 = ones(N, N, J) * (1 / N)
Adot = ones(N, J, periods)
Adot[1,1,:] .= 1.5

kdot = ones(N, N, J, periods)
pdotArray = ones(N, J, periods)
d1wdot = ones(N, J)

μtminus1 = zeros(N, N, J, J) #value of μ at time t = -1
πt0 = ones(N, N, J) #initial trade shares
μt = ones(N, N, J, J, periods)*(1/(N*J))

# a guess for path of udot:
udotPathGuess = 1.1*ones(N, J, periods+1)
udotPathUpdate = zeros(N, J, periods+1)
errormax = 1.0 #the maximum log difference between guesses and updates for appendix D algorithm


function pdot(n, j, d1wdot, kdot, Adot, time, tradesharest0) #pdot(nj) from equation (12)
    (sum(tradesharest0[n, i, j] * (d1wdot[i, j] * kdot[n, i, j, time])^-θ[j] * Adot[i, j, time]^θ[j] for i in 1:N))^(-1 / θ[j])
end 

function tradeSharest0(n, i, j, wt0, At0, τ) #trade shares to nj from ij, equation (7)
    (wt0[i, j] * τ[n, i, j])^(-θ[j]) * At0[i, j] ^θ[j] / (sum((wt0[m, j] *τ[n, m, j])^(-θ[j]) * At0[m, j]^θ[j] for m in 1:N)) 
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


while errormax > .00001 
    for time in 1:periods-1
        # Update μt step 2 of appendix D
        for n in 1:N               # destination country
            for i in 1:N           # source country
                for j in 1:J       # destination sector
                    for k in 1:J   # source sector
                        num = μt[n, i, j, k, time] *
                            (udotPathGuess[i, k, time+1])^(β/v)

                        # denominator: sum over all (m, h)
                        denom = sum(μt[n, m, j, h, time] * (udotPathGuess[m, h, time+1])^(β/v) for m in 1:N, h in 1:J)
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
        # Unpack the first N*J entries into a matrix:
        wt0 = reshape(wt0[1:N*J], N, J)
        # Fill your first N*J equations (market clearing)
        idx = 1
        for n in 1:N, j in 1:J
            G[idx] = incomet0(n, j, wt0, Lt) - sum(Xt0(i, j, α, wt0, Lt) * tradeSharest0(n, i, j, wt0, At0, τ) for i in 1:N)
            idx += 1
        end
        # The last equation pins d1[1,1] to 1.0
        G[N*J + 1] = wt0[1,1] - 1.0
    end
    
    initial = [1.0, 2.11, 1.2, 1.15, 5.22, 1.21, 1.01]    # size: 2×3 if N=2, J=3
    results = nlsolve(g!, initial) #solve for wages
    
    # extract the wage solution as an N×J matrix
    wt0 = reshape(results.zero[1:N*J], N, J)
    
    wt[:,:, 1] .= wt0[:,:]

    for n in 1:N
        for i in 1:N
            for j in 1:J
                tradesharest0[n, i , j] = tradeSharest0(n, i, j, wt0, At0, τ)
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

        initial = [1.0  2.11  1.2;
           1.15 5.22 1.21] 
        res_f = nlsolve(f!, initial)
        #println("Random guesses:", initial_rand)

        wdot[:, :, time] .= res_f.zero[:,:] # updates wdot with solution to the system of equations
        d1wdot[:,:] .= wdot[:,:,time] 

        for n in 1:N, j in 1:J
            println(wdot[n, j, time])
        end

        println("checkpoint")
        ################### not edited past this point
        for n in 1:N, j in 1:J
            pdotArray[n,j,time] = pdot(n, j, d1wdot, kdot, Adot,time, tradesharest0)
        end
    
        tradesharest0[:,:,:] .= [tradeSharest1(n,i,j,d1wdot,Ldot,Adot,kdot,time,tradesharest0) for n in 1:N, i in 1:N, j in 1:J]

        wt[:, :, time + 1] .= wdot[:, :, time] .* wt[:, :, time]

    end
    println("udotPathGuess = ", udotPathGuess)
    for time in 1:periods-1
        for n in 1:N, j in 1:J
            udotPathUpdate[n,j,time] = wdot[n,j,time]*(sum(μt[n,i,j,k,time]*(udotPathGuess[i,k,time+1])^(β/v) for i in 1:N, k in 1:J))^(v) ##needs verification but is equation 17 as specified by step 5
        end## Note: trying to figure out how to make sure the time+1 does not end the program with infs or NaN in the error
    end
    println("udotPathUpdate = ", udotPathUpdate)
    
    udotPathUpdate[:,:,periods] .= udotPathGuess[:,:,periods] # makes it so that udotPathUpdate in time 6 never goes to 0

    # Take the log difference of the guess and updated udots to get the error
    logudotPathGuess = log.(udotPathGuess) 

    logudotPathUpdate = log.(udotPathUpdate)
    logdifference = abs.(logudotPathGuess[:,:,1:5] - logudotPathUpdate[:,:,1:5])
    errormax = maximum(logdifference[:,:,1:5])

    udotPathGuess = udotPathUpdate
    println("WHILE LOOP")
    display(errormax)
end