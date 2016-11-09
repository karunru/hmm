using Distributions
using StatsBase

# 状態数
state_num = 2
# 出力シンボル(この場合は 0 と 1 )
symbol = [0 1]
# 出力シンボルの数
symbol_num = length(symbol)

# これが本当の値
# 遷移確率行列
A = [0.85 0.15; 0.12 0.88]
# 出力確率行列
B = [0.8 0.2; 0.4 0.6]
# 初期確率
ρ = [1/2 1/2]

# これはパラメータ学習の初期値
# 遷移確率行列
eA = [0.7 0.3; 0.4 0.6]
# 出力確率行列
eB = [0.6 0.4; 0.5 0.5]
# 初期確率
eρ = [1/2 1/2]


srand(1234)

function simulate(nSteps)

  observations = zeros(nSteps)
  states = zeros(nSteps)
  states[1] = sample(symbol, WeightVec(vec(ρ)))
  observations[1] = sample(symbol, WeightVec(B[Int(states[1])+1, :]))
  for t in 2:nSteps
    states[t] = sample(symbol, WeightVec(A[Int(states[t-1])+1, :]))
    observations[t] = sample(symbol, WeightVec(B[Int(states[t])+1, :]))
  end
  return observations,states
end

function baum_welch(obs, A, B, ρ, eps = 1e-4, max_iter = 1000)

  # 対数尤度保持用
  old_loglikelihood = 0.0
  # 観測系列の長さ
  n = length(obs)

  for count in 1:max_iter
    # E-Step
    # scaled forwardアルゴリズム
    # 変数の初期化
    α = zeros(n, state_num)
    c = zeros(n)

    # 初期化
    α[1, :] = ρ[:] .* B[:, Int(obs[1])+1]
    c[1] = 1.0 / sum(α[1, :])
    α[1, :] = c[1] * α[1, :]

    # 再帰的計算
    for t in 2:n
      # np.dot(alpha[t-1, :], A)の代わりがダサい
      α[t, :] = [dot(α[t-1, :], A[:,1]), dot(α[t-1, :], A[:,2])] .* B[:, Int(obs[t])+1]
      # println(sum(α[t,:]))
      c[t] = 1.0 / sum(α[t, :])
      α[t, :] = c[t] * α[t, :]
    end

    # scaled backwardアルゴリズム
    # 変数の初期化
    β = zeros(n, state_num)

    # 初期化
    β[n, :] = c[n]

    # 再帰的計算
    for t in n:-1:2
      β[t-1, :] = [dot(A[1,:],(B[:, Int(obs[t]+1)] .* β[t, :])),
                    dot(A[2,:],(B[:, Int(obs[t]+1)] .* β[t, :]))]
      β[t-1, :] = c[t-1] * β[t-1, :]
    end



    # M-Step
    # update A
    newA = zeros(state_num, state_num)
    for i in 1:state_num
      for j in 1:state_num
        numer_A = denom_A = 0.0
        for t in i:n-1
          numer_A += α[t,i] * A[i,j] * B[j,Int(obs[t+1])+1] * β[t+1,j]
          denom_A += α[t,i] * β[t,j] / c[t]
        end
        newA[i,j] = numer_A / denom_A
      end
    end

    # update B
    # ダサい。
    newB = zeros(state_num, symbol_num)
    for j in 1:state_num
      for k in 1:symbol_num
        numer_B = denom_B = 0.0
        for t in 1:n
          numer_B += (obs[t] == symbol[k]) * α[t, j] * β[t, j] / c[t]
          denom_B +=  α[t, j] * β[t, j] / c[t]
        end
        newB[j,k] = numer_B / denom_B
      end
    end

    # update ρ
    newρ = α[1, :]' .* β[1, :]' / c[1]

    # update new parameters
    A = newA
    B = newB
    ρ = newρ

    # convergence check
    loglikelihood = -sum(log(c[:]))
    if abs(old_loglikelihood - loglikelihood) < eps
      break
    end
    old_loglikelihood = loglikelihood

    println("iter: ", count, " loglikelihood: ", loglikelihood)

  end

  return [A, B, ρ]
end


# 観測系列
obs, state = simulate(1000)

#obs = array([0, 1, 0])

eA, eB, eρ = baum_welch(obs, eA, eB, eρ)

println("Actual parameters")
println(A)
println(B)
println(ρ)

println("Estimated parameters")
println(eA)
println(eB)
println(eρ)
