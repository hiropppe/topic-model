import numpy as np

from collections import Counter
from scipy.stats import multinomial
from scipy.special import rel_entr
from scipy.optimize import linear_sum_assignment

from pprint import pprint
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.style.use("ggplot")


def generate_lda_toy_data(D, V, K, alpha=1.0, beta=1.0, seed=123):
    np.random.seed(seed)

    ## モデルのパラメータを準備
    # トピック分布のハイパーパラメータ
    alpha_k = [alpha] * K
    #rng = np.random.default_rng()
    #alpha_k = rng.uniform(low=1, high=2, size=K)

    # トピック分布のパラメータを生成
    theta_dk = np.random.dirichlet(alpha=alpha_k, size=D)
    assert np.all(np.abs(theta_dk.sum(axis=1) - 1.0) < 1e-5)

    # 単語分布のハイパーパラメータを設定
    beta_v = [beta] * V
    #rng = np.random.default_rng()
    #beta_v = rng.uniform(low=1, high=2, size=V)

    # 単語分布のパラメータを生成
    phi_kv = np.random.dirichlet(alpha=beta_v, size=K)
    assert np.all(np.abs(phi_kv.sum(axis=1) - 1.0) < 1e-5)
    
    ## テスト文書を生成
    W = []                  # 文書集合を初期化
    Z = []                  # トピック集合を初期化
    N_d = [None] * D        # 各文書の単語数を初期化
    N_dw = np.zeros((D, V)) # 文書ごとの各語彙の出現頻度を初期化
    
    min_N_d = 100 # 各文書の単語数の上限
    max_N_d = 200 # 各文書の単語数の下限

    print("generating...")
    for d in tqdm(range(D)):
        # 単語数を生成
        N_d[d] = np.random.randint(low=min_N_d, high=max_N_d)
        # 各単語のトピックを初期化
        z_dn = [None] * N_d[d]
        # 各単語の語彙を初期化
        w_dn = [None] * N_d[d]
    
        for n in range(N_d[d]):
            # トピックを生成
            k = np.random.choice(K, p=theta_dk[d])
            z_dn[n] = k
            # 語彙を生成
            w = np.random.choice(V, p=phi_kv[k])
            w_dn[n] = w
            # 頻度をカウント
            N_dw[d, w] += 1
    
        # トピック集合を格納
        Z.append(z_dn)
        W.append(w_dn)

    ## 生成物をファイルに出力
    # テスト文書を出力
    with open("lda.test.txt", mode="w") as f:
        print("\n".join([" ".join([str(w) for w in words]) for words in W]), file=f)

    # テスト文書 (BoW) をに出力
    with open("lda.test.bow.txt", mode="w") as f:
        print("\n".join([" ".join([f"{k}:{v}" for k, v in sorted(Counter(w).items(), key=lambda x: x[0])]) for w in W]), file=f)

    # 語彙ID を出力
    with open("lda.test.word2id.txt", mode="w") as f:
        print("\n".join([f"{w}\t{w}" for w in range(V)]), file=f)

    # 分布のパラメータを出力
    params = {
        "K": K,
        "alpha": alpha,
        "beta": beta,
        "phi": phi_kv,
        "theta": theta_dk,
    }
    np.savez("lda.test.params.npz", **params)

    return params


def assign_phi(true_phi, phi):
    K = true_phi.shape[0]
    # 2つの分布間の KL ダイバージェンス
    def kl_divergence(p, q):
        # log(0)を避けるために小さな値を加える
        p = np.asarray(p) + 1e-12
        q = np.asarray(q) + 1e-12
        return np.sum(rel_entr(p, q))
    
    # 真の分布と推定分布の間のKLダイバージェンス行列を計算する
    kl_matrix = np.zeros((K, K))
    for i in range(K):  # 真のトピック i
        for j in range(K):  # 推定されたトピック j
            kl_matrix[i, j] = kl_divergence(true_phi[i], phi[j])
    
    # トピックを一致させるために割り当て問題 （Hungarian algorithm） を解く
    row_ind, col_ind = linear_sum_assignment(kl_matrix)
    
    # マッチングとダイバージェンスの表示
    print("\nTopic Matching (True -> Estimated):")
    for i, j in zip(row_ind, col_ind):
        print(f"True Topic {i} matched with Estimated Topic {j} | KL Divergence: {kl_matrix[i, j]:.4f}")
    
    # マッチング後の平均　KL　ダイバージェンスを計算する
    mean_kl = np.mean([kl_matrix[i, j] for i, j in zip(row_ind, col_ind)])
    print(f"\nMean KL Divergence after optimal topic matching: {mean_kl:.4f}")

    # マッチングに従ってトピック番号を割り当て 
    assign = {i:j for i, j in zip(row_ind, col_ind)}
    return assign, np.stack([phi[assign[k]] for k in range(K)])


def assign_theta(theta, assign):
    D = theta.shape[0]
    K = theta.shape[1]
    new_theta = []
    for d in range(D):
        new_theta.append(np.stack([theta[d][assign[k]] for k in range(K)]))
    return np.stack(new_theta)


def plot_phi(phi_kv, true_phi, n_rows):
    K = phi_kv.shape[0]
    V = phi_kv.shape[1]
    n_cols = K//n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*1.5), sharex=True, sharey=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"$\phi$", fontsize=15)
    v_vals = np.arange(1, V + 1)
    p_max = np.max(np.concatenate([phi_kv, true_phi]))
    for k in range(K):
        if axes.ndim == 1:
            ax = axes[k]
        else:
            row, col = k//n_cols, k%n_cols
            ax = axes[row, col]
        ax.bar(x=v_vals, height=list(phi_kv[k]), color='#00A968', alpha=0.5)
        ax.bar(x=v_vals, height=list(true_phi[k]), color='#a80041', alpha=0.5)
        ax.text(0.05, 0.8, f'topic #{k}', transform=ax.transAxes, ha='center', color='red')

    for ax in axes.flat:
        ax.set_yticks(ticks=np.linspace(0, p_max, num=5))
        ax.label_outer()

    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    plt.show()


def plot_theta(theta_dk, true_theta, n_rows):
    D = theta_dk.shape[0]
    K = theta_dk.shape[1]
    n_cols = D//n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*2), sharex=True, sharey=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"$\theta$", fontsize=15)
    k_vals = np.arange(1, K + 1)
    for d in range(D):
        if axes.ndim == 1:
            ax = axes[d]
        else:
            row, col = d//n_cols, d%n_cols
            ax = axes[row, col]
        ax.bar(x=k_vals, height=list(theta_dk[d]), color='#00A968', alpha=0.5)
        ax.bar(x=k_vals, height=list(true_theta[d]), color='#a80041', alpha=0.5)
        #ax.set_title(r'$\theta=(' + ', '.join([str(theta) for theta in theta_dk[d].round(2)]) + ')$', loc='left', fontsize=9)
        ax.text(0.2, 0.9, f'doc #{d}', transform=ax.transAxes, ha='center', color='red', fontsize=10)
    
    for ax in axes.flat:
        ax.set_xticks(ticks=k_vals)
        ax.set_yticks(ticks=np.linspace(0, 1, num=5))
        ax.label_outer()
        
    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    plt.show()

