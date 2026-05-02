
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tucker_hooi_gaussian_whiten import Tucker

ppp = pd.read_csv("ppp.csv")
cols = ["start_lon", "start_lat","tor_length", "tor_width"]#,"end_lat","end_lon" ]

#cols = ["start_lon", "start_lat", "end_lat","end_lon"]

data_locations = ppp[[c for c in cols if c in ppp.columns]].copy()

data=np.array(data_locations)

from distance import  sliced_wasserstein_2


train_min = data.min(axis=0)
train_max = data.max(axis=0)
denom= np.maximum(train_max - train_min, 1e-12)
data = (data - train_min) / denom

RR = 40

e_TT = np.zeros(RR)
e_kde = np.zeros(RR)
e_vae = np.zeros(RR)
e_diff = np.zeros(RR)

n = 25

from kde import kernel_density
from VAE import unconditional_samples_vae
from diffusion import TorchCFMTabularGenerator

for rr in range(RR):

    X_train, X_test = train_test_split(data, test_size=0.3)
    N = X_test.shape[0] * 4

    TT_model = Tucker(
        n,
        X_train,
        10,
        threshold=0.1,
        rank_rule="relative",
        verbose=True,
    )

    X_TT = TT_model.sample(N, verbose=False)
    e_TT[rr] = sliced_wasserstein_2(X_test, X_TT)
    print("TT", rr, e_TT[rr])

    kde = kernel_density(X_train)
    X_kde = kde.sample(N)
    e_kde[rr] = sliced_wasserstein_2(X_test, X_kde)
    print("kde", rr, e_kde[rr])

    X_vae = unconditional_samples_vae(X_train, N, verbose=True)
    e_vae[rr] = sliced_wasserstein_2(X_test, X_vae)
    print("vae", rr, e_vae[rr])

    diff = TorchCFMTabularGenerator(
        hidden_dims=(128, 128, 128),
        lr=3e-4,
        batch_size=256,
        epochs=200,
        sigma=0.03,
        standardize=True,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_ema=True,
        ema_decay=0.9995,
        sample_solver="rk4",
        sample_steps=128,
        device=None,
    ).fit(X_train)

    X_diff = diff.sample(N, chunk_size=10000)
    e_diff[rr] = sliced_wasserstein_2(X_test, X_diff)
    print("diff", rr, e_diff[rr])

    print("e_TT", e_TT[: rr + 1])
    print("e_kde", e_kde[: rr + 1])
    print("e_vae", e_vae[: rr + 1])
    print("e_diff", e_diff[: rr + 1])
    