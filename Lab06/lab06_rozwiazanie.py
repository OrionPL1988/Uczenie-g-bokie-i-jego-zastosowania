# -*- coding: utf-8 -*-
"""Lab06: PCA/SVD/LDA, autoenkodery, k-means, DAE/CAE i anomalie (MNIST).

Uruchomienie lokalne:
    python lab06_rozwiazanie.py --fast
    python lab06_rozwiazanie.py --balanced
    python lab06_rozwiazanie.py --full --bonus
"""
import os
import argparse
from pathlib import Path

def _parametry():
    ap = argparse.ArgumentParser(description="Lab06 - lokalnie bez Colaba")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true", help="krótki test: 2000/500, 2 epoki")
    g.add_argument("--balanced", action="store_true", help="kompromis dla CPU: 12000/2500, 8 epok")
    g.add_argument("--full", action="store_true", help="pełne MNIST 60000/10000, 20 epok")
    ap.add_argument("--bonus", action="store_true", help="sparse AE oraz 3 klasy anomalii")
    return ap.parse_args()

_args = _parametry()
_profil = "full" if _args.full else "balanced" if _args.balanced else "fast"
os.environ["DL_PROFILE"] = _profil
os.environ["DL_BONUS"] = "1" if _args.bonus else "0"
os.environ["DL_NONINTERACTIVE"] = "1"
os.environ["MPLBACKEND"] = "Agg"
# Xeon 4C/8T: umiarkowana liczba wątków zamiast 32 procesów walczących o CPU.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ["DL_OUTPUT_DIR"] = str(Path(__file__).resolve().parent / ("wyniki_" + _profil))
print("Lab06: start", _profil, "bonus", _args.bonus, flush=True)


# Lab 6 - Autoenkodery, redukcja wymiarowości i klasteryzacja

# Cel zajęć

# 0. Przygotowanie środowiska - wersja lokalna (VS Code / Windows)

# ==========================================================================
# CELL 03: 
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import gc
import json
import re
import time
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, ops
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

print("TensorFlow:", tf.__version__, "| Keras:", keras.__version__, "| backend:", keras.backend.backend())


# ==========================================================================
# CELL 04: 
# Nie korzystamy z Colaba. Profil ustawiany parametrem --fast/--balanced/--full.
gpus = tf.config.list_physical_devices("GPU")
print("GPU widziane przez TensorFlow:", gpus if gpus else "brak, liczę na CPU")
SEED = 42
keras.utils.set_random_seed(SEED)
DL_PROFILE = os.environ.get("DL_PROFILE", "fast").lower().strip()
if DL_PROFILE not in ("fast", "balanced", "full"):
    raise ValueError("DL_PROFILE musi mieć wartość fast, balanced albo full")
SZYBKI_TRYB = DL_PROFILE == "fast"
BONUS = os.environ.get("DL_BONUS", "0") == "1"
# Pełny tryb zachowuje liczby próbek i epok z instrukcji prowadzącej.
USTAWIENIA = {
    "fast": dict(EPOKI=2, EPOKI_KROTKO=1, EPOKI_LIN=2, N_TRAIN=2000, N_TEST=500, N_VIZ=500),
    "balanced": dict(EPOKI=8, EPOKI_KROTKO=4, EPOKI_LIN=12, N_TRAIN=12000, N_TEST=2500, N_VIZ=2000),
    "full": dict(EPOKI=20, EPOKI_KROTKO=10, EPOKI_LIN=30, N_TRAIN=60000, N_TEST=10000, N_VIZ=5000),
}
globals().update(USTAWIENIA[DL_PROFILE])
BATCH = 256
OUT_DIR = Path(os.environ.get("DL_OUTPUT_DIR", str(Path.cwd() / f"wyniki_{DL_PROFILE}")))
WYKRESY = OUT_DIR / "wykresy"
WYKRESY.mkdir(parents=True, exist_ok=True)
PLIKOWANE_WYKRESY = []
_prawdziwy_show = plt.show
def _pokaz_i_zapisz(*args, **kwargs):
    # W PyCharm/VS Code okienka potrafią wstrzymać kod. Zapis PNG jest pewniejszy.
    for nr in plt.get_fignums():
        fig = plt.figure(nr)
        if getattr(fig, "_lab06_zapisany", False):
            continue
        tytul = fig._suptitle.get_text() if fig._suptitle else ""
        if not tytul:
            tytul = next((ax.get_title() for ax in fig.axes if ax.get_title()), "wykres")
        nazwa = unicodedata.normalize("NFKD", tytul).encode("ascii", "ignore").decode().lower()
        nazwa = re.sub(r"[^a-z0-9]+", "_", nazwa).strip("_")[:65] or "wykres"
        sciezka = WYKRESY / f"{len(PLIKOWANE_WYKRESY)+1:02d}_{nazwa}.png"
        fig.savefig(sciezka, dpi=145, bbox_inches="tight")
        PLIKOWANE_WYKRESY.append(sciezka.name)
        fig._lab06_zapisany = True
    if os.environ.get("DL_NONINTERACTIVE") == "1":
        plt.close("all")
    else:
        _prawdziwy_show(*args, **kwargs)
plt.show = _pokaz_i_zapisz
CZAS_START = time.perf_counter()
print(f"Profil: {DL_PROFILE} | train {N_TRAIN} | test {N_TEST} | epoki {EPOKI} | bonus: {BONUS}")
print("Wyniki:", OUT_DIR)


# 1. Dane: MNIST

# ==========================================================================
# CELL 06: 
try:
    # MNIST pobiera sie raz; kolejne przebiegi korzystaja z cache Keras.
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.mnist.load_data()
except Exception as exc:
    raise RuntimeError('Nie udalo sie pobrac MNIST. Sprawdz polaczenie z internetem.') from exc

# Podzbiory zależne od trybu (w pełnym trybie - całość)
x_train = x_train_full[:N_TRAIN].astype("float32") / 255.0   # skalowanie do [0, 1]
y_train = y_train_full[:N_TRAIN]
x_test = x_test_full[:N_TEST].astype("float32") / 255.0
y_test = y_test_full[:N_TEST]

# Wersje spłaszczone (N, 784) - dla PCA, SVD, LDA i AE gęstego
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)
# Wersje z kanałem (N, 28, 28, 1) - dla AE konwolucyjnego
x_train_img = x_train[..., None]
x_test_img = x_test[..., None]

print("train:", x_train_flat.shape, "test:", x_test_flat.shape, "zakres:", x_train.min(), x_train.max())

# ==========================================================================
# CELL 07: 
# Podgląd kilku obrazów z etykietami
fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(f"y={y_train[i]}", fontsize=9)
    ax.axis("off")
fig.suptitle("Przykładowe obrazy MNIST")
plt.tight_layout(); plt.show()

# ==========================================================================
# CELL 08: 
# Funkcje pomocnicze używane w całym notebooku

def mse_per_sample(x, x_hat):
    # Błąd średniokwadratowy rekonstrukcji osobno dla każdego obrazu
    x = np.asarray(x).reshape(len(x), -1)
    x_hat = np.asarray(x_hat).reshape(len(x_hat), -1)
    return np.mean((x - x_hat) ** 2, axis=1)

def pokaz_rekonstrukcje(oryginaly, rekonstrukcje, tytuly=("oryginał", "rekonstrukcja"), n=10, suptitle=""):
    # Rysuje w wierszach: oryginały i jedną lub więcej wersji rekonstrukcji
    wiersze = [oryginaly] + (list(rekonstrukcje) if isinstance(rekonstrukcje, (list, tuple)) else [rekonstrukcje])
    fig, axes = plt.subplots(len(wiersze), n, figsize=(1.2 * n, 1.3 * len(wiersze)))
    axes = np.atleast_2d(axes)
    for r, dane in enumerate(wiersze):
        for i in range(n):
            axes[r, i].imshow(np.asarray(dane[i]).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
        axes[r, 0].set_title(tytuly[r] if r < len(tytuly) else "", fontsize=8, loc="left")
    fig.suptitle(suptitle)
    plt.tight_layout(); plt.show()

def scatter_2d(z, y, tytul, ax=None):
    # Wykres punktów 2D pokolorowanych etykietą cyfry
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    z = np.asarray(z)
    if z.shape[1] == 1:   # zabezpieczenie: metoda zwróciła tylko 1 wymiar (np. LDA na zdegenerowanych danych)
        z = np.column_stack([z[:, 0], np.random.default_rng(0).normal(scale=0.05, size=len(z))])
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=4, alpha=0.7)
    ax.set_title(tytul); ax.set_xlabel("wymiar 1"); ax.set_ylabel("wymiar 2")
    ax.legend(*sc.legend_elements(), title="cyfra", fontsize=7, loc="best", markerscale=0.8)
    return ax

# 2. PCA - analiza składowych głównych

# PCA "ręcznie" przez SVD

# ==========================================================================
# CELL 11: 
X_sub = x_train_flat[:600 if SZYBKI_TRYB else 2000]                                   # podzbiór, żeby SVD liczyło się szybko
mu = X_sub.mean(axis=0)                                       # średni obraz
U, S, Vt = np.linalg.svd(X_sub - mu, full_matrices=False)     # SVD wycentrowanych danych
lam_svd = S ** 2 / (len(X_sub) - 1)                           # wartości własne macierzy kowariancji

pca_ref = PCA(n_components=5).fit(X_sub)
print("wartości własne (SVD):    ", lam_svd[:5].round(3))
print("wartości własne (sklearn):", pca_ref.explained_variance_.round(3))
print("|cos| kąta między kierunkami:", np.abs(np.sum(Vt[:5] * pca_ref.components_, axis=1)).round(4))

plt.figure(figsize=(2.5, 2.5)); plt.imshow(mu.reshape(28, 28), cmap="gray"); plt.axis("off")
plt.title("średni obraz $\\bar{x}$"); plt.show()

# ==========================================================================
# CELL 12: 
# Jedno dopasowanie pełnego PCA, a przy rekonstrukcji bierzemy pierwsze k osi.
# Dzięki temu nie liczymy SVD od nowa dla każdego k - oszczędność spora.
pca_full = PCA(svd_solver="covariance_eigh", random_state=SEED).fit(x_train_flat)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)

def pca_rekonstrukcja(k, X):
    skl = pca_full.components_[:int(k)]
    X = np.asarray(X, dtype=np.float32)
    return ((X - pca_full.mean_) @ skl.T) @ skl + pca_full.mean_

def pca_wspolrzedne(k, X):
    skl = pca_full.components_[:int(k)]
    return (np.asarray(X) - pca_full.mean_) @ skl.T

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(np.arange(1, 51), evr[:50], "o-", ms=3, label="pojedyncza składowa")
ax[0].set_xlabel("numer składowej")
ax[0].set_ylabel("udział wariancji")
ax[0].set_title("Widmo PCA (50 pierwszych)")
ax[0].legend()
ax[1].plot(np.arange(1, len(cum) + 1), cum, label="skumulowana")
for prog in (0.8, 0.9, 0.95):
    k_p = int(np.searchsorted(cum, prog) + 1)
    ax[1].axhline(prog, ls="--", lw=0.9, color="gray")
    ax[1].axvline(k_p, ls=":", lw=1)
    ax[1].text(k_p + 3, prog - 0.035, f"{int(100*prog)}%: k={k_p}", fontsize=8)
ax[1].set_xlabel("liczba składowych k")
ax[1].set_ylabel("wyjaśniona wariancja")
ax[1].set_title("Skumulowana wyjaśniona wariancja")
ax[1].legend()
plt.tight_layout(); plt.show()
print(f"2 składowe wyjaśniają {cum[1]:.1%} wariancji")


# ==========================================================================
# CELL 13: 
# Rzut na 2 pierwsze składowe i wizualizacja (dla N_VIZ punktów testowych)
pca2 = PCA(n_components=2, random_state=SEED).fit(x_train_flat)
z_pca2 = pca2.transform(x_test_flat[:N_VIZ])
scatter_2d(z_pca2, y_test[:N_VIZ], "PCA - 2 składowe (zbiór testowy)")
plt.show()

# Pierwsze składowe główne same są "obrazami" 28x28 - można je obejrzeć
fig, axes = plt.subplots(1, 8, figsize=(12, 1.8))
for i, ax in enumerate(axes):
    ax.imshow(pca_full.components_[i].reshape(28, 28), cmap="RdBu_r")
    ax.set_title(f"PC{i+1}", fontsize=9); ax.axis("off")
fig.suptitle("Składowe główne jako obrazy (czerwony +, niebieski -)")
plt.tight_layout(); plt.show()

# **Co obserwować:** cyfra 1 i cyfra 0 zwykle leżą na przeciwnych końcach osi PC1 (PC1 mierzy z grubsza "ile tuszu jest na obrzeżach vs w środku"). Pozostałe cyfry mocno się nakładają - dwie liniowe osie to za mało.

# ==========================================================================
# CELL 15: 
k_progi = {prog: int(np.searchsorted(cum, prog) + 1) for prog in (0.8, 0.9, 0.95)}
mse_pca_k = {}
rek_lista = []
for k in (2, 10, 50, 150):
    rek = pca_rekonstrukcja(k, x_test_flat)
    # MSE bez przycinania wartości do 0..1. Inaczej PCA ma nieuczciwą ulgę.
    mse_pca_k[k] = float(mse_per_sample(x_test_flat, rek).mean())
    rek_lista.append(np.clip(rek[:10], 0, 1))
print("k dla progów wariancji:", k_progi)
print("MSE rekonstrukcji PCA:", {k: round(v, 5) for k, v in mse_pca_k.items()})
pokaz_rekonstrukcje(x_test_flat[:10], rek_lista,
                    tytuly=("oryginał", "k=2", "k=10", "k=50", "k=150"),
                    suptitle="Rekonstrukcje PCA dla różnych k")


# 3. TruncatedSVD i LDA

# ==========================================================================
# CELL 17: 
# TruncatedSVD na 2 składowe (bez centrowania danych)
svd2 = TruncatedSVD(n_components=2, random_state=SEED).fit(x_train_flat)
z_svd2 = svd2.transform(x_test_flat[:N_VIZ])
print("Wyjaśniona wariancja przez 2 składowe SVD:", svd2.explained_variance_ratio_.round(3))
scatter_2d(z_svd2, y_test[:N_VIZ], "TruncatedSVD - 2 składowe")
plt.show()

# Zadanie 3.1 - LDA i porównanie trzech metod

# ==========================================================================
# CELL 19: 
# LDA wykorzystuje etykiety treningowe. PCA/SVD tego nie robią - ważna różnica.
lda2 = LinearDiscriminantAnalysis(n_components=2, solver="svd")
z_lda2 = lda2.fit(x_train_flat, y_train).transform(x_test_flat[:N_VIZ])
try:
    LinearDiscriminantAnalysis(n_components=10).fit(x_train_flat, y_train)
except ValueError as exc:
    print("LDA(10) nie działa dla 10 klas, bo maksymalnie C-1=9:", exc)
lda9 = LinearDiscriminantAnalysis(n_components=9)
lda9.fit(x_train_flat, y_train)
print("LDA(9) - suma explained_variance_ratio_:",
      round(float(lda9.explained_variance_ratio_.sum()), 5))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
scatter_2d(z_pca2, y_test[:N_VIZ], "PCA 2D (bez etykiet)", ax=ax[0])
scatter_2d(z_svd2, y_test[:N_VIZ], "TruncatedSVD 2D", ax=ax[1])
scatter_2d(z_lda2, y_test[:N_VIZ], "LDA 2D (z etykietami)", ax=ax[2])
fig.suptitle("PCA / SVD / LDA - porównanie projekcji 2D")
plt.tight_layout(); plt.show()


# 4. Autoenkoder gęsty

# ==========================================================================
# CELL 21: 
def zbuduj_ae_gesty(latent_dim, ukryte=(512, 256), aktywacja_latentu=None, nazwa="ae", regularizer_latentu=None):
    # Enkoder: 784 -> 512 -> 256 -> latent_dim
    wej = keras.Input(shape=(784,), name="obraz")
    h = wej
    for n in ukryte:
        h = layers.Dense(n, activation="relu")(h)
    z = layers.Dense(latent_dim, activation=aktywacja_latentu, activity_regularizer=regularizer_latentu, name="z")(h)   # liniowy latent (domyślnie)
    enkoder = keras.Model(wej, z, name=f"{nazwa}_enkoder")

    # Dekoder: lustrzane odbicie enkodera, latent_dim -> 256 -> 512 -> 784
    wej_z = keras.Input(shape=(latent_dim,), name="z_wej")
    h = wej_z
    for n in reversed(ukryte):
        h = layers.Dense(n, activation="relu")(h)
    wyj = layers.Dense(784, activation="sigmoid", name="rekonstrukcja")(h)    # sigmoid -> piksele w [0, 1]
    dekoder = keras.Model(wej_z, wyj, name=f"{nazwa}_dekoder")

    # Autoenkoder = dekoder(enkoder(x))
    ae = keras.Model(wej, dekoder(enkoder(wej)), name=nazwa)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["mse"])
    return ae, enkoder, dekoder

ae2, enc2, dec2 = zbuduj_ae_gesty(2, nazwa="ae2")
enc2.summary()
dec2.summary()

# Uwaga do straty: używamy **BCE** dla pikseli w $[0,1]$ - działa jak "miękka" klasyfikacja każdego piksela (tło/tusz) i zwykle daje ostrzejsze rekonstrukcje niż MSE. Metrykę MSE zostawiamy, żeby porównywać się z PCA (które minimalizuje MSE).

# ==========================================================================
# CELL 23: 
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]

t0 = time.time()
hist2 = ae2.fit(x_train_flat, x_train_flat,          # wejście = cel!
                epochs=EPOKI, batch_size=BATCH, validation_split=0.1,
                callbacks=callbacks, verbose=2)
print(f"Czas treningu: {time.time() - t0:.1f} s")

plt.figure(figsize=(6, 4))
plt.plot(hist2.history["loss"], "o-", label="train")
plt.plot(hist2.history["val_loss"], "o-", label="val")
plt.xlabel("epoka"); plt.ylabel("BCE"); plt.title("Krzywe uczenia AE (latent 2)"); plt.legend(); plt.show()

# ==========================================================================
# CELL 24: 
# Przestrzeń latentna 2D: każdy obraz testowy -> punkt (z1, z2)
z_ae2 = enc2.predict(x_test_flat[:N_VIZ], verbose=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
scatter_2d(z_pca2, y_test[:N_VIZ], "PCA 2D", ax=axes[0])
scatter_2d(z_ae2, y_test[:N_VIZ], "Autoenkoder - latent 2D", ax=axes[1])
plt.tight_layout(); plt.show()

# Rekonstrukcje z zaledwie 2 liczb
rek2 = ae2.predict(x_test_flat[:10], verbose=0)
pokaz_rekonstrukcje(x_test_flat[:10], rek2, tytuly=("oryginał", "AE d=2"), suptitle="Rekonstrukcje AE z latentem 2D")

# ==========================================================================
# CELL 25: 
# Dekoder jako "generator": przechodzimy po siatce punktów w przestrzeni latentnej
n = 12
z1 = np.linspace(np.percentile(z_ae2[:, 0], 5), np.percentile(z_ae2[:, 0], 95), n)
z2 = np.linspace(np.percentile(z_ae2[:, 1], 5), np.percentile(z_ae2[:, 1], 95), n)
siatka = np.array([[a, b] for b in z2[::-1] for a in z1], dtype="float32")   # n*n punktów
obrazy = dec2.predict(siatka, verbose=0).reshape(n, n, 28, 28)
plansza = obrazy.transpose(0, 2, 1, 3).reshape(n * 28, n * 28)               # sklejamy w jeden obraz
plt.figure(figsize=(7, 7)); plt.imshow(plansza, cmap="gray"); plt.axis("off")
plt.title("Dekoder AE: obrazy z siatki punktów latentnych"); plt.show()

# **Typowe błędy przy autoenkoderach:**

# ==========================================================================
# CELL 27: 
# Rozkład współrzędnych latentnych AE 2D - porównaj z rozkładem normalnym N(0, 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for j in range(2):
    ax[j].hist(z_ae2[:, j], bins=50, alpha=0.8, density=True, label=f"z{j+1}")
    ax[j].set_xlabel(f"wartość z{j+1}"); ax[j].set_ylabel("gęstość"); ax[j].set_title(f"Rozkład z{j+1} (AE)")
    ax[j].legend()
plt.tight_layout(); plt.show()
print("średnia:", z_ae2.mean(0).round(2), "odch. std.:", z_ae2.std(0).round(2))

# **Co obserwować:** w przestrzeni latentnej AE klasy tworzą wyraźniejsze, "promieniste" skupiska niż w PCA, mimo że AE nie widział etykiet. Na siatce dekodera widać płynne przejścia między cyframi, ale też obszary "pomiędzy" dające nieczytelne kształty - przestrzeń latentna zwykłego AE **nie jest regularna** (dziury, nierówna gęstość). Ten problem rozwiązuje VAE na Lab 7.

# ==========================================================================
# CELL 29: 
ae32, enc32, dec32 = zbuduj_ae_gesty(32, nazwa="ae32")
hist32 = ae32.fit(x_train_flat, x_train_flat,
                  epochs=EPOKI, batch_size=BATCH, validation_split=0.1,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                           restore_best_weights=True)], verbose=2)
rek_ae32 = ae32.predict(x_test_flat, batch_size=BATCH, verbose=0)
rek_pca32 = pca_rekonstrukcja(32, x_test_flat)
mse_ae32 = float(mse_per_sample(x_test_flat, rek_ae32).mean())
mse_pca32 = float(mse_per_sample(x_test_flat, rek_pca32).mean())
pokaz_rekonstrukcje(x_test_flat[:10], [np.clip(rek_pca32[:10], 0, 1), rek_ae32[:10]],
                    tytuly=("oryginał", "PCA 32", "AE 32"),
                    suptitle="Porównanie rekonstrukcji PCA i AE (d=32)")
print(f"MSE PCA(32) = {mse_pca32:.5f}, MSE AE(32) = {mse_ae32:.5f}")
if mse_ae32 >= mse_pca32:
    print("Uwaga: AE nie pokonał PCA. Przy małym profilu FAST to możliwe - za mało epok.")


# 5. Błąd rekonstrukcji w funkcji wymiaru: AE vs PCA

# ==========================================================================
# CELL 31: 
WYMIARY = [2, 8, 32] if SZYBKI_TRYB else [2, 4, 8, 16, 32, 64]
wyniki_d = []
for d in WYMIARY:
    # Dla PCA używamy tych samych składowych wyznaczonych na train.
    mse_pca = float(mse_per_sample(x_test_flat, pca_rekonstrukcja(d, x_test_flat)).mean())
    # Osobny trening KAŻDEGO d z takim samym budżetem EPOKI_KROTKO.
    # Nie mieszamy tu ae32 trenowanego dłużej, bo porównanie byłoby nierówne.
    mdl, _, _ = zbuduj_ae_gesty(d, nazwa=f"ae_dim_{d}")
    mdl.fit(x_train_flat, x_train_flat, epochs=EPOKI_KROTKO,
            batch_size=BATCH, validation_split=0.1, verbose=0)
    mse_ae = float(mse_per_sample(x_test_flat,
                mdl.predict(x_test_flat, batch_size=BATCH, verbose=0)).mean())
    wyniki_d.append({"d": int(d), "mse_pca": mse_pca, "mse_ae": mse_ae})
    print(f"d={d:3d}: PCA {mse_pca:.5f}, AE {mse_ae:.5f}")
    del mdl
    gc.collect()
df_d = pd.DataFrame(wyniki_d)
display(df_d.round(5))
if len(df_d):
    plt.figure(figsize=(7, 4.5))
    plt.plot(df_d["d"], df_d["mse_pca"], "o-", label="PCA")
    plt.plot(df_d["d"], df_d["mse_ae"], "s-", label="Autoenkoder")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("wymiar przestrzeni ukrytej d")
    plt.ylabel("MSE rekonstrukcji (test)")
    plt.title("Błąd rekonstrukcji: AE vs PCA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
print("Hipoteza o przewadze AE dla wszystkich d:",
      "potwierdzona w tym przebiegu" if (df_d.mse_ae < df_d.mse_pca).all()
      else "niepotwierdzona dla części d (zobacz czas treningu i konfigurację)")


# 6. Liniowy autoenkoder ≈ PCA

# ==========================================================================
# CELL 33: 
from scipy.linalg import subspace_angles
# Lin AE: bez ReLU, bez sigmoidu, loss=MSE. To ma być uczciwe porównanie do PCA.
inp_lin = keras.Input(shape=(784,))
z_lin = layers.Dense(16, use_bias=True, name="latent_liniowy")(inp_lin)
out_lin = layers.Dense(784, use_bias=True, name="rek_liniowa")(z_lin)
lin_ae = keras.Model(inp_lin, out_lin, name="liniowy_ae")
lin_ae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
hist_lin = lin_ae.fit(x_train_flat, x_train_flat, epochs=EPOKI_LIN,
                      batch_size=BATCH, validation_split=0.1, verbose=0)
rek_lin = lin_ae.predict(x_test_flat, batch_size=BATCH, verbose=0)
mse_lin_ae = float(mse_per_sample(x_test_flat, rek_lin).mean())
mse_pca16 = float(mse_per_sample(x_test_flat, pca_rekonstrukcja(16, x_test_flat)).mean())
W_dec = lin_ae.get_layer("rek_liniowa").get_weights()[0]  # 16x784
katy = np.degrees(subspace_angles(W_dec.T, pca_full.components_[:16].T))
print(f"MSE liniowy AE = {mse_lin_ae:.6f}, MSE PCA(16) = {mse_pca16:.6f}")
print("Kąty główne [°] (od największego):", np.round(np.sort(katy)[::-1], 1))
print("To porównanie dotyczy podprzestrzeni, a nie równych wag modelu i PCA.")


# 7. Klasteryzacja w różnych przestrzeniach cech

# ==========================================================================
# CELL 35: 
def ocen_klasteryzacje(X, y, nazwa, k=10):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    etyk = km.labels_
    # Silhouette liczymy na podpróbce - koszt rośnie kwadratowo z liczbą obrazów.
    n_sil = min(500 if SZYBKI_TRYB else 2000, len(X))
    return {"przestrzeń": nazwa, "wymiar": int(X.shape[1]),
            "ARI": float(adjusted_rand_score(y, etyk)),
            "NMI": float(normalized_mutual_info_score(y, etyk)),
            "silhouette": float(silhouette_score(X, etyk, sample_size=n_sil, random_state=SEED))}
Xk, yk = x_test_flat[:N_VIZ], y_test[:N_VIZ]
print(ocen_klasteryzacje(Xk, yk, "piksele"))


# Zadanie 7.1 - tabela porównawcza klasteryzacji

# ==========================================================================
# CELL 37: 
lista = []
lista.append(ocen_klasteryzacje(Xk, yk, "piksele"))
lista.append(ocen_klasteryzacje(pca_wspolrzedne(32, Xk), yk, "PCA (32)"))
z_ae32_k = enc32.predict(Xk, batch_size=BATCH, verbose=0)
z_ae2_k = enc2.predict(Xk, batch_size=BATCH, verbose=0)
lista.append(ocen_klasteryzacje(z_ae32_k, yk, "AE (32)"))
lista.append(ocen_klasteryzacje(z_ae2_k, yk, "AE (2)"))
lista.append(ocen_klasteryzacje(pca_wspolrzedne(2, Xk), yk, "PCA (2)"))
df_klast = pd.DataFrame(lista).sort_values("ARI", ascending=False).reset_index(drop=True)
display(df_klast.round(3))
ax = df_klast.plot.bar(x="przestrzeń", y=["ARI", "NMI"], figsize=(8, 4.5), rot=25)
ax.set_ylabel("wartość metryki")
ax.set_title("k-means w różnych przestrzeniach: ARI i NMI")
plt.tight_layout(); plt.show()
print("Najlepsza przestrzeń wg ARI:", df_klast.iloc[0]["przestrzeń"])
print("Silhouette między różnymi wymiarami nie porównujemy wprost.")


# Które cyfry k-means myli? Macierz kontyngencji

# ==========================================================================
# CELL 39: 
km_ae = KMeans(n_clusters=10, n_init=10, random_state=SEED).fit(enc32.predict(Xk, verbose=0))
tab = pd.crosstab(pd.Series(yk, name="cyfra"), pd.Series(km_ae.labels_, name="skupienie"))
plt.figure(figsize=(7, 5))
plt.imshow(tab.values, cmap="Blues"); plt.colorbar(label="liczba obrazów")
plt.xticks(range(tab.shape[1]), tab.columns); plt.yticks(range(tab.shape[0]), tab.index)
plt.xlabel("numer skupienia (k-means na latencie AE 32)"); plt.ylabel("prawdziwa cyfra")
plt.title("Tablica kontyngencji: cyfra vs skupienie"); plt.show()
# Czystość (purity): odsetek obrazów należących do dominującej cyfry w swoim skupieniu
print(f"purity = {tab.max(axis=0).sum() / tab.values.sum():.3f}")

# Wizualizacja nieliniowa: t-SNE latentu

# ==========================================================================
# CELL 41: 
from sklearn.manifold import TSNE
n_tsne = min(1000 if SZYBKI_TRYB else 3000, len(Xk))
z_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(
    enc32.predict(Xk[:n_tsne], verbose=0))
scatter_2d(z_tsne, yk[:n_tsne], "t-SNE latentu AE 32 (tylko do wizualizacji!)")
plt.show()

# 8. Autoenkoder odszumiający (Denoising AE)

# ==========================================================================
# CELL 43: 
SIGMA = 0.5   # odchylenie standardowe szumu gaussowskiego
rng = np.random.default_rng(SEED)
x_train_noisy = np.clip(x_train_flat + SIGMA * rng.normal(size=x_train_flat.shape), 0, 1).astype("float32")
x_test_noisy = np.clip(x_test_flat + SIGMA * rng.normal(size=x_test_flat.shape), 0, 1).astype("float32")

def psnr(x, x_hat):
    # PSNR w dB dla obrazów w [0, 1], uśredniony po próbkach
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse_per_sample(x, x_hat), 1e-10))))

print(f"PSNR obrazu zaszumionego względem czystego: {psnr(x_test_flat, x_test_noisy):.2f} dB")
pokaz_rekonstrukcje(x_test_flat[:10], x_test_noisy[:10], tytuly=("czysty", f"szum σ={SIGMA}"), suptitle="Dane z szumem")

# Zadanie 8.1 - trening DAE i porównanie

# ==========================================================================
# CELL 45: 
dae, enc_dae, dec_dae = zbuduj_ae_gesty(64, nazwa="dae")
# Cel zawsze czysty, szum jest TYLKO na wejściu. Banalny detal, łatwo popsuć.
hist_dae = dae.fit(x_train_noisy, x_train_flat, epochs=EPOKI,
                    batch_size=BATCH, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                             restore_best_weights=True)], verbose=2)
rek_dae = dae.predict(x_test_noisy, batch_size=BATCH, verbose=0)
rek_ae_noisy = ae32.predict(x_test_noisy, batch_size=BATCH, verbose=0)
psnr_we = psnr(x_test_flat, x_test_noisy)
psnr_dae = psnr(x_test_flat, rek_dae)
psnr_ae_zwykly = psnr(x_test_flat, rek_ae_noisy)
pokaz_rekonstrukcje(x_test_noisy[:10], [rek_ae_noisy[:10], rek_dae[:10], x_test_flat[:10]],
                    tytuly=("zaszumiony", "AE zwykły", "DAE", "czysty"),
                    suptitle="Odszumianie: zwykły AE vs DAE")
print(f"PSNR: wejście {psnr_we:.2f} dB | zwykły AE {psnr_ae_zwykly:.2f} dB | DAE {psnr_dae:.2f} dB")


# 9. Autoenkoder konwolucyjny - "warstwy rozwijające"

# ==========================================================================
# CELL 47: 
# Demonstracja: jak zmieniają się kształty w Conv2D ze stride 2 i Conv2DTranspose ze stride 2
x_demo = np.zeros((1, 28, 28, 1), dtype="float32")
c1 = layers.Conv2D(8, 3, strides=2, padding="same")(x_demo)
c2 = layers.Conv2D(8, 3, strides=2, padding="same")(c1)
t1 = layers.Conv2DTranspose(8, 3, strides=2, padding="same")(c2)
t2 = layers.Conv2DTranspose(1, 3, strides=2, padding="same")(t1)
t_valid = layers.Conv2DTranspose(1, 3, strides=2, padding="valid")(c2)
print("wejście", x_demo.shape, "-> conv", tuple(c1.shape), "-> conv", tuple(c2.shape))
print("-> convT", tuple(t1.shape), "-> convT", tuple(t2.shape))
print("convT padding='valid' z 7x7:", tuple(t_valid.shape), " (wzór: (7-1)*2+3 = 15)")

# **Alternatywa dla splotu transponowanego:** `UpSampling2D(2)` powiela każdy piksel do bloku 2x2 (bez parametrów), a następujący po nim `Conv2D(padding="same")` "wygładza" wynik. Taki dekoder nie ma artefaktów szachownicy, ale jest nieco wolniejszy. Porównaj liczbę parametrów:

# ==========================================================================
# CELL 49: 
# UpSampling2D + Conv2D vs Conv2DTranspose - kształty i liczba parametrów
inp = keras.Input(shape=(7, 7, 64))
m_up = keras.Model(inp, layers.Conv2D(32, 3, padding="same")(layers.UpSampling2D(2)(inp)))
m_tr = keras.Model(inp, layers.Conv2DTranspose(32, 3, strides=2, padding="same")(inp))
print("UpSampling2D+Conv2D :", m_up.output_shape, "parametry:", m_up.count_params())
print("Conv2DTranspose     :", m_tr.output_shape, "parametry:", m_tr.count_params())

# Zadanie 9.1 - autoenkoder konwolucyjny

# ==========================================================================
# CELL 51: 
def zbuduj_ae_conv(latent_dim=32):
    # 28 -> 14 -> 7 pikseli. Wracamy splotem transponowanym 7 -> 14 -> 28.
    wej = keras.Input(shape=(28, 28, 1), name="wej_conv")
    h = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(wej)
    h = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.Flatten()(h)
    z = layers.Dense(latent_dim, name="z_conv")(h)
    enkoder = keras.Model(wej, z, name="conv_encoder")
    wej_z = keras.Input(shape=(latent_dim,), name="wej_z")
    h = layers.Dense(7*7*64, activation="relu")(wej_z)
    h = layers.Reshape((7, 7, 64))(h)
    h = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(h)
    wyj = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(h)
    dekoder = keras.Model(wej_z, wyj, name="conv_decoder")
    wej_ae = keras.Input(shape=(28, 28, 1))
    cae = keras.Model(wej_ae, dekoder(enkoder(wej_ae)), name="cae")
    cae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return cae, enkoder, dekoder

cae, cenc, cdec = zbuduj_ae_conv(32)
print("CAE output:", cae.output_shape)
cdec.summary()
assert cae.output_shape == (None, 28, 28, 1)


# ==========================================================================
# CELL 52: 
hist_cae = cae.fit(x_train_img, x_train_img, epochs=EPOKI, batch_size=BATCH,
                    validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                             restore_best_weights=True)], verbose=2)
rek_cae = cae.predict(x_test_img, batch_size=BATCH, verbose=0)
mse_cae32 = float(mse_per_sample(x_test_img, rek_cae).mean())
pokaz_rekonstrukcje(x_test_flat[:10], [rek_ae32[:10], rek_cae[:10]],
                    tytuly=("oryginał", "AE gęsty", "CAE"),
                    suptitle="Rekonstrukcje CAE vs AE gęsty (d=32)")
print(f"MSE: AE gęsty 32 = {mse_ae32:.5f} ({ae32.count_params():,} param.) | "
      f"CAE 32 = {mse_cae32:.5f} ({cae.count_params():,} param.)")


# 10. Detekcja anomalii błędem rekonstrukcji

# Zadanie 10.1 - AE jako detektor anomalii

# ==========================================================================
# CELL 55: 
ANOMALIA = 9
mask_norm = y_train != ANOMALIA
x_norm_cale = x_train_flat[mask_norm]
rng_a = np.random.default_rng(SEED)
indeksy = rng_a.permutation(len(x_norm_cale))
n_val_anom = max(1, round(len(indeksy) * 0.10))
x_norm_val = x_norm_cale[indeksy[:n_val_anom]]
x_norm = x_norm_cale[indeksy[n_val_anom:]]
ae_anom, _, _ = zbuduj_ae_gesty(16, nazwa="ae_anom_9")
# Uwaga na leakage: sam próg wybieramy WYŁĄCZNIE na walidacyjnych 0..8.
ae_anom.fit(x_norm, x_norm, epochs=EPOKI, batch_size=BATCH,
            validation_data=(x_norm_val, x_norm_val),
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                     restore_best_weights=True)], verbose=2)
err_val = mse_per_sample(x_norm_val, ae_anom.predict(x_norm_val, batch_size=BATCH, verbose=0))
prog_anom = float(np.percentile(err_val, 95))
err_test = mse_per_sample(x_test_flat, ae_anom.predict(x_test_flat, batch_size=BATCH, verbose=0))
y_anom = (y_test == ANOMALIA).astype(int)
auc_anom = float(roc_auc_score(y_anom, err_test))
pred_anom = (err_test >= prog_anom).astype(int)
anom_precision = float(precision_score(y_anom, pred_anom, zero_division=0))
anom_recall = float(recall_score(y_anom, pred_anom, zero_division=0))
anom_f1 = float(f1_score(y_anom, pred_anom, zero_division=0))
fpr, tpr, _ = roc_curve(y_anom, err_test)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(err_test[y_anom == 0], bins=45, density=True, alpha=0.6, label="cyfry 0..8")
axes[0].hist(err_test[y_anom == 1], bins=45, density=True, alpha=0.6, label="cyfra 9")
axes[0].axvline(prog_anom, color="red", ls="--", label="95. percentyl (walidacja)")
axes[0].set_xlabel("MSE rekonstrukcji")
axes[0].set_ylabel("gęstość")
axes[0].set_title("Rozkład błędu: normalne vs anomalie")
axes[0].legend(fontsize=8)
axes[1].plot(fpr, tpr, label=f"AUC={auc_anom:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].set_xlabel("FPR")
axes[1].set_ylabel("TPR")
axes[1].set_title("Detekcja cyfry 9 - ROC")
axes[1].legend()
fig.suptitle("Anomalie wykrywane błędem rekonstrukcji AE")
plt.tight_layout(); plt.show()
print(f"AUC={auc_anom:.4f}, próg={prog_anom:.5f}, precision={anom_precision:.3f}, "
      f"recall={anom_recall:.3f}, F1={anom_f1:.3f}")


# Obrazy "najbardziej nietypowe" według AE

# ==========================================================================
# CELL 57: 
if auc_anom is not None:
    idx = np.argsort(err_test)[::-1][:10]                     # indeksy największych błędów
    fig, axes = plt.subplots(1, 10, figsize=(13, 1.8))
    for ax, i in zip(axes, idx):
        ax.imshow(x_test[i], cmap="gray"); ax.axis("off")
        ax.set_title(f"y={y_test[i]}\ne={err_test[i]:.3f}", fontsize=8)
    fig.suptitle("Najbardziej nietypowe obrazy testowe (największy błąd rekonstrukcji)")
    plt.tight_layout(); plt.show()

# Pytania kontrolne

# Podsumowanie - co warto zapamiętać

# Do sprawozdania końcowego

# Zadanie dodatkowe (dla chętnych) - autoenkoder rzadki i porównanie klas anomalii

# ==========================================================================
# CELL 62: 
wyniki_rzadkie, wyniki_anom = [], []
if BONUS:
    for lam in (0.0, 1e-4):
        # Szeroki latent 128. Kara L1 jest na AKTYWACJACH, nie na wagach.
        keras.utils.set_random_seed(SEED)
        reg = keras.regularizers.L1(lam) if lam > 0 else None
        sparse, sparse_enc, _ = zbuduj_ae_gesty(
            128, ukryte=(256, 128), aktywacja_latentu="relu",
            nazwa=f"sparse_{lam:g}", regularizer_latentu=reg)
        sparse.fit(x_train_flat, x_train_flat, epochs=EPOKI_KROTKO,
                   batch_size=BATCH, validation_split=0.1, verbose=0)
        rek = sparse.predict(x_test_flat, batch_size=BATCH, verbose=0)
        z_mean = sparse_enc.predict(x_test_flat.mean(axis=0, keepdims=True), verbose=0)
        z_wiele = sparse_enc.predict(x_test_flat, batch_size=BATCH, verbose=0)
        wyniki_rzadkie.append({"lambda": lam,
            "aktywny_latent_sredni_obraz": float(np.mean(z_mean > 0.01)),
            "aktywny_latent_srednio": float(np.mean(z_wiele > 0.01)),
            "mse": float(mse_per_sample(x_test_flat, rek).mean())})
        del sparse, sparse_enc
        gc.collect()
    for digit in (0, 1, 9):
        # Porównanie anomalii: ten sam split i kryterium dla każdej cyfry.
        norm_all = x_train_flat[y_train != digit]
        idx_b = np.random.default_rng(SEED).permutation(len(norm_all))
        val_n = max(1, round(len(idx_b) * 0.1))
        xb_val = norm_all[idx_b[:val_n]]
        xb_train = norm_all[idx_b[val_n:]]
        ae_b, _, _ = zbuduj_ae_gesty(16, nazwa=f"anom_bonus_{digit}")
        ae_b.fit(xb_train, xb_train, epochs=EPOKI_KROTKO,
                 batch_size=BATCH, validation_data=(xb_val, xb_val), verbose=0)
        err_b = mse_per_sample(x_test_flat,
                ae_b.predict(x_test_flat, batch_size=BATCH, verbose=0))
        y_b = (y_test == digit).astype(int)
        auc_b = float(roc_auc_score(y_b, err_b))
        wyniki_anom.append({"cyfra_anomalia": digit, "roc_auc": auc_b})
        del ae_b
        gc.collect()
    display(pd.DataFrame(wyniki_rzadkie).round(5))
    display(pd.DataFrame(wyniki_anom).round(4))
    df_bonus_an = pd.DataFrame(wyniki_anom)
    ax = df_bonus_an.plot.bar(x="cyfra_anomalia", y="roc_auc", figsize=(6, 4), rot=0, legend=False)
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Bonus: która cyfra daje wykrywalne anomalie?")
    plt.tight_layout(); plt.show()
else:
    print("Bonus pominięty. Aby wykonać: python lab06_rozwiazanie.py --full --bonus")


# Eksport wyników lokalnych

# ==========================================================================
# CELL 64: 
# Eksport spójny dla notebooka i wersji .py - niczego nie wpisujemy na sztywno.
df_d.to_csv(OUT_DIR / "porownanie_ae_pca.csv", index=False, encoding="utf-8-sig")
df_klast.to_csv(OUT_DIR / "klasteryzacja.csv", index=False, encoding="utf-8-sig")
if BONUS:
    pd.DataFrame(wyniki_rzadkie).to_csv(OUT_DIR / "bonus_sparse.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(wyniki_anom).to_csv(OUT_DIR / "bonus_anomalie.csv", index=False, encoding="utf-8-sig")

def _czyste(v):
    if isinstance(v, dict):
        return {str(k): _czyste(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_czyste(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return _czyste(v.tolist())
    return v

PODSUMOWANIE = {
    "profil": DL_PROFILE, "bonus_wykonany": BONUS,
    "czas_eksperymentu_sekundy": round(time.perf_counter() - CZAS_START, 2),
    "liczba_probek_train": len(x_train_flat), "liczba_probek_test": len(x_test_flat),
    "epoki_glowne_limit": EPOKI, "epoki_krotkie_limit": EPOKI_KROTKO,
    "pca_progi": {f"{int(p*100)}%": k for p, k in k_progi.items()},
    "pca_mse": mse_pca_k,
    "rekonstrukcja_d32": {"mse_pca": mse_pca32, "mse_ae": mse_ae32,
                            "mse_cae": mse_cae32,
                            "parametry_ae": ae32.count_params(),
                            "parametry_cae": cae.count_params()},
    "hipoteza_ae_lepszy_d": bool((df_d.mse_ae < df_d.mse_pca).all()),
    "mse_w_funkcji_d": wyniki_d,
    "liniowy_ae": {"mse_ae": mse_lin_ae, "mse_pca16": mse_pca16,
                    "max_kat_stopnie": float(np.max(katy))},
    "klasteryzacja": df_klast.to_dict(orient="records"),
    "odszumianie_psnr_db": {"wejscie": psnr_we, "ae_zwykly": psnr_ae_zwykly, "dae": psnr_dae},
    "anomalia": {"cyfra": ANOMALIA, "roc_auc": auc_anom, "prog_95p": prog_anom,
                   "precision": anom_precision, "recall": anom_recall, "f1": anom_f1,
                   "liczba_anomalii_test": int(y_anom.sum())},
    "bonus_sparse": wyniki_rzadkie, "bonus_anomalie": wyniki_anom,
    "wykresy": PLIKOWANE_WYKRESY,
}
(OUT_DIR / "podsumowanie.json").write_text(
    json.dumps(_czyste(PODSUMOWANIE), ensure_ascii=False, indent=2), encoding="utf-8")
wnioski = [
    f"LAB06 - PROFIL {DL_PROFILE.upper()} | BONUS: {BONUS}",
    f"Train={N_TRAIN}, test={N_TEST}; wyniki są zależne od profilu i liczby epok.",
    f"PCA: k90={k_progi[0.9]}, k95={k_progi[0.95]}; 2 wymiary wyjaśniają {cum[1]:.1%} wariancji.",
    f"AE(32) MSE={mse_ae32:.5f}; PCA(32) MSE={mse_pca32:.5f}; CAE(32) MSE={mse_cae32:.5f}.",
    "Hipoteza AE lepszy dla wszystkich d: " +
        ("potwierdzona na zmierzonych d" if PODSUMOWANIE["hipoteza_ae_lepszy_d"] else
         "NIE potwierdzona na wszystkich d; interpretować z uwzględnieniem treningu."),
    f"Liniowy AE MSE={mse_lin_ae:.5f}, PCA(16) MSE={mse_pca16:.5f}, max kąt={max(katy):.1f} st.",
    f"Najlepsze ARI: {df_klast.iloc[0]['przestrzeń']} = {df_klast.iloc[0]['ARI']:.3f}.",
    f"Odszumianie: wejście={psnr_we:.2f}, zwykły AE={psnr_ae_zwykly:.2f}, DAE={psnr_dae:.2f} dB.",
    f"Anomalia {ANOMALIA}: ROC-AUC={auc_anom:.3f}, precision={anom_precision:.3f}, "
        f"recall={anom_recall:.3f}, F1={anom_f1:.3f}.",
    "Uwaga: LDA korzysta z etykiet; PCA/SVD/AE nie. Nie porównywać silhouette między wymiarami wprost.",
    "Próg anomalii ustalony na 95. percentylu błędów walidacji samych danych normalnych.",
    "Propozycja usprawnienia do sprawozdania: sprawdzić CAE zamiast AE gęstego "
        "dla detekcji anomalii na tym samym podziale; obecnie CAE porównano rekonstrukcyjnie.",
]
if SZYBKI_TRYB:
    wnioski.append("UWAGA: FAST to test techniczny - wyników nie raportować jako FULL.")
(OUT_DIR / "wnioski.txt").write_text("\n".join(wnioski)+"\n", encoding="utf-8")
print("Zapisano:", OUT_DIR / "podsumowanie.json", "| wykresów:", len(PLIKOWANE_WYKRESY))

