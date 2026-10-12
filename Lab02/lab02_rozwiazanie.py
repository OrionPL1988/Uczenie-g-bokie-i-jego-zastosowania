# -*- coding: utf-8 -*-
"""Lab02 - Perceptron / MLP. Rozwiązanie z notebooka prowadzącej.

Uruchom: python lab02_rozwiazanie.py             (pełny eksperyment)
         python lab02_rozwiazanie.py --fast      (krótki test)

Kompatybilne z Windows / Python 3.11, CPU wystarcza. Pliki trafiają do wyniki/.
"""
import os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("MPLBACKEND", "Agg")  # Bez 13 okien blokujących terminal.
if "--fast" in sys.argv:
    os.environ["DL_FAST"] = "1"
if "--full" in sys.argv:
    os.environ["DL_FAST"] = "0"


# Lab 2 - Perceptron i MLP (rozwiązanie lokalne, VS Code)


# 0. Uruchomienie lokalne w Visual Studio Code


# ---------------- Komórka 3 z notebooka ----------------
# Lokalny Windows / VS Code: karta NVIDIA jest opcjonalna, bez Colaba.
# Jeżeli nvidia-smi jest dostępne, wyświetlam stan sterownika. W przeciwnym razie spokojnie używam CPU.
import shutil, subprocess
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("nvidia-smi nie znaleziony - to nie blokuje tego laboratorium.")



# 0.2 Importy, tryb szybki, seedy


# ---------------- Komórka 5 z notebooka ----------------
import os, time, math
from pathlib import Path
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Działa także bez okna wykresów (uruchomienie zwykłego .py).
import matplotlib.pyplot as plt
from scipy.special import erf

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import tensorflow as tf
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)   # TF bierze pamięć GPU stopniowo
import keras
from keras import layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split

# Tryb szybki: True = mniej danych i epok (praca bez GPU)
SZYBKI_TRYB = os.environ.get("DL_FAST", "0") == "1"
# SZYBKI_TRYB = True   # <- odkomentuj, jeśli pracujesz bez GPU

SEED = 42
keras.utils.set_random_seed(SEED)      # Python + NumPy + TF
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("SZYBKI_TRYB =", SZYBKI_TRYB)
print("TensorFlow", tf.__version__, "| Keras", keras.__version__, "| backend:", keras.backend.backend())
print("PyTorch", torch.__version__, "| DEVICE =", DEVICE)
print("GPU (TF):", tf.config.list_physical_devices("GPU"))
# Zapisuję wszystko do podfolderu przy projekcie, a nie gdzieś do katalogu użytkownika.
KATALOG_WYNIKOW = Path("wyniki")
KATALOG_WYKRESOW = KATALOG_WYNIKOW / "wykresy"
KATALOG_WYKRESOW.mkdir(parents=True, exist_ok=True)
WYKRESY = []
WYNIKI = {"tryb": "szybki" if SZYBKI_TRYB else "pełny", "seed": SEED,
          "python": platform.python_version(), "numpy": np.__version__,
          "tensorflow": tf.__version__, "keras": keras.__version__,
          "pytorch": torch.__version__, "device": str(DEVICE)}
OSTRZEZENIA = []

def sprawdz(warunek, opis):
    # Kontrola ma sens. Nie chcę sytuacji jak kiedyś z MSE=nan, a w raporcie wszystko OK.
    if not bool(warunek):
        raise AssertionError("Nieudany test: " + opis)
    print("[OK]", opis)

def uwaga(warunek, opis):
    # Gdy kryterium jakościowe zależy od losowości, ostrzegam zamiast udawać sukces.
    if not bool(warunek):
        OSTRZEZENIA.append(opis)
        print("[UWAGA]", opis)

def zapisz_wykres(nazwa):
    # Wykresy zapisuję osobno, żeby można je było od razu oglądać na GitHubie.
    sciezka = KATALOG_WYKRESOW / nazwa
    plt.savefig(sciezka, dpi=145, bbox_inches="tight")
    plt.close('all')
    WYKRESY.append(str(sciezka).replace("\\", "/"))
    print("[WYKRES]", sciezka)



# 0.3 Funkcja pomocnicza: rysowanie granicy decyzyjnej


# ---------------- Komórka 7 z notebooka ----------------
def rysuj_granice(predict_fn, X, y, ax=None, tytul="", rozdz=200, wieloklasowa=False):
    '''Rysuje granicę decyzyjną klasyfikatora 2D i punkty danych.'''
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    marg = 0.5
    x_min, x_max = X[:, 0].min() - marg, X[:, 0].max() + marg
    y_min, y_max = X[:, 1].min() - marg, X[:, 1].max() + marg
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, rozdz), np.linspace(y_min, y_max, rozdz))
    siatka = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)   # (rozdz^2, 2) punktów
    Z = np.asarray(predict_fn(siatka), dtype=float).reshape(xx.shape)
    if wieloklasowa:
        k = int(max(Z.max(), y.max())) + 1
        ax.contourf(xx, yy, Z, levels=np.arange(k + 1) - 0.5, cmap="tab10", vmin=0, vmax=9, alpha=0.3)
        kolory = plt.cm.tab10(np.arange(k))
    else:
        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.45)
        if Z.min() < 0.5 < Z.max():                       # granica p = 0.5
            ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)
        kolory = ["tab:blue", "tab:red"]
    for kl in np.unique(y):                               # punkty każdej klasy osobno -> legenda
        m = y == kl
        ax.scatter(X[m, 0], X[m, 1], color=kolory[int(kl)], edgecolors="k", s=18, label=f"klasa {int(kl)}")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_title(tytul); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=8, loc="best")
    return ax


# 1. Perceptron Rosenblatta


# ---------------- Komórka 9 z notebooka ----------------
# Dane liniowo separowalne: dwie dobrze rozdzielone chmury punktów
X_lin, y_lin = make_blobs(n_samples=200, centers=[[-2, -2], [2, 2]], cluster_std=0.9, random_state=SEED)
print("X:", X_lin.shape, " klasy:", np.bincount(y_lin))

plt.figure(figsize=(5.5, 4.5))
for kl, kol in zip([0, 1], ["tab:blue", "tab:red"]):
    plt.scatter(X_lin[y_lin == kl, 0], X_lin[y_lin == kl, 1], color=kol, edgecolors="k", s=18, label=f"klasa {kl}")
plt.title("Dane liniowo separowalne"); plt.xlabel("$x_1$"); plt.ylabel("$x_2$"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("01_dane_liniowe.png")



# Zadanie 1.1 - Perceptron od zera w NumPy


# ---------------- Komórka 11 z notebooka ----------------
class PerceptronRosenblatta:
    def __init__(self, n_wejsc, eta=0.1, seed=0):
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(scale=0.01, size=n_wejsc)
        self.b = 0.0
        self.eta = eta
        self.bledy = []

    def net(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        # Próg jest tutaj twardy: albo klasa 1, albo 0. Przy XOR właśnie to jest ograniczeniem.
        return (np.asarray(self.net(X)) >= 0).astype(int)

    def fit(self, X, y, epoki=20):
        for ep in range(epoki):
            bledy = 0
            for i in self.rng.permutation(len(X)):
                xi, yi = X[i], y[i]
                y_hat = int(self.net(xi) >= 0)
                blad = int(yi) - y_hat
                if blad != 0:
                    # Jak źle ustawiona geometria w Fiacie: korekta ma sens tylko, gdy jest błąd.
                    self.w += self.eta * blad * xi
                    self.b += self.eta * blad
                    bledy += 1
            self.bledy.append(bledy)
            if bledy == 0:
                break
        return self



# ---------------- Komórka 12 z notebooka ----------------
perc = PerceptronRosenblatta(n_wejsc=2, eta=0.1).fit(X_lin, y_lin, epoki=50)
acc = np.mean(perc.predict(X_lin) == y_lin)
print(f"epok: {len(perc.bledy)}, błędy w epokach: {perc.bledy}")
print(f"dokładność: {acc:.3f}, w = {np.round(perc.w, 3)}, b = {perc.b:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(range(1, len(perc.bledy) + 1), perc.bledy, "o-", label="błędne aktualizacje")
ax[0].set_title("Perceptron: liczba błędów w epoce"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("liczba błędów")
ax[0].grid(alpha=0.3); ax[0].legend()
rysuj_granice(perc.predict, X_lin, y_lin, ax=ax[1], tytul=f"Granica perceptronu (acc = {acc:.2f})")
plt.tight_layout(); zapisz_wykres("02_perceptron_liniowy.png")



# ---------------- Komórka 13 z notebooka ----------------
sprawdz(acc >= 0.999, "Perceptron na liniowo separowalnych danych osiąga 100%")
WYNIKI["perceptron_liniowy"] = {"accuracy": float(acc), "epoki": len(perc.bledy)}


# Zadanie 1.2 - Porażka perceptronu na XOR


# ---------------- Komórka 15 z notebooka ----------------
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
rng = np.random.default_rng(SEED)
X_xor_szum = np.repeat(X_xor, 100, axis=0) + rng.normal(scale=0.15, size=(400, 2))
y_xor_szum = np.repeat(y_xor, 100)

# Tutaj nie poprawiam na siłę dokładności. Eksperyment ma pokazać granicę możliwości perceptronu.
perc_xor = PerceptronRosenblatta(n_wejsc=2, eta=0.1, seed=0).fit(X_xor, y_xor, epoki=100)
perc_xor_szum = PerceptronRosenblatta(n_wejsc=2, eta=0.1, seed=0).fit(X_xor_szum, y_xor_szum, epoki=100)



# ---------------- Komórka 16 z notebooka ----------------
if perc_xor is not None and perc_xor_szum is not None:
    acc4 = np.mean(perc_xor.predict(X_xor) == y_xor)
    acc_s = np.mean(perc_xor_szum.predict(X_xor_szum) == y_xor_szum)
    print(f"XOR (4 punkty): dokładność {acc4:.2f}, błędy w ostatnich epokach: {perc_xor.bledy[-5:]}")
    print(f"XOR zaszumiony: dokładność {acc_s:.2f}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(perc_xor.bledy, label="XOR (4 pkt)")
    ax[0].plot(np.array(perc_xor_szum.bledy) / 100, label="XOR zaszumiony (błędy / 100)")
    ax[0].set_title("Perceptron na XOR: błędy nie znikają"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("liczba błędów")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    rysuj_granice(perc_xor.predict, X_xor, y_xor, ax=ax[1], tytul=f"XOR, 4 punkty (acc = {acc4:.2f})")
    rysuj_granice(perc_xor_szum.predict, X_xor_szum, y_xor_szum, ax=ax[2], tytul=f"XOR zaszumiony (acc = {acc_s:.2f})")
    plt.tight_layout(); zapisz_wykres("03_perceptron_xor.png")



# ---------------- Komórka 17 z notebooka ----------------
sprawdz(acc4 <= 0.75 + 1e-12, "Perceptron nie rozwiązuje XOR (maksymalnie 75%)")
WYNIKI["perceptron_xor"] = {"accuracy": float(acc4), "accuracy_noisy":float(acc_s), "bledy_last": perc_xor.bledy[-5:]}


# 2. Funkcje aktywacji i ich pochodne


# ---------------- Komórka 19 z notebooka ----------------
# Implementacje funkcji aktywacji w NumPy
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0.0, z)

def leaky_relu(z, alfa=0.1):
    return np.where(z > 0, z, alfa * z)

def gelu(z):
    return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))       # z * Phi(z), wersja dokładna

AKTYWACJE = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": relu, "LeakyReLU (0.1)": leaky_relu, "GELU": gelu}

z = np.linspace(-5, 5, 401)
plt.figure(figsize=(8, 5))
for nazwa, f in AKTYWACJE.items():
    plt.plot(z, f(z), label=nazwa, lw=2)
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.ylim(-1.5, 3); plt.title("Funkcje aktywacji"); plt.xlabel("z"); plt.ylabel("f(z)"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("04_funkcje_aktywacji.png")



# 2.2 Softmax i stabilność numeryczna


# ---------------- Komórka 21 z notebooka ----------------
def softmax_naiwny(z):
    e = np.exp(z)
    return e / e.sum()

def softmax_stabilny(z):
    e = np.exp(z - np.max(z))            # przesunięcie nie zmienia wyniku, a chroni przed overflow
    return e / e.sum()

logity = np.array([2.0, 1.0, 0.1])
print("softmax([2, 1, 0.1])       :", np.round(softmax_stabilny(logity), 4), " suma =", softmax_stabilny(logity).sum())
duze = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("naiwny dla [1000,1001,1002]:", softmax_naiwny(duze))
print("stabilny                   :", np.round(softmax_stabilny(duze), 4))
print("PyTorch                    :", torch.softmax(torch.tensor(duze), dim=0).numpy().round(4))


# Zadanie 2.1 - Pochodne funkcji aktywacji: ręcznie vs autograd


# ---------------- Komórka 23 z notebooka ----------------
# Pochodne piszę ręcznie, żeby porównać je z automatycznym różniczkowaniem PyTorch.
def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2

def d_relu(z):
    return np.where(z > 0.0, 1.0, 0.0)

def d_leaky_relu(z, alfa=0.1):
    return np.where(z > 0.0, 1.0, alfa)

def d_gelu(z):
    # Dokładna GELU: Phi(z) + z * phi(z), bez przybliżenia tanh.
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    return Phi + z * phi



# ---------------- Komórka 24 z notebooka ----------------
# Komórka sprawdzająca: autograd w PyTorch (float64)
z_t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
torch_akt = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "ReLU": torch.relu,
             "LeakyReLU (0.1)": lambda t: F.leaky_relu(t, 0.1), "GELU": F.gelu}
reczne = {"sigmoid": d_sigmoid, "tanh": d_tanh, "ReLU": d_relu, "LeakyReLU (0.1)": d_leaky_relu, "GELU": d_gelu}

roznice_pochodnych = {}
if d_sigmoid is not None:
    plt.figure(figsize=(8, 5))
    for nazwa, f_t in torch_akt.items():
        z_t.grad = None
        f_t(z_t).sum().backward()                      # suma -> gradient elementowy
        g_auto = z_t.grad.numpy()
        g_rec = reczne[nazwa](z)
        roznice_pochodnych[nazwa] = float(np.abs(g_rec - g_auto).max())
        print(f"{nazwa:>16}: max |ręczna - autograd| = {roznice_pochodnych[nazwa]:.2e}")
        plt.plot(z, g_rec, lw=2, label=f"{nazwa}'")
    plt.axhline(0.25, color="gray", ls=":", label="0.25 = max pochodnej sigmoidu")
    plt.title("Pochodne funkcji aktywacji"); plt.xlabel("z"); plt.ylabel("f'(z)"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("05_pochodne_autograd.png")



# ---------------- Komórka 25 z notebooka ----------------
# Kontroluję największą różnicę pochodnych, nie tylko wygląd wykresu.
WYNIKI["pochodne_autograd"] = roznice_pochodnych
sprawdz(all(w < 1e-6 for w in roznice_pochodnych.values()), "Pochodne NumPy i PyTorch zgodne (<1e-6)")


# 2.3 Znikający gradient


# ---------------- Komórka 27 z notebooka ----------------
def zbuduj_gleboka_siec(aktywacja, L=20, szer=64, seed=0):
    torch.manual_seed(seed)
    akt = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}[aktywacja]
    warstwy = []
    for _ in range(L):
        lin = nn.Linear(szer, szer)
        if aktywacja == "relu":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # inicjalizacja He
        else:
            nn.init.xavier_normal_(lin.weight)                         # inicjalizacja Xavier/Glorot
        nn.init.zeros_(lin.bias)
        warstwy += [lin, akt()]
    warstwy.append(nn.Linear(szer, 1))                                  # wyjście
    return nn.Sequential(*warstwy)

torch.manual_seed(SEED)
x_batch = torch.randn(256, 64)
y_batch = torch.randn(256, 1)

normy = {}
for aktywacja in ["sigmoid", "tanh", "relu"]:
    siec = zbuduj_gleboka_siec(aktywacja, L=20, szer=64, seed=SEED)
    prognoza = siec(x_batch)
    strata = F.mse_loss(prognoza, y_batch)
    siec.zero_grad(set_to_none=True)
    strata.backward()
    # Kolejność z model.modules() odpowiada kolejności od wejścia do wyjścia.
    normy[aktywacja] = [warstwa.weight.grad.norm().item()
                        for warstwa in siec.modules() if isinstance(warstwa, nn.Linear)]

if normy:
    plt.figure(figsize=(8, 4.5))
    for akt_nazwa, nr in normy.items():
        plt.semilogy(range(1, len(nr) + 1), nr, "o-", label=akt_nazwa)
        print(f"{akt_nazwa:>8}: |grad| warstwa 1 = {nr[0]:.2e}, ostatnia = {nr[-1]:.2e}, stosunek = {nr[0] / nr[-1]:.1e}")
    plt.title("Norma gradientu wag w 20-warstwowej sieci (warstwa 1 = przy wejściu)")
    plt.xlabel("numer warstwy"); plt.ylabel("||dL/dW|| (skala log)"); plt.legend(); plt.grid(alpha=0.3, which="both"); zapisz_wykres("06_zanik_gradientu.png")



# ---------------- Komórka 28 z notebooka ----------------
WYNIKI["normy_gradientow"] = {k: {"pierwsza":float(v[0]), "ostatnia":float(v[-1])} for k,v in normy.items()}


# 3. MLP od zera z ręczną propagacją wsteczną (XOR)


# ---------------- Komórka 30 z notebooka ----------------
X_x = X_xor.copy()                       # (4, 2)
Y_x = y_xor.reshape(-1, 1).astype(float) # (4, 1) - kolumna, żeby kształty pasowały do A2

def init_params(n_wej=2, n_ukr=4, n_wyj=1, seed=1):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 1.0, (n_wej, n_ukr)), "b1": np.zeros(n_ukr),
            "W2": rng.normal(0, 1.0, (n_ukr, n_wyj)), "b2": np.zeros(n_wyj)}

def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]      # (n, n_ukr)
    A1 = np.tanh(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]     # (n, 1)
    A2 = sigmoid(Z2)
    cache = {"X": X, "A1": A1, "A2": A2}     # zapamiętujemy wartości potrzebne w backward
    return A2, cache

def bce(A2, Y, eps=1e-12):
    # eps chroni przed log(0)
    return float(-np.mean(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps)))

params = init_params()
A2, _ = forward(params, X_x)
print("predykcje przed treningiem:", A2.ravel().round(3), " strata BCE =", round(bce(A2, Y_x), 4))
print("liczba parametrów:", sum(v.size for v in params.values()))


# Zadanie 3.1 - Implementacja `backward` i trening MLP na XOR


# ---------------- Komórka 32 z notebooka ----------------
def backward(params, cache, Y):
    X, A1, A2 = cache["X"], cache["A1"], cache["A2"]
    n = Y.shape[0]
    # Cała propagacja wsteczna z reguły łańcuchowej. Przy sigmoid+BCE wychodzi A2-Y.
    delta2 = (A2 - Y) / n
    dW2 = A1.T @ delta2
    db2 = delta2.sum(axis=0)
    delta1 = (delta2 @ params["W2"].T) * (1.0 - A1 ** 2)
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}



# ---------------- Komórka 33 z notebooka ----------------
EPOKI_XOR = 2000 if SZYBKI_TRYB else 5000
lr_xor = 1.0
params = init_params()
hist_xor = []
for ep in range(EPOKI_XOR):
    A2, cache = forward(params, X_x)           # 1) forward
    hist_xor.append(bce(A2, Y_x))              # 2) strata
    grads = backward(params, cache, Y_x)       # 3) backward (ręcznie!)
    for k in params:                           # 4) krok spadku gradientu
        params[k] -= lr_xor * grads[k]

A2, _ = forward(params, X_x)
print("predykcje po treningu:", A2.ravel().round(3), "->", (A2.ravel() > 0.5).astype(int), " (cel: [0 1 1 0])")
print("końcowa strata BCE:", round(hist_xor[-1], 5))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(hist_xor, label="BCE train"); ax[0].set_yscale("log")
ax[0].set_title("MLP od zera na XOR: krzywa uczenia"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("BCE (log)")
ax[0].legend(); ax[0].grid(alpha=0.3)
rysuj_granice(lambda P: forward(params, P)[0].ravel(), X_xor_szum, y_xor_szum, ax=ax[1],
              tytul="Granica MLP 2-4-1 (punkty: zaszumiony XOR)")
plt.tight_layout(); zapisz_wykres("07_mlp_xor.png")



# ---------------- Komórka 34 z notebooka ----------------
sprawdz(hist_xor[-1] < 0.05 and np.array_equal((A2.ravel() > 0.5).astype(int), y_xor), "MLP NumPy rozwiązuje XOR (BCE <0.05)")
WYNIKI["mlp_xor"] = {"BCE": float(hist_xor[-1]), "pred": A2.ravel().tolist()}


# Zadanie 3.2 - Gradient check ręcznego backpropu


# ---------------- Komórka 36 z notebooka ----------------
def gradient_check(params, X, Y, h=1e-5):
    _, cache = forward(params, X)
    g_bp = backward(params, cache, Y)
    bledy = {}
    for nazwa, p in params.items():
        g_num = np.zeros_like(p)
        for indeks in np.ndindex(p.shape):
            # Dwie kopie parametrów, bo sprawdzanie gradientu nie może zmieniać modelu.
            p_plus = {k: v.copy() for k, v in params.items()}
            p_minus = {k: v.copy() for k, v in params.items()}
            p_plus[nazwa][indeks] += h
            p_minus[nazwa][indeks] -= h
            l_plus = bce(forward(p_plus, X)[0], Y)
            l_minus = bce(forward(p_minus, X)[0], Y)
            g_num[indeks] = (l_plus - l_minus) / (2.0 * h)
        mianownik = np.linalg.norm(g_num) + np.linalg.norm(g_bp[nazwa])
        bledy[nazwa] = float(np.linalg.norm(g_num - g_bp[nazwa]) / max(mianownik, 1e-15))
    return bledy

p_test = init_params(seed=7)
for nazwa, blad in gradient_check(p_test, X_x, Y_x).items():
    print(f"{nazwa}: błąd względny = {blad:.2e}  {'OK' if blad < 1e-7 else 'SPRAWDŹ!'}")
    sprawdz(blad < 1e-7, "gradient check dla " + nazwa)



# ---------------- Komórka 37 z notebooka ----------------
WYNIKI["gradient_check"] = gradient_check(p_test, X_x, Y_x)


# 4. MLP w Keras


# ---------------- Komórka 39 z notebooka ----------------
N_PROBEK = 300 if SZYBKI_TRYB else 1000
EPOKI_KERAS = 5 if SZYBKI_TRYB else 150

X_m, y_m = make_moons(n_samples=N_PROBEK, noise=0.2, random_state=SEED)
X_m = X_m.astype(np.float32)
Xm_tr, Xm_va, ym_tr, ym_va = train_test_split(X_m, y_m, test_size=0.25, random_state=SEED, stratify=y_m)
print("train:", Xm_tr.shape, " walidacja:", Xm_va.shape, " proporcje klas (train):", np.bincount(ym_tr))

plt.figure(figsize=(5.5, 4.5))
for kl, kol in zip([0, 1], ["tab:blue", "tab:red"]):
    plt.scatter(Xm_tr[ym_tr == kl, 0], Xm_tr[ym_tr == kl, 1], color=kol, edgecolors="k", s=15, label=f"klasa {kl}")
plt.title("make_moons (noise = 0.2), zbiór treningowy"); plt.xlabel("$x_1$"); plt.ylabel("$x_2$"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("08_dane_moons.png")



# 4.2 Sequential API


# ---------------- Komórka 41 z notebooka ----------------
model_seq = keras.Sequential([
    keras.Input(shape=(2,)),                              # próbka = 2 cechy
    layers.Dense(16, activation="relu", name="ukryta_1"),
    layers.Dense(16, activation="relu", name="ukryta_2"),
    layers.Dense(1, activation="sigmoid", name="wyjscie"),  # p(klasa = 1)
], name="mlp_sequential")
model_seq.summary()


# ---------------- Komórka 42 z notebooka ----------------
model_seq.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

t0 = time.perf_counter()
hist_seq = model_seq.fit(Xm_tr, ym_tr, validation_data=(Xm_va, ym_va),
                         epochs=EPOKI_KERAS, batch_size=32, verbose=0)   # verbose=0: bez paska postępu
print(f"czas treningu: {time.perf_counter() - t0:.1f} s")
strata_va, acc_va = model_seq.evaluate(Xm_va, ym_va, verbose=0)
print(f"walidacja: strata = {strata_va:.4f}, dokładność = {acc_va:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
ax[0].plot(hist_seq.history["loss"], label="train"); ax[0].plot(hist_seq.history["val_loss"], label="walidacja")
ax[0].set_title("Krzywe uczenia - strata BCE"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("BCE"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(hist_seq.history["accuracy"], label="train"); ax[1].plot(hist_seq.history["val_accuracy"], label="walidacja")
ax[1].set_title("Krzywe uczenia - dokładność"); ax[1].set_xlabel("epoka"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
rysuj_granice(lambda P: model_seq.predict(P, verbose=0, batch_size=4096).ravel(), Xm_va, ym_va, ax=ax[2],
              tytul=f"Keras Sequential (val acc = {acc_va:.3f})")
plt.tight_layout(); zapisz_wykres("09_keras_sequential.png")



# ---------------- Komórka 43 z notebooka ----------------
WYNIKI["keras_sequential"] = {"accuracy_val":float(acc_va),"loss_val":float(strata_va), "parametry":int(model_seq.count_params())}
if not SZYBKI_TRYB:
    uwaga(acc_va >= 0.90, "Keras Sequential: accuracy val poniżej 0.90")


# Zadanie 4.1 - Ręczne liczenie liczby parametrów


# ---------------- Komórka 46 z notebooka ----------------
def liczba_parametrow(rozmiary):
    # Każdy neuron ma wagę do każdego wejścia plus swój bias.
    # Na wszelki wypadek liczę to też ręcznie - przy Mateuszu kręcącym się przy biurku
    # łatwo byłoby pomylić liczbę wejść z liczbą neuronów w warstwie.
    na_warstwe = [(a + 1) * b for a, b in zip(rozmiary[:-1], rozmiary[1:])]
    return na_warstwe, sum(na_warstwe)

model_duzy = keras.Sequential([
    keras.Input(shape=(2,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
], name="mlp_duzy")

for model, rozmiary in [(model_seq, [2, 16, 16, 1]), (model_duzy, [2, 64, 32, 8, 1])]:
    rek, razem = liczba_parametrow(rozmiary)
    biblioteka = [int(w.count_params()) for w in model.layers]
    print(model.name, "ręcznie:", rek, razem, "Keras:", biblioteka, model.count_params())
    sprawdz(rek == biblioteka and razem == model.count_params(), "liczba parametrów " + model.name)

print("Dense(512) na obrazie 28x28:", liczba_parametrow([28 * 28, 512])[1], "parametrów")



# ---------------- Komórka 47 z notebooka ----------------
WYNIKI["liczby_parametrow"] = {"sequential":int(model_seq.count_params()), "duzy":int(model_duzy.count_params()), "dense_28x28_512":int(liczba_parametrow([784,512])[1])}


# 4.3 Functional API


# ---------------- Komórka 49 z notebooka ----------------
wej = keras.Input(shape=(2,), name="wejscie")
h1 = layers.Dense(32, activation="relu", name="ukryta_1")(wej)          # warstwa wywołana na tensorze
h2 = layers.Dense(32, activation="relu", name="ukryta_2")(h1)
polacz = layers.Concatenate(name="skip")([wej, h2])                      # połączenie pomijające: [x, h2]
wyj = layers.Dense(1, activation="sigmoid", name="wyjscie")(polacz)
model_func_demo = keras.Model(inputs=wej, outputs=wyj, name="mlp_functional_skip")
model_func_demo.summary()
# Zwróć uwagę: warstwa wyjściowa ma (32 + 2 + 1) * 1 = 35 parametrów - wejście jest doklejone


# Zadanie 4.2 - Functional API na danych `make_circles`


# ---------------- Komórka 51 z notebooka ----------------
X_c, y_c = make_circles(n_samples=N_PROBEK, noise=0.1, factor=0.5, random_state=SEED)
X_c = X_c.astype(np.float32)
Xc_tr, Xc_va, yc_tr, yc_va = train_test_split(X_c, y_c, test_size=0.25, random_state=SEED, stratify=y_c)

# Dwa okręgi: potrzebna jest nieliniowość, sama prosta nie ma jak ich oddzielić.
wej_c = keras.Input(shape=(2,), name="wejscie_circles")
h_c = layers.Dense(16, activation="tanh", name="ukryta_c1")(wej_c)
h_c = layers.Dense(8, activation="tanh", name="ukryta_c2")(h_c)
wyj_c = layers.Dense(1, activation="sigmoid", name="wyjscie_c")(h_c)
model_c = keras.Model(inputs=wej_c, outputs=wyj_c, name="mlp_circles_functional")
model_c.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss="binary_crossentropy", metrics=["accuracy"])
hist_c = model_c.fit(Xc_tr, yc_tr, validation_data=(Xc_va, yc_va),
                     epochs=EPOKI_KERAS, batch_size=32, verbose=0)



# ---------------- Komórka 52 z notebooka ----------------
if model_c is not None and hist_c is not None:
    _, acc_c = model_c.evaluate(Xc_va, yc_va, verbose=0)
    print(f"make_circles: dokładność walidacyjna = {acc_c:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(hist_c.history["loss"], label="train"); ax[0].plot(hist_c.history["val_loss"], label="walidacja")
    ax[0].set_title("make_circles: krzywe uczenia"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("BCE"); ax[0].legend(); ax[0].grid(alpha=0.3)
    rysuj_granice(lambda P: model_c.predict(P, verbose=0, batch_size=4096).ravel(), Xc_va, yc_va, ax=ax[1],
                  tytul=f"Functional API, make_circles (val acc = {acc_c:.3f})")
    plt.tight_layout(); zapisz_wykres("10_keras_functional_circles.png")



# ---------------- Komórka 53 z notebooka ----------------
if model_c is not None and hist_c is not None:
    WYNIKI["keras_circles"]={"accuracy_val":float(acc_c),"parametry":int(model_c.count_params())}
    if not SZYBKI_TRYB:
        uwaga(acc_c >= 0.97, "Functional API circles: val acc <0.97 (kryterium pełnego trybu)")


# 5. Ten sam model w PyTorch


# ---------------- Komórka 55 z notebooka ----------------
class MLP(nn.Module):
    def __init__(self, n_wej=2, n_ukr=16, n_warstw=2):
        super().__init__()
        warstwy, wej = [], n_wej
        for _ in range(n_warstw):                       # n_warstw warstw ukrytych z ReLU
            warstwy += [nn.Linear(wej, n_ukr), nn.ReLU()]
            wej = n_ukr
        warstwy.append(nn.Linear(wej, 1))               # wyjście: 1 logit (bez sigmoidu!)
        self.siec = nn.Sequential(*warstwy)

    def forward(self, x):
        return self.siec(x).squeeze(-1)                 # (batch, 1) -> (batch,)

torch.manual_seed(SEED)
model_pt = MLP().to(DEVICE)
print(model_pt)
print("liczba parametrów:", sum(p.numel() for p in model_pt.parameters()), "(porównaj z model_seq w Keras)")

# Dane -> tensory float32 -> TensorDataset -> DataLoader
ds_tr = TensorDataset(torch.tensor(Xm_tr), torch.tensor(ym_tr, dtype=torch.float32))
ds_va = TensorDataset(torch.tensor(Xm_va), torch.tensor(ym_va, dtype=torch.float32))
loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)     # tasowanie co epokę
loader_va = DataLoader(ds_va, batch_size=256, shuffle=False)   # walidacja: bez tasowania
xb, yb = next(iter(loader_tr))
print("przykładowy batch:", tuple(xb.shape), tuple(yb.shape), xb.dtype)


# Zadanie 5.1 - Pętla treningowa w PyTorch


# ---------------- Komórka 57 z notebooka ----------------
def trenuj_epoke(model, loader, opt, loss_fn):
    model.train()
    suma_straty, poprawne, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)  # Zeruję przy KAŻDYM batchu: gradienty inaczej się sumują.
        logity = model(xb)
        strata = loss_fn(logity, yb)
        strata.backward()
        opt.step()
        n_batcha = yb.shape[0]
        suma_straty += float(strata.item()) * n_batcha
        poprawne += int(((logity.detach() > 0) == (yb > 0.5)).sum().item())
        n += n_batcha
    if n == 0 or not np.isfinite(suma_straty):
        raise ValueError("Trening niepoprawny: pusty DataLoader albo strata NaN/inf")
    return suma_straty / n, poprawne / n


def ewaluuj(model, loader, loss_fn):
    model.eval()
    suma_straty, poprawne, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logity = model(xb)
            strata = loss_fn(logity, yb)
            n_batcha = yb.shape[0]
            suma_straty += float(strata.item()) * n_batcha
            poprawne += int(((logity > 0) == (yb > 0.5)).sum().item())
            n += n_batcha
    if n == 0 or not np.isfinite(suma_straty):
        raise ValueError("Walidacja niepoprawna: puste dane lub NaN/inf")
    return suma_straty / n, poprawne / n



# ---------------- Komórka 58 z notebooka ----------------
torch.manual_seed(SEED)
model_pt = MLP().to(DEVICE)
opt = torch.optim.Adam(model_pt.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

hist_pt = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
t0 = time.perf_counter()
for ep in range(EPOKI_KERAS):
    l_tr, a_tr = trenuj_epoke(model_pt, loader_tr, opt, loss_fn)
    l_va, a_va = ewaluuj(model_pt, loader_va, loss_fn)
    for k, v in zip(hist_pt, [l_tr, l_va, a_tr, a_va]):
        hist_pt[k].append(v)
    if ep % max(1, EPOKI_KERAS // 5) == 0 or ep == EPOKI_KERAS - 1:
        print(f"epoka {ep + 1:3d}: loss {l_tr:.4f} | val_loss {l_va:.4f} | val_acc {a_va:.4f}")
print(f"czas: {time.perf_counter() - t0:.1f} s")

def predict_pt(P):
    model_pt.eval()
    with torch.no_grad():
        return torch.sigmoid(model_pt(torch.tensor(P, device=DEVICE))).cpu().numpy()   # logit -> p

fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
ax[0].plot(hist_pt["loss"], label="PyTorch train"); ax[0].plot(hist_pt["val_loss"], label="PyTorch walidacja")
ax[0].plot(hist_seq.history["val_loss"], "--", alpha=0.7, label="Keras walidacja")
ax[0].set_title("Strata BCE: PyTorch vs Keras"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("BCE"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(hist_pt["acc"], label="PyTorch train"); ax[1].plot(hist_pt["val_acc"], label="PyTorch walidacja")
ax[1].plot(hist_seq.history["val_accuracy"], "--", alpha=0.7, label="Keras walidacja")
ax[1].set_title("Dokładność: PyTorch vs Keras"); ax[1].set_xlabel("epoka"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
rysuj_granice(predict_pt, Xm_va, ym_va, ax=ax[2], tytul=f"PyTorch MLP (val acc = {hist_pt['val_acc'][-1]:.3f})")
plt.tight_layout(); zapisz_wykres("11_pytorch_mlp.png")



# ---------------- Komórka 59 z notebooka ----------------
WYNIKI["pytorch_moons"]={"accuracy_val":float(hist_pt["val_acc"][-1]), "loss_val":float(hist_pt["val_loss"][-1])}
if not SZYBKI_TRYB:
    uwaga(hist_pt["val_acc"][-1] >= 0.94, "PyTorch moons: walidacja poniżej 0.94")


# 6. Pojemność sieci: liczba warstw i neuronów a granica decyzyjna


# Zadanie 6.1 - Siatka architektur: 1-3 warstwy x 2/8/64 neurony


# ---------------- Komórka 62 z notebooka ----------------
KROKI_SIATKA = 60 if SZYBKI_TRYB else 2000
X_s_tr, y_s_tr = make_moons(n_samples=200, noise=0.3, random_state=1)
X_s_va, y_s_va = make_moons(n_samples=1000, noise=0.3, random_state=2)
X_s_tr, X_s_va = X_s_tr.astype(np.float32), X_s_va.astype(np.float32)

Xs_tr_t = torch.tensor(X_s_tr, device=DEVICE); ys_tr_t = torch.tensor(y_s_tr, dtype=torch.float32, device=DEVICE)
Xs_va_t = torch.tensor(X_s_va, device=DEVICE); ys_va_t = torch.tensor(y_s_va, dtype=torch.float32, device=DEVICE)

wyniki_siatka = {}
t0 = time.perf_counter()
for n_warstw in [1, 2, 3]:
    for n_neuronow in [2, 8, 64]:
        # Ten sam zbiór i optymalizator. Zmieniam tylko głębokość i szerokość sieci.
        torch.manual_seed(SEED + n_warstw * 100 + n_neuronow)
        model = MLP(n_wej=2, n_ukr=n_neuronow, n_warstw=n_warstw).to(DEVICE)
        opt_s = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_s = nn.BCEWithLogitsLoss()
        for krok in range(KROKI_SIATKA):
            opt_s.zero_grad(set_to_none=True)
            logity = model(Xs_tr_t)
            strata = loss_s(logity, ys_tr_t)
            if not torch.isfinite(strata):
                raise ValueError(f"Siatka {n_warstw}x{n_neuronow}: niestabilna strata")
            strata.backward()
            opt_s.step()
        model.eval()
        with torch.no_grad():
            acc_tr = ((model(Xs_tr_t) > 0) == (ys_tr_t > 0.5)).float().mean().item()
            acc_va = ((model(Xs_va_t) > 0) == (ys_va_t > 0.5)).float().mean().item()
        wyniki_siatka[(n_warstw, n_neuronow)] = (model, float(acc_tr), float(acc_va))
        print(f"  architektura {n_warstw}x{n_neuronow}: train={acc_tr:.3f}, val={acc_va:.3f}")
print(f"czas treningu 9 modeli: {time.perf_counter() - t0:.1f} s")



# ---------------- Komórka 63 z notebooka ----------------
if wyniki_siatka:
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for i, n_warstw in enumerate([1, 2, 3]):
        for j, n_neuronow in enumerate([2, 8, 64]):
            m, acc_tr, acc_va = wyniki_siatka[(n_warstw, n_neuronow)]
            n_par = sum(p.numel() for p in m.parameters())
            def pred(P, m=m):
                with torch.no_grad():
                    return torch.sigmoid(m(torch.tensor(P, device=DEVICE))).cpu().numpy()
            rysuj_granice(pred, X_s_tr, y_s_tr, ax=axes[i, j], rozdz=150,
                          tytul=f"{n_warstw} warstw. x {n_neuronow} neur. ({n_par} par.)\ntrain {acc_tr:.2f} | val {acc_va:.2f}")
    plt.suptitle("Granice decyzyjne MLP (punkty: 200 próbek treningowych)", fontsize=14)
    plt.tight_layout(); zapisz_wykres("12_siatka_architektur.png")

    najlepszy = max(wyniki_siatka, key=lambda k: wyniki_siatka[k][2])
    print("najlepsza architektura wg walidacji (warstwy, neurony):", najlepszy,
          "val acc =", round(wyniki_siatka[najlepszy][2], 3))



# ---------------- Komórka 64 z notebooka ----------------
WYNIKI["siatka_architektur"]={f"{a}x{b}":{"accuracy_train":float(tr), "accuracy_val":float(va), "parametry":int(sum(p.numel() for p in m.parameters()))} for (a,b),(m,tr,va) in wyniki_siatka.items()}
WYNIKI["najlepsza_architektura"] = f"{najlepszy[0]}x{najlepszy[1]}"


# Pytania kontrolne


# Podsumowanie - co warto zapamiętać


# Do sprawozdania końcowego


# Zadanie dodatkowe (dla chętnych) - Uniwersalny MLP w NumPy dla klasyfikacji wieloklasowej (spirala)


# ---------------- Komórka 69 z notebooka ----------------
def spirala(n_na_klase=100, k=3, szum=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for j in range(k):
        r = np.linspace(0.0, 1.0, n_na_klase)                                   # promień
        t = np.linspace(j * 4, (j + 1) * 4, n_na_klase) + rng.normal(0, szum, n_na_klase)  # kąt
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y.append(np.full(n_na_klase, j))
    return np.vstack(X), np.concatenate(y)

X_sp, y_sp = spirala()
ITER_SP = 300 if SZYBKI_TRYB else 5000


class MLPNumpy:
    def __init__(self, rozmiary, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(rozmiary[:-1], rozmiary[1:])]
        self.b = [np.zeros(b) for b in rozmiary[1:]]

    def forward(self, X):
        # Tu nie używam żadnej magicznej biblioteki: macierze, ReLU i stabilny softmax.
        self.aktywacje = [X]
        self.pre_aktywacje = []
        A = X
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            self.pre_aktywacje.append(Z)
            A = np.maximum(Z, 0.0)
            self.aktywacje.append(A)
        Z = A @ self.W[-1] + self.b[-1]
        self.pre_aktywacje.append(Z)
        Z -= Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        P = e / e.sum(axis=1, keepdims=True)
        self.aktywacje.append(P)
        return P

    def strata(self, P, y):
        return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))))

    def backward(self, y):
        n = len(y)
        delta = self.aktywacje[-1].copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n
        self.dW = [np.zeros_like(W) for W in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
        for i in reversed(range(len(self.W))):
            self.dW[i] = self.aktywacje[i].T @ delta
            self.db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (self.pre_aktywacje[i - 1] > 0)

    def krok(self, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * self.dW[l]
            self.b[l] -= lr * self.db[l]


mlp_np = MLPNumpy([2, 64, 64, 3])
hist_sp = []
for it in range(ITER_SP):
    P = mlp_np.forward(X_sp)
    hist_sp.append(mlp_np.strata(P, y_sp))
    mlp_np.backward(y_sp)
    mlp_np.krok(lr=0.5)
acc_sp = np.mean(mlp_np.forward(X_sp).argmax(axis=1) == y_sp)
print(f"spirala: dokładność treningowa = {acc_sp:.3f}, strata = {hist_sp[-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(hist_sp, label="CE train"); ax[0].set_title("MLP NumPy na spirali: krzywa uczenia")
ax[0].set_xlabel("iteracja"); ax[0].set_ylabel("entropia krzyżowa"); ax[0].legend(); ax[0].grid(alpha=0.3)
rysuj_granice(lambda Pn: mlp_np.forward(Pn).argmax(axis=1), X_sp, y_sp, ax=ax[1], wieloklasowa=True,
              tytul=f"Granice 3 klas (acc = {acc_sp:.2f})")
plt.tight_layout(); zapisz_wykres("13_spirala_numpy.png")



# ---------------- Komórka 70 z notebooka ----------------
WYNIKI["spirala_bonus"]={"accuracy_train":float(acc_sp), "loss_last":float(hist_sp[-1])}
if not SZYBKI_TRYB:
    uwaga(acc_sp > 0.95, "Spirala bonus: wynik poniżej celu 95%")


# Wyniki uruchomienia lokalnego


# ---------------- Komórka 72 z notebooka ----------------
WYNIKI["wykresy"] = WYKRESY
WYNIKI["ostrzezenia"] = OSTRZEZENIA
WYNIKI["status"] = "UWAGA - sprawdź kryteria jakościowe" if OSTRZEZENIA else "OK - bez błędów testów"
(KATALOG_WYNIKOW / "podsumowanie.json").write_text(
    json.dumps(WYNIKI, ensure_ascii=False, indent=2), encoding="utf-8")
wnioski = [
    "LAB02 - Perceptron i MLP | wyniki z tego uruchomienia",
    "Tryb: " + WYNIKI["tryb"],
    "Perceptron klasyfikuje dane separowalne, ale nie rozwiązuje XOR.",
    f"XOR perceptron accuracy: {WYNIKI['perceptron_xor']['accuracy']:.3f}; MLP BCE: {WYNIKI['mlp_xor']['BCE']:.5f}.",
    f"Keras Sequential val acc: {WYNIKI['keras_sequential']['accuracy_val']:.3f}.",
    f"Keras Functional circles val acc: {WYNIKI['keras_circles']['accuracy_val']:.3f}.",
    f"PyTorch val acc: {WYNIKI['pytorch_moons']['accuracy_val']:.3f}.",
    "Najlepsza architektura z siatki (walidacja): " + WYNIKI['najlepsza_architektura'],
    f"Spirala (bonus) accuracy: {WYNIKI['spirala_bonus']['accuracy_train']:.3f}.",
    "Najważniejsze: nieliniowość rozwiązuje XOR, ale większa pojemność nie gwarantuje lepszej walidacji.",
    "Ostrzeżenia: " + ("; ".join(OSTRZEZENIA) if OSTRZEZENIA else "brak"),
]
(KATALOG_WYNIKOW / "wnioski.txt").write_text("\n".join(wnioski)+"\n", encoding="utf-8")
print("\n".join(wnioski))
print("Zapisano", len(WYKRESY), "wykresów w", KATALOG_WYKRESOW)
print("STATUS:", WYNIKI["status"])
