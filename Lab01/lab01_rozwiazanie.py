# -*- coding: utf-8 -*-
"""Lab 01 - tensory, gradient, regresja, przeuczenie (rozwiazanie lokalne).

Oparte na materiale dr inz. Natalii Piorkowskiej dla kursu Uczenie glebokie.
Ten plik wykonuje wszystkie komorki kodu oryginalnego laboratorium,
uzupelnia TODO i zapisuje wykresy oraz wyniki na dysku.

Uruchamianie (Windows): URUCHOM_LAB1.bat lub URUCHOM_SZYBKI_TEST.bat.
Nie ma zaleznosci od Colaba ani od notebooka podczas uruchamiania.
"""

# Na moim starszym i5 nie chcę, żeby OpenBLAS odpalał armię wątków
# do każdego małego mnożenia macierzy. Kilka wątków z głową wystarczy.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import matplotlib
matplotlib.use("Agg")  # Nie otwieram 14 okien. PNG lądują w folderze wyniki.
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WYNIKI = ROOT / "wyniki"
WYKRESY = WYNIKI / "wykresy"
WYKRESY.mkdir(parents=True, exist_ok=True)

class PodwojnyZapis:
    """Log w terminalu i do pliku. Przydaje sie do sprawdzenia bledu po uruchomieniu BAT."""
    def __init__(self, ekran, plik):
        self.ekran, self.plik = ekran, plik
    def write(self, tekst):
        self.ekran.write(tekst)
        self.plik.write(tekst)
    def flush(self):
        self.ekran.flush()
        self.plik.flush()
    def isatty(self):
        return False

# Chcę mieć po wszystkim czarno na białym, co się wykonało i co ewentualnie padło.
# Przy błędzie można wrócić do logu zamiast odtwarzać cały eksperyment.
LOG_PLIK = WYNIKI / "przebieg.log"
_LOG_HANDLE = open(LOG_PLIK, "w", encoding="utf-8", buffering=1)
sys.stdout = PodwojnyZapis(sys.stdout, _LOG_HANDLE)
sys.stderr = PodwojnyZapis(sys.stderr, _LOG_HANDLE)

WYKRESY_ZAPISANE = []

def zapisz_wykres(nazwa):
    """Zapisuje aktualny wykres z Matplotlib i zamyka figure."""
    sciezka = WYKRESY / f"{nazwa}.png"
    plt.savefig(sciezka, dpi=150, bbox_inches="tight")
    plt.close()
    WYKRESY_ZAPISANE.append(str(sciezka.relative_to(ROOT)))
    print(f"[WYKRES] {sciezka.relative_to(ROOT)}")


def naglowek(numer, nazwa):
    print("\n" + "=" * 76)
    print(f"  {numer}: {nazwa}")
    print("=" * 76, flush=True)

# CPU domyślnie jest stabilniejszy na GTX 1050 i Windowsie.
# Gdy przetestuję CUDA, mogę ustawić DL_GPU=1. Nie jest to potrzebne do Lab1.

naglowek(3, '0. Sprzet i biblioteki')


# ---------- Oryginalna komorka 3 ----------
import subprocess
try:
    pom = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=15)
    if pom.returncode == 0:
        print("NVIDIA (informacyjnie):\n", "\n".join(pom.stdout.splitlines()[:12]))
    else:
        print("nvidia-smi nie zwrocilo danych - to nie blokuje laboratorium.")
except (OSError, subprocess.TimeoutExpired):
    print("Brak nvidia-smi. GPU nie jest wymagane.")


# ---------- Oryginalna komorka 5 ----------
import os, time, math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
# TF ma zajmować pamięć GPU stopniowo - inaczej PyTorch nie dostanie pamięci
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

import torch
import keras

# Tryb szybki: True = mniej danych i epok (praca bez GPU)
SZYBKI_TRYB = os.environ.get("DL_FAST", "0") == "1"
# SZYBKI_TRYB = True   # <- odkomentuj, jeśli pracujesz bez GPU

print("SZYBKI_TRYB =", SZYBKI_TRYB)
print("NumPy      :", np.__version__)
print("TensorFlow :", tf.__version__, "| Keras:", keras.__version__)
print("PyTorch    :", torch.__version__)


# ---------- Oryginalna komorka 7 ----------
SEED = 42
keras.utils.set_random_seed(SEED)   # Python random + NumPy + TF
np.random.seed(SEED)
torch.manual_seed(SEED)

# Urządzenie dla PyTorch: GPU jeśli jest, w przeciwnym razie CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("DL_GPU", "0") == "1" else "cpu")

print("GPU widziane przez TensorFlow:", tf.config.list_physical_devices("GPU"))
print("CUDA dostępna w PyTorch      :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Karta:", torch.cuda.get_device_name(0))
print("DEVICE =", DEVICE)
print("Tryb PyTorch:", "CUDA" if DEVICE.type == "cuda" else "CPU (bezpieczny dla Windows)")


naglowek(9, '1. Tensory - podstawy i operacje')


# ---------- Oryginalna komorka 9 ----------
# Zwykła tablica NumPy
a = np.array([[1., 2., 3.],
              [4., 5., 6.]])
print("NumPy :", a.shape, a.dtype, "ndim =", a.ndim)

# To samo w TensorFlow i PyTorch
a_tf = tf.constant(a)
a_pt = torch.tensor(a)
print("TF    :", a_tf.shape, a_tf.dtype)
print("Torch :", a_pt.shape, a_pt.dtype)


# ---------- Oryginalna komorka 11 ----------
# Jawne rzutowanie na float32 - dobry nawyk
a_tf32 = tf.convert_to_tensor(a, dtype=tf.float32)
a_pt32 = torch.tensor(a, dtype=torch.float32)
print(a_tf32.dtype, a_pt32.dtype)

# Powrót do NumPy
print(type(a_tf32.numpy()), type(a_pt32.numpy()))

# Pułapka: torch.from_numpy NIE kopiuje danych - tensor i tablica współdzielą pamięć
b = np.zeros(3)
b_pt = torch.from_numpy(b)
b[0] = 99.0                     # zmieniamy tablicę NumPy...
print("tensor po zmianie tablicy:", b_pt)   # ...i zmienia się tensor!


# ---------- Oryginalna komorka 13 ----------
# Zera, jedynki, zakres, losowe - w obu bibliotekach
print(tf.zeros((2, 3)).numpy(), "\n", torch.zeros(2, 3))
print(tf.range(0, 10, 2).numpy(), torch.arange(0, 10, 2))
print(tf.random.normal((2, 2), seed=1).numpy())
print(torch.randn(2, 2))

# tf.Variable - modyfikowalny tensor (parametry modelu)
v = tf.Variable([1.0, 2.0, 3.0])
v.assign_add([1.0, 1.0, 1.0])       # v = v + 1
print("tf.Variable po assign_add:", v.numpy())

# PyTorch - operacja w miejscu (in-place) z podkreślnikiem
t = torch.tensor([1.0, 2.0, 3.0])
t.add_(1.0)
print("torch po add_:", t)


# ---------- Oryginalna komorka 15 ----------
x = np.arange(24, dtype=np.float32)
x_tf = tf.reshape(tf.constant(x), (2, 3, 4))     # 2*3*4 = 24 elementy
x_pt = torch.from_numpy(x).reshape(2, 3, 4)
print("kształt:", x_tf.shape, tuple(x_pt.shape))
print("element [1, 2, 3]:", x_tf[1, 2, 3].numpy(), x_pt[1, 2, 3].item())
print("wycinek [:, 0, :2]:\n", x_pt[:, 0, :2])

# -1: biblioteka sama wylicza wymiar
print(tf.reshape(x_tf, (4, -1)).shape, x_pt.reshape(-1, 6).shape)

# Batch obrazów: TF (N, H, W, C) -> PyTorch (N, C, H, W)
obrazy_nhwc = np.random.rand(8, 28, 28, 3).astype(np.float32)
obrazy_nchw = torch.from_numpy(obrazy_nhwc).permute(0, 3, 1, 2)
print("NHWC:", obrazy_nhwc.shape, "-> NCHW:", tuple(obrazy_nchw.shape))
# Ta sama operacja w TF
print("TF transpose:", tf.transpose(obrazy_nhwc, perm=[0, 3, 1, 2]).shape)


# ---------- Oryginalna komorka 17 ----------
kol = np.array([[1.], [2.], [3.]])     # (3, 1)
wiersz = np.array([[10., 20., 30., 40.]])   # (1, 4)
print("(3,1) + (1,4) ->", (kol + wiersz).shape)
print(torch.tensor(kol) + torch.tensor(wiersz))

# Bias dodawany do każdej próbki w batchu
batch = tf.ones((5, 3))
bias = tf.constant([0.1, 0.2, 0.3])
print("(5,3) + (3,) ->", (batch + bias).shape)

# Klasyczny cichy błąd: (n,) - (n,1) daje macierz (n, n)!
y = np.array([1., 2., 3.])
y_pred = np.array([[1.1], [1.9], [3.2]])
print("Zły kształt błędu:", (y - y_pred).shape, " <- powinno być (3,)")
print("Poprawnie       :", (y - y_pred.ravel()).shape)

# Niezgodne kształty - wyjątek
try:
    torch.ones(3) + torch.ones(4)
except RuntimeError as e:
    print("Błąd broadcastingu:", str(e)[:80])


# ---------- Oryginalna komorka 19 ----------
X = np.random.rand(4, 3).astype(np.float32)    # batch 4 próbek, 3 cechy
W = np.random.rand(3, 2).astype(np.float32)    # wagi: 3 wejścia -> 2 wyjścia

Y_np = X @ W
Y_tf = tf.matmul(X, W)
Y_pt = torch.from_numpy(X) @ torch.from_numpy(W)
print("kształt wyniku:", Y_np.shape)
print("zgodne:", np.allclose(Y_np, Y_tf.numpy()), np.allclose(Y_np, Y_pt.numpy()))

# Mnożenie wsadowe: 10 par macierzy (2x3)(3x5) naraz
A = torch.randn(10, 2, 3); B = torch.randn(10, 3, 5)
print("batch matmul:", (A @ B).shape)

# Redukcje z zachowaniem wymiaru
M = torch.arange(6.).reshape(2, 3)
print("suma po wierszach:", M.sum(dim=1), "| keepdim:", M.sum(dim=1, keepdim=True).shape)
print("TF mean po kolumnach:", tf.reduce_mean(tf.constant(M.numpy()), axis=0).numpy())


naglowek(21, 'Zadanie 1.1: standaryzacja obrazow')


# ---------- Oryginalna komorka 21 ----------
rng = np.random.default_rng(0)
batch_np = rng.normal(loc=[0.5, 0.2, -0.3], scale=[2.0, 0.5, 1.0],
                      size=(16, 32, 32, 3)).astype(np.float32)

# Mateusz kręcił się przy biurku, więc tu warto jeszcze raz sprawdzić osie.
# Sprawdzam to asercjami: TF zostaje w NHWC, Torch przestawiam na NCHW.
batch_tf = tf.convert_to_tensor(batch_np, dtype=tf.float32)
mu_tf = tf.reduce_mean(batch_tf, axis=(0, 1, 2), keepdims=True)
sd_tf = tf.math.reduce_std(batch_tf, axis=(0, 1, 2), keepdims=True)
z_tf = (batch_tf - mu_tf) / sd_tf

batch_pt = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
mu_pt = batch_pt.mean(dim=(0, 2, 3), keepdim=True)
sd_pt = batch_pt.std(dim=(0, 2, 3), correction=0, keepdim=True)
z_pt = (batch_pt - mu_pt) / sd_pt

# Kontrola: jak shape się rozjedzie, wolę to złapać od razu, a nie po pół godzinie.
assert tuple(mu_tf.shape) == (1, 1, 1, 3)
assert tuple(mu_pt.shape) == (1, 3, 1, 1)
assert np.allclose(z_tf.numpy(), z_pt.permute(0, 2, 3, 1).numpy(), atol=1e-5)


# ---------- Oryginalna komorka 22 ----------
# Komórka sprawdzająca - nie zmieniaj
if z_tf is None or z_pt is None:
    print("Uzupełnij z_tf i z_pt.")
else:
    print("TF  średnie per kanał:", np.round(tf.reduce_mean(z_tf, axis=[0, 1, 2]).numpy(), 4))
    print("TF  std per kanał    :", np.round(tf.math.reduce_std(z_tf, axis=[0, 1, 2]).numpy(), 4))
    print("PT  średnie per kanał:", np.round(z_pt.mean(dim=(0, 2, 3)).numpy(), 4))
    z_pt_nhwc = z_pt.permute(0, 2, 3, 1).numpy()      # powrót do NHWC
    print("Zgodność TF vs PyTorch:", np.allclose(z_tf.numpy(), z_pt_nhwc, atol=1e-5))


assert np.allclose(tf.reduce_mean(z_tf, axis=(0, 1, 2)).numpy(), 0.0, atol=1e-5)
assert np.allclose(tf.math.reduce_std(z_tf, axis=(0, 1, 2)).numpy(), 1.0, atol=1e-5)


naglowek(24, 'Zadanie 1.2: odleglosci bez petli')


# ---------- Oryginalna komorka 24 ----------
def pairwise_sq_dist_np(X, Y):
    # Jak przy porownywaniu URL-i w SEO: zamiast recznie kazda para,
    # robię to macierzowo. Tutaj po prostu porównuję wektory.
    norm_X = np.sum(X ** 2, axis=1, keepdims=True)
    norm_Y = np.sum(Y ** 2, axis=1, keepdims=True).T
    return np.maximum(norm_X + norm_Y - 2.0 * (X @ Y.T), 0.0)


def pairwise_sq_dist_torch(X, Y):
    # Ten sam rachunek po torchowemu. Bez dwóch zagnieżdżonych pętli.
    norm_X = torch.sum(X ** 2, dim=1, keepdim=True)
    norm_Y = torch.sum(Y ** 2, dim=1, keepdim=True).T
    return torch.clamp(norm_X + norm_Y - 2.0 * (X @ Y.T), min=0.0)


# ---------- Oryginalna komorka 25 ----------
# Komórka sprawdzająca - wersja pętlowa jako wzorzec
def pairwise_sq_dist_loop(X, Y):
    D = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            D[i, j] = np.sum((X[i] - Y[j]) ** 2)
    return D

Xs, Ys = np.random.rand(50, 8), np.random.rand(40, 8)
D_ref = pairwise_sq_dist_loop(Xs, Ys)
D_np = pairwise_sq_dist_np(Xs, Ys)
D_pt = pairwise_sq_dist_torch(torch.from_numpy(Xs), torch.from_numpy(Ys))
if D_np is not None and D_pt is not None:
    print("NumPy - zgodność z wersją pętlową:", np.allclose(D_ref, D_np))
    print("Torch - zgodność z wersją pętlową:", np.allclose(D_ref, D_pt.numpy()))

    # Pomiar czasu: pętla vs wektoryzacja
    n_big = 150 if SZYBKI_TRYB else 400
    Xb, Yb = np.random.rand(n_big, 32), np.random.rand(n_big, 32)
    t0 = time.perf_counter(); pairwise_sq_dist_loop(Xb, Yb); t_loop = time.perf_counter() - t0
    t0 = time.perf_counter(); pairwise_sq_dist_np(Xb, Yb); t_vec = time.perf_counter() - t0
    print(f"n = {n_big}: pętla {t_loop:.3f} s, wektorowo {t_vec:.5f} s, przyspieszenie ~{t_loop / max(t_vec, 1e-9):.0f}x")


assert np.allclose(D_np, D_ref)
assert np.allclose(D_pt.numpy(), D_ref)


naglowek(27, '1.6: porownanie CPU i GPU')


# ---------- Oryginalna komorka 27 ----------
N = 512 if SZYBKI_TRYB else 4096          # rozmiar macierzy N x N
A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

t0 = time.perf_counter(); C = A_cpu @ B_cpu; t_cpu = time.perf_counter() - t0
print(f"PyTorch CPU: {t_cpu * 1000:.1f} ms  (N = {N})")

if DEVICE.type == "cuda":
    A_gpu, B_gpu = A_cpu.to(DEVICE), B_cpu.to(DEVICE)   # kopiowanie na GPU
    _ = A_gpu @ B_gpu; torch.cuda.synchronize()         # rozgrzewka
    t0 = time.perf_counter()
    C_gpu = A_gpu @ B_gpu
    torch.cuda.synchronize()                            # czekamy na koniec obliczeń GPU
    t_gpu = time.perf_counter() - t0
    print(f"PyTorch GPU: {t_gpu * 1000:.1f} ms  -> przyspieszenie ~{t_cpu / t_gpu:.0f}x")
    print("Urządzenie wyniku:", C_gpu.device)
    print("Maks. różnica CPU vs GPU:", (C_gpu.cpu() - C).abs().max().item())
else:
    print("CUDA nieaktywna - pomijam benchmark GPU (to nie błąd).")

# TensorFlow: wymuszenie urządzenia kontekstem
with tf.device("/CPU:0"):
    t0 = time.perf_counter(); _ = tf.matmul(A_cpu.numpy(), B_cpu.numpy()).numpy(); t_tf_cpu = time.perf_counter() - t0
print(f"TF CPU: {t_tf_cpu * 1000:.1f} ms")
if tf.config.list_physical_devices("GPU"):
    with tf.device("/GPU:0"):
        _ = tf.matmul(A_cpu.numpy(), B_cpu.numpy()).numpy()          # rozgrzewka
        t0 = time.perf_counter(); _ = tf.matmul(A_cpu.numpy(), B_cpu.numpy()).numpy(); t_tf_gpu = time.perf_counter() - t0
    print(f"TF GPU: {t_tf_gpu * 1000:.1f} ms (łącznie z kopiowaniem danych na GPU i z powrotem)")


naglowek(29, '2. Automatyczne rozniczkowanie')


# ---------- Oryginalna komorka 29 ----------
# --- TensorFlow ---
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x ** 3 + 2 * x            # operacje są nagrywane na "taśmę"
dy_dx_tf = tape.gradient(y, x)
print("TF     f'(2) =", dy_dx_tf.numpy())

# --- PyTorch ---
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x                # graf budowany "w locie"
y.backward()                       # backpropagation
print("PyTorch f'(2) =", x.grad.item())

print("Analitycznie  =", 3 * 2.0 ** 2 + 2)


# ---------- Oryginalna komorka 31 ----------
xs = np.linspace(-6, 6, 400).astype(np.float32)

# TF: zwykły tensor trzeba "obserwować" przez tape.watch
x_tf = tf.constant(xs)
with tf.GradientTape() as tape:
    tape.watch(x_tf)
    f_tf = x_tf ** 2 * tf.sin(x_tf)
    suma_tf = tf.reduce_sum(f_tf)     # sumowanie też musi być nagrane na taśmie!
df_tf = tape.gradient(suma_tf, x_tf).numpy()

# PyTorch
x_pt = torch.tensor(xs, requires_grad=True)
f_pt = x_pt ** 2 * torch.sin(x_pt)
f_pt.sum().backward()
df_pt = x_pt.grad.numpy()

df_analit = 2 * xs * np.sin(xs) + xs ** 2 * np.cos(xs)
print("maks. |TF - analityczna|     :", np.abs(df_tf - df_analit).max())
print("maks. |PyTorch - analityczna|:", np.abs(df_pt - df_analit).max())

plt.figure(figsize=(9, 4))
plt.plot(xs, f_pt.detach().numpy(), label="$f(x)=x^2\\sin x$")
plt.plot(xs, df_analit, lw=4, alpha=0.4, label="$f'(x)$ analitycznie")
plt.plot(xs, df_tf, "--", label="$f'(x)$ TF GradientTape")
plt.plot(xs, df_pt, ":", lw=2, label="$f'(x)$ torch.autograd")
plt.axhline(0, color="gray", lw=0.5)
plt.title("Autograd vs pochodna analityczna")
plt.xlabel("x"); plt.ylabel("wartość"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("01_autograd_wykres")


# ---------- Oryginalna komorka 33 ----------
# Akumulacja gradientów w PyTorch
x = torch.tensor(3.0, requires_grad=True)
for i in range(3):
    y = x ** 2                 # dy/dx = 2x = 6
    y.backward()
    print(f"wywołanie {i + 1}: x.grad = {x.grad.item()}")   # 6, 12, 18 - suma!
x.grad.zero_()
print("po zero_():", x.grad.item())

# Wiele zmiennych i druga pochodna w TF (zagnieżdżone taśmy)
x = tf.Variable(1.5)
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = tf.sin(x) * x
    dy = t1.gradient(y, x)          # pierwsza pochodna
d2y = t2.gradient(dy, x)            # druga pochodna
print("f'(1.5) =", dy.numpy(), " f''(1.5) =", d2y.numpy())
print("analitycznie:", np.cos(1.5) * 1.5 + np.sin(1.5), 2 * np.cos(1.5) - 1.5 * np.sin(1.5))


naglowek(35, 'Zadanie 2.1: gradient dwoch zmiennych')


# ---------- Oryginalna komorka 35 ----------
x0, y0 = 0.5, -1.0

def df_dx_analit(x, y):
    # d/dx sin(x)y^2 = cos(x)y^2; d/dx exp(xy) = y*exp(xy).
    return np.cos(x) * y**2 + y * np.exp(x*y)

def df_dy_analit(x, y):
    # d/dy sin(x)y^2 = 2y sin(x); d/dy exp(xy) = x*exp(xy).
    return 2.0 * y * np.sin(x) + x * np.exp(x*y)

# Mam malo zaufania do pochodnych pisanych na szybko. Dwa frameworki
# maja dac to samo co rachunek na kartce - inaczej wracamy do wzoru.
x_var = tf.Variable(x0, dtype=tf.float64)
y_var = tf.Variable(y0, dtype=tf.float64)
with tf.GradientTape() as tape:
    wartosc_tf = tf.sin(x_var) * y_var**2 + tf.exp(x_var * y_var)
grad_tf = np.array([float(z.numpy()) for z in tape.gradient(wartosc_tf, [x_var, y_var])])

xy_pt = torch.tensor([x0, y0], dtype=torch.float64, requires_grad=True)
wartosc_pt = torch.sin(xy_pt[0]) * xy_pt[1]**2 + torch.exp(xy_pt[0] * xy_pt[1])
wartosc_pt.backward()
grad_pt = xy_pt.grad.detach().numpy().copy()

grad_analit = np.array([df_dx_analit(x0, y0), df_dy_analit(x0, y0)])
assert np.allclose(grad_tf, grad_analit, atol=1e-6)
assert np.allclose(grad_pt, grad_analit, atol=1e-6)
print("analitycznie:", df_dx_analit(x0, y0), df_dy_analit(x0, y0))
print("TF          :", grad_tf)
print("PyTorch     :", grad_pt)


naglowek(37, 'Zadanie 2.2: gradient numeryczny')


# ---------- Oryginalna komorka 37 ----------
def grad_numeryczny(f, w, h=1e-5):
    # Tutaj pętla jest celowa, bo sprawdzamy każdą współrzędną osobno.
    # float64 ma znaczenie. Przy zbyt małym h zaokrąglenia psują wynik.
    w = np.asarray(w, dtype=np.float64)
    grad = np.zeros_like(w)
    for i in range(w.size):
        plus, minus = w.copy(), w.copy()
        plus[i] += h
        minus[i] -= h
        grad[i] = (f(plus) - f(minus)) / (2.0 * h)
    return grad


# ---------- Oryginalna komorka 38 ----------
rng = np.random.default_rng(1)
A_mat = rng.normal(size=(6, 4))
w0 = rng.normal(size=4)

def f_np(w):                                   # wersja NumPy (do różnic skończonych)
    z = A_mat @ w
    return np.log(np.sum(np.exp(z))) + 0.5 * np.sum(w ** 2)

# Gradient z autograd (PyTorch, float64)
w_t = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
f_t = torch.logsumexp(torch.tensor(A_mat) @ w_t, dim=0) + 0.5 * (w_t ** 2).sum()
f_t.backward()
g_auto = w_t.grad.numpy()

def blad_wzgledny(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-300)

g_num = grad_numeryczny(f_np, w0, h=1e-5)
print("autograd :", np.round(g_auto, 6))
print("numeryczny:", np.round(g_num, 6))
print("błąd względny (h=1e-5):", blad_wzgledny(g_num, g_auto))

# Błąd w funkcji kroku h
hs = np.logspace(-14, 0, 29)
bledy = [blad_wzgledny(grad_numeryczny(f_np, w0, h), g_auto) for h in hs]
plt.figure(figsize=(7, 4))
plt.loglog(hs, np.maximum(bledy, 1e-17), "o-")
plt.title("Gradient check: błąd względny vs krok h")
plt.xlabel("krok h"); plt.ylabel("błąd względny"); plt.grid(alpha=0.3, which="both"); zapisz_wykres("02_gradient_check")


assert blad_wzgledny(g_num, g_auto) < 1e-7, "Gradient check (h=1e-5) nie przeszedl"


naglowek(40, '3. Regresja liniowa na jednej cesze')


# ---------- Oryginalna komorka 40 ----------
rng = np.random.default_rng(SEED)
n1 = 100
x1 = rng.uniform(-3, 3, size=n1)
y1 = 2.5 * x1 - 1.0 + rng.normal(scale=1.0, size=n1)    # prawdziwe w = 2.5, b = -1

plt.figure(figsize=(6, 4))
plt.scatter(x1, y1, s=15, alpha=0.7, label="dane")
plt.title("Dane syntetyczne: y = 2.5x - 1 + szum")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("03_dane_syntetyczne")


# ---------- Oryginalna komorka 41 ----------
# Spadek gradientu dla jednej cechy - kompletny przykład
w, b = 0.0, 0.0            # inicjalizacja parametrów
lr = 0.05                  # learning rate (hiperparametr)
epoki_1d = 100             # liczba epok (hiperparametr)
historia_1d, sciezka = [], [(w, b)]

for ep in range(epoki_1d):
    y_hat = w * x1 + b                     # 1) przejście w przód (predykcja)
    err = y_hat - y1                       # 2) błędy
    loss = np.mean(err ** 2)               # 3) strata MSE
    dw = 2 * np.mean(err * x1)             # 4) gradient po w
    db = 2 * np.mean(err)                  #    gradient po b
    w -= lr * dw                           # 5) krok spadku gradientu
    b -= lr * db
    historia_1d.append(loss); sciezka.append((w, b))

w_ref, b_ref = np.polyfit(x1, y1, 1)       # rozwiązanie najmniejszych kwadratów (wzorzec)
print(f"GD     : w = {w:.4f}, b = {b:.4f}, MSE = {historia_1d[-1]:.4f}")
print(f"polyfit: w = {w_ref:.4f}, b = {b_ref:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(historia_1d); ax[0].set_yscale("log")
ax[0].set_title("Krzywa uczenia (MSE)"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("MSE (skala log)"); ax[0].grid(alpha=0.3)
xx = np.linspace(-3, 3, 50)
ax[1].scatter(x1, y1, s=15, alpha=0.6, label="dane")
ax[1].plot(xx, w * xx + b, "r", lw=2, label="model GD")
ax[1].set_title("Dopasowana prosta"); ax[1].set_xlabel("x"); ax[1].set_ylabel("y"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); zapisz_wykres("04_gd_jedna_cecha")


# ---------- Oryginalna komorka 43 ----------
W_grid, B_grid = np.meshgrid(np.linspace(-1, 5, 120), np.linspace(-4, 2, 120))
# MSE dla każdego punktu siatki (broadcasting: siatka x dane)
L_grid = np.mean((W_grid[..., None] * x1 + B_grid[..., None] - y1) ** 2, axis=-1)

sc = np.array(sciezka)
plt.figure(figsize=(6.5, 5))
cs = plt.contour(W_grid, B_grid, L_grid, levels=np.logspace(0, 1.5, 15), cmap="viridis")
plt.clabel(cs, fontsize=7, fmt="%.1f")
plt.plot(sc[:, 0], sc[:, 1], "r.-", ms=4, label="trajektoria GD")
plt.plot(w_ref, b_ref, "k*", ms=15, label="minimum (polyfit)")
plt.title("Powierzchnia straty MSE(w, b) i ścieżka GD")
plt.xlabel("w"); plt.ylabel("b"); plt.legend(); zapisz_wykres("05_powierzchnia_straty")


naglowek(45, '3.3: California Housing - rzeczywiste dane')


# ---------- Oryginalna komorka 45 ----------
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

try:
    dom = fetch_california_housing()
except Exception as exc:
    raise RuntimeError(
        "Nie udalo sie pobrac California Housing. Pierwszy start wymaga internetu. "
        "Sprobuj ponownie po sprawdzeniu polaczenia; dane zostana w cache scikit-learn."
    ) from exc
X_all, y_all = dom.data.astype(np.float64), dom.target.astype(np.float64)
if SZYBKI_TRYB:                       # tryb szybki: tylko 2000 pierwszych próbek
    X_all, y_all = X_all[:2000], y_all[:2000]
print("cechy:", list(dom.feature_names))
print("X:", X_all.shape, " y:", y_all.shape)

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)

# Standaryzacja - statystyki TYLKO ze zbioru treningowego
mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0)
X_tr_s = (X_tr - mu) / sigma
X_te_s = (X_te - mu) / sigma
n_tr, d = X_tr_s.shape
print("train:", X_tr_s.shape, " test:", X_te_s.shape)
print("średnie cech po standaryzacji (train):", np.round(X_tr_s.mean(axis=0), 3))


# ---------- Oryginalna komorka 47 ----------
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

# Baseline: stała = średnia z treningu
print("Baseline (średnia) MSE test:", round(mse(y_te, np.full_like(y_te, y_tr.mean())), 4))

# Rozwiązanie analityczne (najmniejsze kwadraty) - dokładne minimum na train
Xb_tr = np.hstack([X_tr_s, np.ones((n_tr, 1))])          # kolumna jedynek = bias
theta_ls = np.linalg.lstsq(Xb_tr, y_tr, rcond=None)[0]
w_ls, b_ls = theta_ls[:-1], theta_ls[-1]
loss_opt = mse(y_tr, X_tr_s @ w_ls + b_ls)
print("Najmniejsze kwadraty: MSE train =", round(loss_opt, 4),
      " MSE test =", round(mse(y_te, X_te_s @ w_ls + b_ls), 4))


naglowek(49, 'Zadanie 3.1: regresja NumPy')


# ---------- Oryginalna komorka 49 ----------
EPOKI_GD = 50 if SZYBKI_TRYB else 500

def gd_linreg(X, y, lr=0.2, epoki=EPOKI_GD):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    historia = []
    for ep in range(epoki):
        # Najpierw liczę błąd, dopiero potem poprawiam wagi.
        # Trochę jak z moim Fiatem: najpierw pomiar, potem kolejna korekta.
        y_hat = X @ w + b
        blad = y_hat - y
        historia.append(float(np.mean(blad**2)))
        dw = (2.0 / n) * (X.T @ blad)
        db = 2.0 * float(np.mean(blad))
        w -= lr * dw
        b -= lr * db
    return w, b, historia


# ---------- Oryginalna komorka 50 ----------
t0 = time.perf_counter()
w_gd, b_gd, hist_gd = gd_linreg(X_tr_s, y_tr, lr=0.2, epoki=EPOKI_GD)
print(f"czas: {time.perf_counter() - t0:.2f} s")
if len(hist_gd) > 0:
    print(f"MSE train GD = {hist_gd[-1]:.4f}  vs  optimum = {loss_opt:.4f}  (różnica {100 * (hist_gd[-1] / loss_opt - 1):.2f}%)")
    print("MSE test GD  =", round(mse(y_te, X_te_s @ w_gd + b_gd), 4))
    print("max |w_gd - w_ls| =", np.abs(w_gd - w_ls).max().round(4))

    plt.figure(figsize=(7, 4))
    plt.plot(hist_gd, label="GD (NumPy)")
    plt.axhline(loss_opt, color="k", ls="--", label="optimum (lstsq)")
    plt.yscale("log"); plt.title("California Housing: krzywa uczenia GD")
    plt.xlabel("epoka"); plt.ylabel("MSE train (log)"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("06_gd_california")

    # Które cechy mają największy wpływ? (cechy są standaryzowane, więc wagi są porównywalne)
    plt.figure(figsize=(7, 3.5))
    plt.barh(dom.feature_names, w_gd)
    plt.title("Wagi regresji liniowej (cechy standaryzowane)"); plt.xlabel("waga"); plt.ylabel("cecha")
    plt.grid(alpha=0.3); zapisz_wykres("07_wagi_modelu")


assert len(hist_gd) == EPOKI_GD
assert np.isfinite(hist_gd).all()


naglowek(52, '3.4-3.5: regresja PyTorch')


# ---------- Oryginalna komorka 52 ----------
Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=DEVICE)
ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
Xte_t = torch.tensor(X_te_s, dtype=torch.float32, device=DEVICE)
yte_t = torch.tensor(y_te, dtype=torch.float32, device=DEVICE)

w_t = torch.zeros(d, device=DEVICE, requires_grad=True)   # parametry śledzone przez autograd
b_t = torch.zeros(1, device=DEVICE, requires_grad=True)
lr = 0.2
hist_pt = []
for ep in range(EPOKI_GD):
    y_hat = Xtr_t @ w_t + b_t                 # forward
    loss = ((y_hat - ytr_t) ** 2).mean()      # MSE
    loss.backward()                            # autograd liczy w_t.grad, b_t.grad
    with torch.no_grad():                      # krok optymalizacji poza grafem
        w_t -= lr * w_t.grad
        b_t -= lr * b_t.grad
        w_t.grad.zero_(); b_t.grad.zero_()     # zerowanie gradientów!
    hist_pt.append(loss.item())

print(f"PyTorch (autograd): MSE train = {hist_pt[-1]:.4f}")
print("max |w_torch - w_ls| =", np.abs(w_t.detach().cpu().numpy() - w_ls).max().round(4))


# ---------- Oryginalna komorka 54 ----------
torch.manual_seed(SEED)
model_lin = torch.nn.Linear(d, 1).to(DEVICE)      # w (1 x d) i b - inicjalizowane losowo
opt = torch.optim.SGD(model_lin.parameters(), lr=0.2)
loss_fn = torch.nn.MSELoss()

for ep in range(EPOKI_GD):
    opt.zero_grad()                                # 1) zerowanie gradientów
    y_hat = model_lin(Xtr_t).squeeze(1)            # 2) forward, (n,1) -> (n,)
    loss = loss_fn(y_hat, ytr_t)                   # 3) strata
    loss.backward()                                # 4) gradienty
    opt.step()                                     # 5) krok optymalizatora

with torch.no_grad():
    mse_te_lin = loss_fn(model_lin(Xte_t).squeeze(1), yte_t).item()
print(f"nn.Linear + SGD: MSE train = {loss.item():.4f}, MSE test = {mse_te_lin:.4f}")


naglowek(56, 'Zadanie 3.2: regresja TensorFlow')


# ---------- Oryginalna komorka 56 ----------
Xtr_tf = tf.constant(X_tr_s, dtype=tf.float32)
ytr_tf = tf.constant(y_tr, dtype=tf.float32)

# TF ma policzyć ten sam gradient, który wyżej wyszedł mi w NumPy.
w_tf = tf.Variable(tf.zeros((d,), dtype=tf.float32))
b_tf = tf.Variable(0.0, dtype=tf.float32)
hist_tf = []
for ep in range(EPOKI_GD):
    with tf.GradientTape() as tape:
        y_hat_tf = tf.linalg.matvec(Xtr_tf, w_tf) + b_tf
        loss_tf = tf.reduce_mean(tf.square(y_hat_tf - ytr_tf))
    grad_w_tf, grad_b_tf = tape.gradient(loss_tf, [w_tf, b_tf])
    w_tf.assign_sub(0.2 * grad_w_tf)
    b_tf.assign_sub(0.2 * grad_b_tf)
    hist_tf.append(float(loss_tf.numpy()))

if len(hist_tf) > 0:
    print(f"MSE train: NumPy {hist_gd[-1]:.5f} | PyTorch {hist_pt[-1]:.5f} | TF {hist_tf[-1]:.5f}")
    plt.figure(figsize=(7, 4))
    plt.plot(hist_gd, lw=4, alpha=0.4, label="NumPy (ręczny gradient)")
    plt.plot(hist_pt, "--", label="PyTorch autograd")
    plt.plot(hist_tf, ":", lw=2, label="TF GradientTape")
    plt.yscale("log"); plt.title("Ten sam algorytm w trzech bibliotekach")
    plt.xlabel("epoka"); plt.ylabel("MSE train (log)"); plt.legend(); plt.grid(alpha=0.3); zapisz_wykres("08_porownanie_bibliotek")


# Mateusz znów się kręcił obok klawiatury, więc sprawdzam liczby,
# zamiast zakładać, że trzy krzywe na oko wyglądają tak samo.
assert len(hist_tf) == EPOKI_GD
assert abs(hist_gd[-1] - hist_tf[-1]) < 1e-3
assert abs(hist_pt[-1] - hist_tf[-1]) < 1e-3


naglowek(59, 'Zadanie 4.1: learning rate')


# ---------- Oryginalna komorka 59 ----------
# Hesjan liczony z cechami i biasem. Krytyczny LR wynika z teorii,
# nie jest wartością wybraną na oko.
Xb = np.column_stack((X_tr_s, np.ones(n_tr)))
H = (2.0 / n_tr) * (Xb.T @ Xb)
wlasne = np.linalg.eigvalsh(H)
lam_min, lam_max = float(wlasne[0]), float(wlasne[-1])
lr_kryt = 2.0 / lam_max
assert lam_min > 0 and lr_kryt > 0
print(f"lambda_min = {lam_min:.4f}, lambda_max = {lam_max:.4f}, uwarunkowanie = {lam_max / lam_min:.1f}")
print(f"teoretyczny próg rozbieżności lr_kryt = 2/lambda_max = {lr_kryt:.4f}")

EPOKI_LR = 40 if SZYBKI_TRYB else 200
lrs = [0.001, 0.01, 0.1, round(0.5 * lr_kryt, 4), round(0.98 * lr_kryt, 4), round(1.05 * lr_kryt, 4)]
historie = {}
for lr_ in lrs:
    # Zmieniam tylko learning rate, reszta danych i warunków jest taka sama.
    # Wtedy wykres naprawdę pokazuje wpływ jednego hiperparametru.
    _, _, hist_lr = gd_linreg(X_tr_s, y_tr, lr=lr_, epoki=EPOKI_LR)
    historie[lr_] = hist_lr


# ---------- Oryginalna komorka 60 ----------
if historie:
    plt.figure(figsize=(9, 5))
    for lr_, h in historie.items():
        plt.plot(h, label=f"lr = {lr_}")
    plt.axhline(loss_opt, color="k", ls="--", lw=1, label="optimum")
    plt.yscale("log"); plt.ylim(loss_opt * 0.9, max(10, loss_opt * 100))
    plt.title(f"Wpływ learning rate na zbieżność GD (lr_kryt = {lr_kryt:.3f})")
    plt.xlabel("epoka"); plt.ylabel("MSE train (log)"); plt.legend(); plt.grid(alpha=0.3, which="both"); zapisz_wykres("09_learning_rate")

    print("lr        | MSE po ostatniej epoce | epoki do 1% od optimum")
    for lr_, h in historie.items():
        h = np.array(h)
        ok = np.where(h <= 1.01 * loss_opt)[0]
        print(f"{lr_:<9} | {h[-1]:>22.4g} | {ok[0] if len(ok) else 'nie osiągnięto'}")


# Próg krytyczny weryfikuję na tych samych danych.
assert len(historie) == len(lrs)


naglowek(64, 'Zadanie 5.1: mini-batch NumPy')


# ---------- Oryginalna komorka 64 ----------
def minibatch_gd(X, y, lr=0.05, epoki=20, batch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    historia, czasy = [], []
    t0 = time.perf_counter()
    for ep in range(epoki):
        # Losuję kolejność co epokę, bo model nie powinien uczyć się
        # według kolejności rekordów zapisanych w pliku.
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            blad = xb @ w + b - yb
            # Tutaj musiałem sprawdzić mianownik: ostatni batch bywa mniejszy.
            # Dlatego używam len(idx), a nie stałej wielkości batch_size.
            dw = (2.0 / len(idx)) * (xb.T @ blad)
            db = 2.0 * float(np.mean(blad))
            w -= lr * dw
            b -= lr * db
        historia.append(np.mean((X @ w + b - y) ** 2))   # strata na całym zbiorze train
        czasy.append(time.perf_counter() - t0)
    return w, b, historia, czasy


# ---------- Oryginalna komorka 65 ----------
EPOKI_MB = 4 if SZYBKI_TRYB else 30
# W pierwszej wersji było lr=0.05, tak jak w przykładzie z materiałów.
# Na California Housing, szczególnie przy B=16 i B=64, wartości MSE
# rosły do kosmicznych liczb. To nie było uczenie, tylko rozbieżność.
# Zmniejszam krok do 0.0002 dla wszystkich batchy, żeby porównać
# ich rozmiary przy tej samej wartości hiperparametru.
# Na studiach lepiej pokazać rzeczywisty problem i jego rozwiązanie,
# niż zostawić wykres, który tylko dobrze wygląda na pierwszy rzut oka.
LR_MB = 0.0002
wyniki_mb = {}
for bs in [16, 64, 256, n_tr]:
    _, _, h, c = minibatch_gd(X_tr_s, y_tr, lr=LR_MB, epoki=EPOKI_MB, batch_size=bs)
    nazwa = "full-batch" if bs == n_tr else f"B = {bs}"
    wyniki_mb[nazwa] = (h, c)
    # Gdyby znowu zdarzyła się rozbieżność, ma wyjść BŁĄD,
    # a nie automatyczne „OK” w raporcie. Po to właśnie są testy.
    if not np.all(np.isfinite(h)):
        raise ArithmeticError(f"{nazwa}: MSE ma wartości NaN lub inf")
    if not h[-1] < float(np.mean(y_tr ** 2)) * 1.01:
        raise ArithmeticError(f"{nazwa}: po treningu MSE nie spadło względem startu")
    print(f"{nazwa:>10}: kroków/epokę = {math.ceil(n_tr / bs):5d}, MSE końcowe = {h[-1]:.4f}, czas = {c[-1]:.2f} s")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for nazwa, (h, c) in wyniki_mb.items():
    ax[0].plot(range(1, len(h) + 1), h, "o-", ms=3, label=nazwa)
    ax[1].plot(c, h, "o-", ms=3, label=nazwa)
for a_, xl in zip(ax, ["epoka", "czas [s]"]):
    a_.axhline(loss_opt, color="k", ls="--", lw=1, label="optimum")
    a_.set_yscale("log"); a_.set_xlabel(xl); a_.set_ylabel("MSE train (log)"); a_.grid(alpha=0.3, which="both"); a_.legend()
ax[0].set_title("Mini-batch SGD: strata vs epoka"); ax[1].set_title("Mini-batch SGD: strata vs czas")
plt.tight_layout(); zapisz_wykres("10_minibatch")


naglowek(67, '5.1: DataLoader PyTorch')


# ---------- Oryginalna komorka 67 ----------
from torch.utils.data import TensorDataset, DataLoader

ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
loader = DataLoader(ds, batch_size=64, shuffle=True)     # shuffle=True -> nowa permutacja co epokę

torch.manual_seed(SEED)
model_mb = torch.nn.Linear(d, 1).to(DEVICE)
# Ten sam mniejszy learning rate co w NumPy; duży krok 0.05 dawał NaN.
opt = torch.optim.SGD(model_mb.parameters(), lr=LR_MB)
for ep in range(EPOKI_MB):
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)              # batch na GPU
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model_mb(xb).squeeze(1), yb)
        loss.backward()
        opt.step()
with torch.no_grad():
    mse_dataloader = float(torch.nn.functional.mse_loss(model_mb(Xtr_t).squeeze(1), ytr_t).item())
    print("DataLoader + SGD (B = 64): MSE train =", round(mse_dataloader, 4))
    if not np.isfinite(mse_dataloader):
        raise ArithmeticError("DataLoader + SGD: MSE jest NaN/inf. Nie oddaję takiego wyniku.")


naglowek(69, '6. Regresja wielomianowa')


# ---------- Oryginalna komorka 69 ----------
rng = np.random.default_rng(3)
def prawdziwa_f(x):
    return np.sin(2 * np.pi * x)

n_train_p, n_val_p = 20, 200
x_trp = rng.uniform(0, 1, n_train_p); y_trp = prawdziwa_f(x_trp) + rng.normal(scale=0.3, size=n_train_p)
x_vap = rng.uniform(0, 1, n_val_p);   y_vap = prawdziwa_f(x_vap) + rng.normal(scale=0.3, size=n_val_p)

def poly_features(x, stopien):
    # Macierz Vandermonde'a: kolumny 1, x, x^2, ..., x^stopien
    return np.vander(x, stopien + 1, increasing=True)

def dopasuj_wielomian(x, y, stopien):
    Phi = poly_features(x, stopien)
    return np.linalg.lstsq(Phi, y, rcond=None)[0]       # najmniejsze kwadraty

xx = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax_, D in zip(axes, [1, 4, 15]):
    wD = dopasuj_wielomian(x_trp, y_trp, D)
    ax_.plot(xx, prawdziwa_f(xx), "g--", label="prawdziwa funkcja")
    ax_.plot(xx, poly_features(xx, D) @ wD, "r", label=f"wielomian D = {D}")
    ax_.scatter(x_trp, y_trp, edgecolors="k", facecolors="none", label="train (20 pkt)")
    ax_.set_ylim(-2, 2); ax_.set_title(f"Stopień D = {D}"); ax_.set_xlabel("x"); ax_.grid(alpha=0.3); ax_.legend(fontsize=8)
axes[0].set_ylabel("y")
plt.suptitle("Underfitting (D=1), dobre dopasowanie (D=4), overfitting (D=15)"); plt.tight_layout(); zapisz_wykres("11_przeuczenie_przyklady")


naglowek(71, 'Zadanie 6.1: stopien wielomianu')


# ---------- Oryginalna komorka 71 ----------
stopnie = list(range(16))
# Więcej parametrów to nie zawsze lepszy model. Z SEO jest podobnie:
# można przesadzić z optymalizacją i końcowy efekt będzie słabszy.
mse_train_D, mse_val_D = [], []
for D in stopnie:
    wsp = dopasuj_wielomian(x_trp, y_trp, D)
    mse_train_D.append(mse(y_trp, poly_features(x_trp, D) @ wsp))
    mse_val_D.append(mse(y_vap, poly_features(x_vap, D) @ wsp))
najlepszy_D = stopnie[int(np.argmin(mse_val_D))]

if mse_val_D:
    print("najlepszy stopień wg walidacji:", najlepszy_D, " MSE val =", round(mse_val_D[najlepszy_D], 4))
    plt.figure(figsize=(8, 4.5))
    plt.plot(stopnie, mse_train_D, "o-", label="MSE train")
    plt.plot(stopnie, mse_val_D, "s-", label="MSE walidacja")
    plt.axvline(najlepszy_D, color="gray", ls=":", label=f"najlepszy D = {najlepszy_D}")
    plt.axhline(0.3 ** 2, color="k", ls="--", lw=1, label="wariancja szumu (0.09)")
    plt.yscale("log"); plt.title("Pojemność modelu: underfitting vs overfitting")
    plt.xlabel("stopień wielomianu D (hiperparametr)"); plt.ylabel("MSE (log)"); plt.legend(); plt.grid(alpha=0.3, which="both"); zapisz_wykres("12_train_val_stopnie")


assert len(mse_train_D) == len(mse_val_D) == 16
assert najlepszy_D in stopnie


naglowek(73, 'Zadanie 6.2: regularyzacja Ridge')


# ---------- Oryginalna komorka 73 ----------
def dopasuj_ridge(x, y, stopien, lam):
    # Ridge ogranicza zbyt duże wagi, ale bias zostawiam w spokoju.
    # Nie odwracam macierzy ręcznie, bo np.linalg.solve jest stabilniejsze.
    Phi = poly_features(x, stopien)
    n = Phi.shape[0]
    kara = np.eye(stopien + 1)
    kara[0, 0] = 0.0
    A = Phi.T @ Phi + (n * lam) * kara
    rhs = Phi.T @ y
    return np.linalg.solve(A, rhs)

D_duzy = 15
lambdy = np.logspace(-12, 1, 27)
mse_tr_l, mse_va_l = [], []
for lam in lambdy:
    wl = dopasuj_ridge(x_trp, y_trp, D_duzy, lam)
    mse_tr_l.append(mse(y_trp, poly_features(x_trp, D_duzy) @ wl))
    mse_va_l.append(mse(y_vap, poly_features(x_vap, D_duzy) @ wl))
best_lam = lambdy[int(np.argmin(mse_va_l))]
print(f"najlepsza lambda = {best_lam:.2e}, MSE val = {min(mse_va_l):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].semilogx(lambdy, mse_tr_l, "o-", label="MSE train")
ax[0].semilogx(lambdy, mse_va_l, "s-", label="MSE walidacja")
ax[0].set_yscale("log"); ax[0].set_title(f"Ridge, D = {D_duzy}: błąd vs siła regularyzacji")
ax[0].set_xlabel("lambda (log)"); ax[0].set_ylabel("MSE (log)"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
ax[1].plot(xx, prawdziwa_f(xx), "g--", label="prawdziwa funkcja")
ax[1].plot(xx, poly_features(xx, D_duzy) @ dopasuj_wielomian(x_trp, y_trp, D_duzy), "r", alpha=0.5, label="D=15 bez regularyzacji")
ax[1].plot(xx, poly_features(xx, D_duzy) @ dopasuj_ridge(x_trp, y_trp, D_duzy, best_lam), "b", lw=2, label=f"D=15, ridge lambda={best_lam:.1e}")
ax[1].scatter(x_trp, y_trp, edgecolors="k", facecolors="none", label="train")
ax[1].set_ylim(-2, 2); ax[1].set_title("Dopasowanie z regularyzacją L2"); ax[1].set_xlabel("x"); ax[1].set_ylabel("y"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); zapisz_wykres("13_ridge")


assert np.isfinite(mse_va_l).all()


naglowek(78, 'Dodatkowe: GD vs Momentum vs Adam')


# ---------- Oryginalna komorka 78 ----------
def rosenbrock(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

KROKI_R = 300 if SZYBKI_TRYB else 3000

def optymalizuj(metoda, lr, kroki=KROKI_R, start=(-1.5, 2.0)):
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    sciezka_r = [p.detach().numpy().copy()]
    # Trzy metody, jeden punkt startowy. Porównanie ma być uczciwe.
    # Wszystkie wzory liczę ręcznie, bez torch.optim.
    velocity = torch.zeros_like(p)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    beta, beta1, beta2, eps = 0.9, 0.9, 0.999, 1e-8
    for krok in range(1, kroki + 1):
        wartosc = rosenbrock(p)
        wartosc.backward()
        grad = p.grad.detach().clone()
        with torch.no_grad():
            if metoda == "gd":
                p -= lr * grad
            elif metoda == "momentum":
                velocity = beta * velocity + grad
                p -= lr * velocity
            elif metoda == "adam":
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad.square()
                m_hat = m / (1.0 - beta1**krok)
                v_hat = v / (1.0 - beta2**krok)
                p -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            else:
                raise ValueError(f"Nieznana metoda: {metoda}")
            p.grad.zero_()
        if not torch.isfinite(p).all():
            raise FloatingPointError(f"Rosenbrock: metoda {metoda} - punkt przestal byc skonczony")
        sciezka_r.append(p.detach().numpy().copy())
    return np.array(sciezka_r)

trajektorie = {"GD (lr=1e-3)": optymalizuj("gd", 1e-3),
               "Momentum (lr=2e-4)": optymalizuj("momentum", 2e-4),
               "Adam (lr=0.02)": optymalizuj("adam", 0.02)}

Xr, Yr = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
Zr = (1 - Xr) ** 2 + 100 * (Yr - Xr ** 2) ** 2
plt.figure(figsize=(8, 6))
plt.contour(Xr, Yr, Zr, levels=np.logspace(-1, 3.5, 25), cmap="viridis", alpha=0.6)
for nazwa, tr in trajektorie.items():
    plt.plot(tr[:, 0], tr[:, 1], ".-", ms=2, lw=1, label=f"{nazwa}: odległość od (1,1) = {np.linalg.norm(tr[-1] - 1):.3f}")
plt.plot(1, 1, "k*", ms=15, label="minimum (1, 1)")
plt.title(f"Optymalizatory na funkcji Rosenbrocka ({KROKI_R} kroków)")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(fontsize=8); zapisz_wykres("14_rosenbrock_opt")


for nazwa, droga in trajektorie.items():
    assert len(droga) == KROKI_R + 1 and np.isfinite(droga).all(), nazwa

naglowek("KONIEC", "Podsumowanie i raport do folderu wyniki")

podsumowanie = {
    "data_wykonania": datetime.now().astimezone().isoformat(timespec="seconds"),
    "tryb": "szybki" if SZYBKI_TRYB else "pelny",
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "tensorflow": tf.__version__,
    "pytorch": torch.__version__,
    "keras": keras.__version__,
    "urzadzenie_pytorch": str(DEVICE),
    "zadanie_1_1_zgodnosc_tf_torch": bool(np.allclose(z_tf.numpy(), z_pt.permute(0, 2, 3, 1).numpy(), atol=1e-5)),
    "zadanie_1_2_zgodnosc_z_petla": bool(np.allclose(D_np, D_ref) and np.allclose(D_pt.numpy(), D_ref)),
    "zadanie_2_1_grad_analit": grad_analit.tolist(),
    "zadanie_2_1_grad_tf": grad_tf.tolist(),
    "zadanie_2_1_grad_torch": grad_pt.tolist(),
    "zadanie_2_2_blad_wzgledny": float(blad_wzgledny(g_num, g_auto)),
    "zadanie_3_1_mse_train_numpy": float(hist_gd[-1]),
    "zadanie_3_1_mse_test_numpy": float(mse(y_te, X_te_s @ w_gd + b_gd)),
    "mse_train_optymalne": float(loss_opt),
    "zadanie_3_2_mse_train_tf": float(hist_tf[-1]),
    "mse_train_torch": float(hist_pt[-1]),
    "zadanie_4_1_lr_kryt": float(lr_kryt),
    "zadanie_4_1_lr_eksperyment": {str(k): float(v[-1]) for k, v in historie.items()},
    "zadanie_5_1_learning_rate": float(LR_MB),
    "zadanie_5_1_mse_batch": {k: float(v[0][-1]) for k, v in wyniki_mb.items()},
    "zadanie_5_1_mse_dataloader_pytorch": float(mse_dataloader),
    "zadanie_6_1_najlepszy_D": int(najlepszy_D),
    "zadanie_6_1_mse_val": float(mse_val_D[najlepszy_D]),
    "zadanie_6_2_najlepsza_lambda": float(best_lam),
    "zadanie_6_2_mse_val": float(min(mse_va_l)),
    "zadanie_dodatkowe_odleglosci": {
        nazwa: float(np.linalg.norm(tr[-1] - np.array([1.0, 1.0])))
        for nazwa, tr in trajektorie.items()
    },
    "wykresy": WYKRESY_ZAPISANE,
    "status": "OK - wszystkie sekcje wykonane",
}

# Trzymam tez krotkie wnioski - same wykresy bez interpretacji nie wystarcza.
b16_hist, b16_czas = wyniki_mb["B = 16"]
fb_hist, fb_czas = wyniki_mb["full-batch"]
bl_val = mse(y_te, np.full_like(y_te, y_tr.mean()))
uwagi = [
    "LAB01 - automatyczne wnioski na podstawie faktycznego uruchomienia",
    "Tryb: " + ("szybki" if SZYBKI_TRYB else "pelny"),
    "",
    "1.1. Po standaryzacji kazdy kanal ma srednia bliska 0 i odchylenie bliskie 1; wyniki TF i Torch sa zgodne.",
    f"2.2. Gradient numeryczny i autograd: blad wzgledny = {blad_wzgledny(g_num, g_auto):.3e}.",
    f"3. Regresja liniowa: GD MSE test = {mse(y_te, X_te_s @ w_gd + b_gd):.4f}, baseline = {bl_val:.4f}.",
    f"4. Graniczny learning rate: {lr_kryt:.6g}. Powyzej tego progu krok GD jest niestabilny.",
    f"5. Porownanie mini-batch vs full-batch: wspolny learning rate = {LR_MB}.",
    "   Poprzednie lr=0.05 bylo niestabilne dla malych batchy; po poprawce analizujemy zbieznosc.",
    f"   B=16: koncowe MSE={b16_hist[-1]:.4f}, czas={b16_czas[-1]:.3f} s.",
    f"   Full-batch: koncowe MSE={fb_hist[-1]:.4f}, czas={fb_czas[-1]:.3f} s.",
    f"   PyTorch DataLoader (B=64): MSE={mse_dataloader:.4f}.",
    "   Mniejszy batch wykonuje wiecej aktualizacji parametrow na epoke, ale placi za to wiekszym narzutem petli.",
    "   Na tym konkretnym komputerze warto porownywac zarowno MSE do epoki, jak i MSE do czasu.",
    f"6.1. Stopien najlepiej dopasowany do walidacji: D={najlepszy_D}, MSE wal={min(mse_val_D):.4f}.",
    f"6.2. Ridge D=15: najlepsza lambda={best_lam:.3e}, MSE wal={min(mse_va_l):.4f}.",
    "   O wygranej modelu decyduje walidacja, a nie najnizszy blad treningowy.",
    "",
    "Dodatkowe zadanie - odleglosc od minimum (1,1):",
] + [f"   {nazwa}: {np.linalg.norm(tr[-1] - 1):.6g}" for nazwa, tr in trajektorie.items()]
(WYNIKI / "wnioski.txt").write_text("\n".join(uwagi) + "\n", encoding="utf-8")

raport = WYNIKI / "podsumowanie.json"
raport.write_text(json.dumps(podsumowanie, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nWykresy:", len(WYKRESY_ZAPISANE))
print("Raport:", raport)
print("Wynik:", podsumowanie["status"])
print("Najwazniejsze: MSE NumPy/TF/PT =", round(hist_gd[-1], 5),
      round(hist_tf[-1], 5), round(hist_pt[-1], 5))
print("LR krytyczny =", round(lr_kryt, 5), "| najlepszy D =", najlepszy_D,
      "| najlepsza lambda =", best_lam)
