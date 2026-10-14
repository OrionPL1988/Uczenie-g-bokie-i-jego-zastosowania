# Lab03: Keras - klasyfikacja, regresja, regularyzacja
# Wersja lokalna: jeden venv na caly przedmiot. Kod: Lab03/, wyniki: Lab03/wyniki/.
# Uruchamiaj z katalogu Lab03: ..\Lab02\.venv\Scripts\python.exe lab03_rozwiazanie.py --fast
# Też działa: python lab03_rozwiazanie.py --full (gdy masz aktywne wspólne venv).

import os
import sys
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab03 / Keras - lokalne laboratorium")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fast", action="store_true", help="2-3 epoki, smoke test")
    modes.add_argument("--full", action="store_true", help="pelne cwiczenia (CPU: dlugo)")
    args = parser.parse_args()
    os.environ["DL_FAST"] = "1" if args.fast else "0"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(Path(__file__).resolve().parent)

# Domyslnie TensorFlow. W laboratorium jest wykorzystywany Keras 3.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# Kolejne sekcje tak samo jak w notebooku prowadzacej.



# --------- Komorka 3 ---------

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except ImportError:
    tf = None
import keras
from keras import layers

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix

# Tryb szybki: True = mniej danych i epok (praca na CPU / szybki test kodu)
SZYBKI_TRYB = os.environ.get("DL_FAST", "0") == "1"
# SZYBKI_TRYB = True   # <- odkomentuj, jeśli pracujesz bez GPU

print("TensorFlow:", tf.__version__ if tf is not None else "brak", "| Keras:", keras.__version__)
print("Backend Keras:", keras.backend.backend())
print("SZYBKI_TRYB =", SZYBKI_TRYB)



# --------- Komorka 4 ---------

# Kod do wykresow i raportu. Nic nie wrzucamy do folderu zrodlowego.
import re
import sys
from pathlib import Path

WYNIKI = Path("wyniki")
WYKRESY = WYNIKI / "wykresy"
MODELE = WYNIKI / "modele"
WYKRESY.mkdir(parents=True, exist_ok=True)
MODELE.mkdir(parents=True, exist_ok=True)
NR_WYKRESU = 0
RAPORT = {"tryb": "szybki" if SZYBKI_TRYB else "pełny", "seed": 42}
OSTRZEZENIA = []


def zapisz_wykres():
    global NR_WYKRESU
    NR_WYKRESU += 1
    fig = plt.gcf()
    tytul = fig._suptitle.get_text() if fig._suptitle is not None else ""
    if not tytul:
        tytul = next((a.get_title() for a in fig.axes if a.get_title()), "wykres")
    skrot = re.sub(r"[^a-z0-9]+", "_", tytul.lower()).strip("_")[:55]
    sciezka = WYKRESY / f"{NR_WYKRESU:02d}_{skrot or 'wykres'}.png"
    fig.savefig(sciezka, dpi=125, bbox_inches="tight")
    print(f"[WYKRES] {sciezka}")
    plt.show()
    plt.close(fig)


def sprawdz(warunek, opis):
    # Mamy to w logu czarno na bialym; zero udawania, ze przeszlo.
    print(("[OK] " if warunek else "[UWAGA] ") + opis)
    if not warunek:
        OSTRZEZENIA.append(opis)
    return bool(warunek)



# --------- Komorka 5 ---------

# Sprawdzenie dostępności GPU
gpus = tf.config.list_physical_devices("GPU") if tf is not None else []
if gpus:
    print("Dostępne GPU:", gpus)
else:
    print("Brak GPU - obliczenia na CPU. Rozważ SZYBKI_TRYB = True.")

# Ziarno losowości: ustawia jednocześnie seed dla Pythona, NumPy i TensorFlow.
# Dzięki temu wyniki są (w przybliżeniu) powtarzalne - na GPU drobne różnice i tak mogą wystąpić.
SEED = 42
keras.utils.set_random_seed(SEED)



# --------- Komorka 7 ---------

def rysuj_historie(historia, metryka="accuracy", tytul=""):
    # Rysuje stratę i wybraną metrykę dla zbioru treningowego i walidacyjnego
    h = historia.history if hasattr(historia, "history") else historia
    epoki = np.arange(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epoki, h["loss"], "o-", label="train")
    if "val_loss" in h:
        ax[0].plot(epoki, h["val_loss"], "s-", label="walidacja")
    ax[0].set_xlabel("epoka"); ax[0].set_ylabel("strata"); ax[0].set_title(f"{tytul} - strata")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    if metryka in h:
        ax[1].plot(epoki, h[metryka], "o-", label="train")
        if "val_" + metryka in h:
            ax[1].plot(epoki, h["val_" + metryka], "s-", label="walidacja")
        ax[1].set_xlabel("epoka"); ax[1].set_ylabel(metryka); ax[1].set_title(f"{tytul} - {metryka}")
        ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); zapisz_wykres()



# --------- Komorka 9 ---------

(x_tr_full, y_tr_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

if SZYBKI_TRYB:
    # w trybie szybkim bierzemy tylko początkowe próbki
    x_tr_full, y_tr_full = x_tr_full[:2000], y_tr_full[:2000]
    x_test, y_test = x_test[:500], y_test[:500]

# Normalizacja do [0, 1]: sieci uczą się znacznie stabilniej na małych wartościach niż na 0..255
x_tr_full = x_tr_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Wydzielamy walidację: ostatnia 1/6 części treningowej (w pełnym trybie 10 000 obrazów)
n_val = len(x_tr_full) // 6
x_train, y_train = x_tr_full[:-n_val], y_tr_full[:-n_val]
x_val, y_val = x_tr_full[-n_val:], y_tr_full[-n_val:]

NAZWY_KLAS = ["T-shirt/top", "Spodnie", "Sweter", "Sukienka", "Płaszcz",
              "Sandał", "Koszula", "Sneaker", "Torba", "Botek"]

print("train:", x_train.shape, "val:", x_val.shape, "test:", x_test.shape)
print("zakres wartości:", x_train.min(), "-", x_train.max())
print("liczność klas w train:", np.bincount(y_train, minlength=10))



# --------- Komorka 10 ---------

# Podgląd kilku obrazów z etykietami
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(NAZWY_KLAS[y_train[i]], fontsize=9)
    ax.axis("off")
plt.suptitle("Przykładowe obrazy Fashion-MNIST (zbiór treningowy)")
plt.tight_layout(); zapisz_wykres()



# --------- Komorka 12 ---------

# Model demonstracyjny: najprostszy MLP z jedną warstwą ukrytą
model_demo = keras.Sequential([
    keras.Input(shape=(28, 28)),            # deklaracja kształtu wejścia (bez wymiaru batcha)
    layers.Flatten(),                       # 28x28 -> wektor 784
    layers.Dense(128, activation="relu"),   # warstwa ukryta
    layers.Dense(10, activation="softmax"), # 10 klas -> rozkład prawdopodobieństwa
], name="mlp_demo")

model_demo.summary()   # sprawdź: 100 480 + 1 290 = 101 770 parametrów



# --------- Komorka 14 ---------

EPOKI_DEMO = 2 if SZYBKI_TRYB else 10

model_demo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

t0 = time.time()
hist_demo = model_demo.fit(
    x_train, y_train,
    epochs=EPOKI_DEMO,
    batch_size=128,
    validation_data=(x_val, y_val),   # walidacja liczona po każdej epoce
    verbose=2,                        # jedna linia na epokę
)
print(f"czas treningu: {time.time() - t0:.1f} s")
rysuj_historie(hist_demo, tytul="MLP demo")



# --------- Komorka 17 ---------

EPOKI_11 = 2 if SZYBKI_TRYB else 15
# 1.1. Dokladnie architektura z polecenia. Na wyjsciu softmax, bez kombinacji.
keras.utils.set_random_seed(SEED)
model_mlp = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="mlp_fashion_256_128")
model_mlp.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
hist_mlp = model_mlp.fit(
    x_train, y_train, epochs=EPOKI_11, batch_size=128,
    validation_data=(x_val, y_val), verbose=2,
)
rysuj_historie(hist_mlp, tytul="1.1 MLP 256-128")
loss_test_mlp, acc_test_mlp = model_mlp.evaluate(x_test, y_test, verbose=0)
liczba_param_recznie = (784*256+256) + (256*128+128) + (128*10+10)
print(f"MLP - val acc={hist_mlp.history['val_accuracy'][-1]:.3f}, test acc={acc_test_mlp:.3f}")
print("parametry ręcznie / Keras:", liczba_param_recznie, model_mlp.count_params())
sprawdz(liczba_param_recznie == model_mlp.count_params(), "1.1: liczba parametrow zgodna")
RAPORT["zad_1_1"] = {
    "val_accuracy_last": float(hist_mlp.history["val_accuracy"][-1]),
    "test_accuracy": float(acc_test_mlp), "test_loss": float(loss_test_mlp),
    "parametry": int(model_mlp.count_params()),
}




# --------- Komorka 19 ---------

# Demonstracja na model_demo: predykcja i macierz pomyłek
proby = model_demo.predict(x_test, verbose=0)     # (n, 10) prawdopodobieństwa
y_pred_demo = proby.argmax(axis=1)                # klasa o największym prawdopodobieństwie
print("suma prawdopodobieństw w 1. wierszu:", proby[0].sum().round(4))  # softmax -> 1.0

cm = confusion_matrix(y_test, y_pred_demo, labels=range(10))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(NAZWY_KLAS, rotation=60, ha="right")
ax.set_yticks(range(10)); ax.set_yticklabels(NAZWY_KLAS)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("klasa przewidziana"); ax.set_ylabel("klasa prawdziwa")
ax.set_title("Macierz pomyłek - MLP demo (test)")
plt.colorbar(im, ax=ax, label="liczba próbek"); plt.tight_layout(); zapisz_wykres()



# --------- Komorka 21 ---------

# 1.2. Prawdopodobienstwa -> indeks klasy. Bez sumowania po wierszach.
p_mlp = model_mlp.predict(x_test, verbose=0)
y_pred_mlp = np.argmax(p_mlp, axis=1)
cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=range(10))
acc_klas = np.diag(cm_mlp) / np.maximum(cm_mlp.sum(axis=1), 1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(NAZWY_KLAS, acc_klas)
ax.set_ylim(0, 1.0)
ax.tick_params(axis="x", labelrotation=45)
ax.set_ylabel("dokładność klasy")
ax.set_xlabel("klasa")
ax.set_title("1.2 Dokładność na klasę - MLP")
ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); zapisz_wykres()

bez_przekatnej = cm_mlp.copy()
np.fill_diagonal(bez_przekatnej, 0)
klasa_prawda, klasa_mylona = np.unravel_index(np.argmax(bez_przekatnej), bez_przekatnej.shape)
print("Najczesciej mylona para:", NAZWY_KLAS[klasa_prawda], "->", NAZWY_KLAS[klasa_mylona],
      "liczba:", bez_przekatnej[klasa_prawda, klasa_mylona])

# Tu celowo wybieram bledy. Nie tylko ladne przyklady.
bledne_indeksy = np.flatnonzero(y_pred_mlp != y_test)
fig, axes = plt.subplots(2, 5, figsize=(13, 6))
for ax, idx in zip(axes.flat, bledne_indeksy[:10]):
    ax.imshow(x_test[idx], cmap="gray")
    conf = p_mlp[idx, y_pred_mlp[idx]]
    ax.set_title(f"{NAZWY_KLAS[y_test[idx]]} / {NAZWY_KLAS[y_pred_mlp[idx]]}\n({conf:.0%})", fontsize=8)
    ax.axis("off")
for ax in list(axes.flat)[min(10, len(bledne_indeksy)):]:
    ax.axis("off")
fig.suptitle("1.2 Dziesięć błędnych predykcji: prawda / model")
plt.tight_layout(); zapisz_wykres()
RAPORT["zad_1_2"] = {
    "accuracies_per_class": {NAZWY_KLAS[i]: round(float(a), 4) for i, a in enumerate(acc_klas)},
    "most_confused_pair": [NAZWY_KLAS[klasa_prawda], NAZWY_KLAS[klasa_mylona]],
    "most_confused_count": int(bez_przekatnej[klasa_prawda, klasa_mylona]),
}




# --------- Komorka 23 ---------

dane = fetch_california_housing(as_frame=True)
df = dane.frame
if SZYBKI_TRYB:
    df = df.iloc[:2000]
print(df.shape)
df.describe().T.round(2)



# --------- Komorka 25 ---------

X = df[dane.feature_names].values.astype("float32")
y = df["MedHouseVal"].values.astype("float32")

# Podział 60 / 20 / 20: najpierw odcinamy test, potem z reszty walidację
X_tmp, X_te_r, y_tmp, y_te_r = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_tr_r, X_va_r, y_tr_r, y_va_r = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=SEED)

scaler = StandardScaler().fit(X_tr_r)       # fit TYLKO na train
X_tr_s = scaler.transform(X_tr_r)           # transform na wszystkich podzbiorach
X_va_s = scaler.transform(X_va_r)
X_te_s = scaler.transform(X_te_r)

print("train/val/test:", X_tr_s.shape, X_va_s.shape, X_te_s.shape)
print("średnie cech po standaryzacji (train):", X_tr_s.mean(axis=0).round(3))
print("średnie cech po standaryzacji (test): ", X_te_s.mean(axis=0).round(3), "<- nie muszą być 0")



# --------- Komorka 27 ---------

# Baseline 0: stała predykcja = średnia celu na zbiorze treningowym
pred_sr = np.full_like(y_te_r, y_tr_r.mean())
mae_sr = mean_absolute_error(y_te_r, pred_sr)
rmse_sr = np.sqrt(mean_squared_error(y_te_r, pred_sr))
print(f"Baseline (średnia): MAE = {mae_sr:.3f}, RMSE = {rmse_sr:.3f}  (jednostka: 100 tys. $)")



# --------- Komorka 29 ---------

# 2.1. Porownanie na tej samej probie testowej - inaczej metryki nic nie mowia.
model_liniowy = LinearRegression()
model_liniowy.fit(X_tr_s, y_tr_r)
pred_lin = model_liniowy.predict(X_te_s)
mae_lin = mean_absolute_error(y_te_r, pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_te_r, pred_lin))
print(f"Średnia: MAE={mae_sr:.3f}, RMSE={rmse_sr:.3f}")
print(f"Linear:  MAE={mae_lin:.3f}, RMSE={rmse_lin:.3f}")
sprawdz(mae_lin < mae_sr, "2.1: regresja liniowa bije baseline ze sredniej")
RAPORT["zad_2_1"] = {"mean_mae": float(mae_sr), "mean_rmse": float(rmse_sr),
                      "linear_mae": float(mae_lin), "linear_rmse": float(rmse_lin)}




# --------- Komorka 31 ---------

EPOKI_22 = 3 if SZYBKI_TRYB else 60
# 2.2. Regresja: ostatnia warstwa MUSI byc liniowa, nie softmax.
keras.utils.set_random_seed(SEED)
model_reg = keras.Sequential([
    keras.Input(shape=(8,)), layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"), layers.Dense(1),
], name="regresja_kalifornia")
model_reg.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
hist_reg = model_reg.fit(X_tr_s, y_tr_r, epochs=EPOKI_22, batch_size=64,
                         validation_data=(X_va_s, y_va_r), verbose=2)
rysuj_historie(hist_reg, metryka="mae", tytul="2.2 Regresja MLP")
pred_nn = model_reg.predict(X_te_s, verbose=0).ravel()
# ravel jest tu wazny. Bez niego broadcasting zrobi macierz N x N.
mae_nn = mean_absolute_error(y_te_r, pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_te_r, pred_nn))
tabela_reg = pd.DataFrame([
    {"model": "średnia", "MAE": mae_sr, "RMSE": rmse_sr},
    {"model": "liniowy", "MAE": mae_lin, "RMSE": rmse_lin},
    {"model": "MLP 64-64", "MAE": mae_nn, "RMSE": rmse_nn},
])
print(tabela_reg.round(4).to_string(index=False))
tabela_reg.to_csv(WYNIKI / "porownanie_regresji.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_te_r, pred_nn, s=9, alpha=0.28)
zakres = [min(float(np.min(y_te_r)), float(np.min(pred_nn))),
          max(float(np.max(y_te_r)), float(np.max(pred_nn)))]
ax.plot(zakres, zakres, "k--", label="idealnie: y = x")
ax.set(xlabel="wartość rzeczywista (100 tys. USD)",
       ylabel="predykcja (100 tys. USD)", title="2.2 California Housing: y vs pred")
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout(); zapisz_wykres()
RAPORT["zad_2_2"] = {"mae_nn": float(mae_nn), "rmse_nn": float(rmse_nn),
                       "lepszy_od_liniowego": bool(mae_nn < mae_lin)}




# --------- Komorka 33 ---------

# Demonstracja: EarlyStopping na klasyfikacji Fashion-MNIST z dużą maksymalną liczbą epok
EPOKI_MAX_DEMO = 3 if SZYBKI_TRYB else 50

wczesne_zatrzymanie = keras.callbacks.EarlyStopping(
    monitor="val_loss",          # obserwujemy stratę walidacyjną
    patience=3,                  # 3 epoki bez poprawy -> stop
    restore_best_weights=True,   # przywróć wagi z najlepszej epoki
    verbose=1,
)

model_es = keras.Sequential([keras.Input(shape=(28, 28)), layers.Flatten(),
                             layers.Dense(256, activation="relu"),
                             layers.Dense(10, activation="softmax")])
model_es.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist_es = model_es.fit(x_train, y_train, epochs=EPOKI_MAX_DEMO, batch_size=128,
                       validation_data=(x_val, y_val), callbacks=[wczesne_zatrzymanie], verbose=0)

n_epok = len(hist_es.history["loss"])
najlepsza = int(np.argmin(hist_es.history["val_loss"])) + 1
print(f"Wykonano {n_epok} z {EPOKI_MAX_DEMO} epok; najlepsza epoka (min val_loss): {najlepsza}")
rysuj_historie(hist_es, tytul="EarlyStopping")



# --------- Komorka 35 ---------

EPOKI_MAX_31 = 3 if SZYBKI_TRYB else 200
# 3.1. Trzy callbacki i nowy model. Checkpoint do wynikow, nie do repo.
keras.utils.set_random_seed(SEED)
model_cb = keras.Sequential([
    keras.Input(shape=(8,)), layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"), layers.Dense(1),
], name="regresja_callbacki")
model_cb.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
sciezka_best = MODELE / "najlepszy_reg.keras"
cb_es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
cb_checkpoint = keras.callbacks.ModelCheckpoint(
    str(sciezka_best), monitor="val_loss", save_best_only=True,
)
cb_reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5,
)
hist_cb = model_cb.fit(X_tr_s, y_tr_r, epochs=EPOKI_MAX_31, batch_size=64,
                       validation_data=(X_va_s, y_va_r),
                       callbacks=[cb_es, cb_checkpoint, cb_reduce], verbose=2)
liczba_epok_cb = len(hist_cb.history["loss"])
najlepsza_epoka_cb = int(np.argmin(hist_cb.history["val_loss"])) + 1
print("Callbacki: wykonano", liczba_epok_cb, "epok; najlepsza:", najlepsza_epoka_cb)
lr_hist = hist_cb.history.get("learning_rate", [])
if len(lr_hist) == 0:
    # Starszy Keras czasem nie dodaje tego klucza; uczciwie zostawiam info.
    OSTRZEZENIA.append("Keras nie zapisal learning_rate w historii; brak krzywej lr")
else:
    lr_hist = np.asarray(lr_hist, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(lr_hist) + 1), lr_hist, "o-")
    ax.set(xlabel="epoka", ylabel="learning rate", title="3.1 ReduceLROnPlateau: przebieg lr")
    ax.grid(alpha=0.25)
    plt.tight_layout(); zapisz_wykres()
model_najlepszy = keras.models.load_model(sciezka_best)
pred_best = model_najlepszy.predict(X_te_s, verbose=0).ravel()
mae_cb = mean_absolute_error(y_te_r, pred_best)
print(f"Checkpoint: test MAE={mae_cb:.4f} (zwykly MLP={mae_nn:.4f})")
sprawdz(sciezka_best.is_file(), "3.1: checkpoint zapisany")
RAPORT["zad_3_1"] = {"epochs": liczba_epok_cb, "best_epoch": najlepsza_epoka_cb,
                       "early_stopped": bool(liczba_epok_cb < EPOKI_MAX_31),
                       "lr_min": float(np.min(lr_hist)) if len(lr_hist) else None,
                       "lr_was_reduced": bool(np.min(lr_hist) < np.max(lr_hist)) if len(lr_hist) else False,
                       "loaded_model_test_mae": float(mae_cb),
                       "checkpoint_exists": bool(sciezka_best.is_file())}




# --------- Komorka 38 ---------

EPOKI_OPT = 2 if SZYBKI_TRYB else 10

def zbuduj_mlp_fashion():
    # Ten sam MLP za każdym razem; seed gwarantuje identyczną inicjalizację wag
    keras.utils.set_random_seed(SEED)
    return keras.Sequential([keras.Input(shape=(28, 28)), layers.Flatten(),
                             layers.Dense(128, activation="relu"),
                             layers.Dense(10, activation="softmax")])

def trenuj_z_optymalizatorem(optymalizator, epoki=EPOKI_OPT):
    # Trenuje świeży model z podanym optymalizatorem i zwraca słownik historii
    model = zbuduj_mlp_fashion()
    model.compile(optimizer=optymalizator, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=epoki, batch_size=128,
                  validation_data=(x_val, y_val), verbose=0)
    return h.history

# Demonstracja: zwykły SGD vs Adam
wyniki_demo = {
    "SGD lr=0.01": trenuj_z_optymalizatorem(keras.optimizers.SGD(learning_rate=0.01)),
    "Adam lr=0.001": trenuj_z_optymalizatorem(keras.optimizers.Adam(learning_rate=0.001)),
}
plt.figure(figsize=(7, 4))
for nazwa, h in wyniki_demo.items():
    plt.plot(np.arange(1, len(h["val_loss"]) + 1), h["val_loss"], "o-", label=nazwa)
plt.xlabel("epoka"); plt.ylabel("val_loss"); plt.title("SGD vs Adam - strata walidacyjna")
plt.legend(); plt.grid(alpha=0.3); zapisz_wykres()



# --------- Komorka 40 ---------

# 4.1. Osobny optymalizator za kazdym razem; Adam ma swoj stan i tyle.
konfiguracje = {
    "SGD": lambda: keras.optimizers.SGD(learning_rate=0.01),
    "SGD+momentum": lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": lambda: keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam": lambda: keras.optimizers.Adam(learning_rate=0.001),
}
wyniki_opt = {nazwa: trenuj_z_optymalizatorem(f()) for nazwa, f in konfiguracje.items()}
for pole, tytul in [("val_loss", "4.1 Optymalizatory: val_loss"),
                   ("val_accuracy", "4.1 Optymalizatory: val_accuracy")]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for nazwa, h in wyniki_opt.items():
        ax.plot(np.arange(1, len(h[pole])+1), h[pole], "o-", label=nazwa)
    ax.set(xlabel="epoka", ylabel=pole, title=tytul)
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout(); zapisz_wykres()

tabela_opt = pd.DataFrame([
    {"optymalizator": nazwa,
     "best_val_accuracy": float(max(h["val_accuracy"])),
     "best_epoch": int(np.argmax(h["val_accuracy"])) + 1}
    for nazwa, h in wyniki_opt.items()
]).sort_values("best_val_accuracy", ascending=False)
print(tabela_opt.to_string(index=False))
tabela_opt.to_csv(WYNIKI / "optymalizatory.csv", index=False)
RAPORT["zad_4_1"] = tabela_opt.to_dict(orient="records")




# --------- Komorka 42 ---------

# 4.2. Na LR 0.1 wynik moze byc kiepski - to czesc eksperymentu.
wartosci_lr = [1e-4, 1e-3, 1e-2, 1e-1]
wyniki_lr = {}
for lr in wartosci_lr:
    print("Adam LR:", lr)
    wyniki_lr[lr] = trenuj_z_optymalizatorem(
        keras.optimizers.Adam(learning_rate=lr))
fig, ax = plt.subplots(figsize=(9, 5))
for lr, h in wyniki_lr.items():
    ax.plot(np.arange(1, len(h["loss"])+1), h["loss"], "o-", label=f"lr={lr:g}")
ax.set(xlabel="epoka", ylabel="loss (trening)", title="4.2 Adam - wpływ learning rate")
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout(); zapisz_wykres()
RAPORT["zad_4_2"] = {
    f"{lr:g}": {
        "loss_last": float(h["loss"][-1]) if np.isfinite(h["loss"][-1]) else None,
        "loss_min": float(np.nanmin(h["loss"])) if np.isfinite(h["loss"]).any() else None,
    } for lr, h in wyniki_lr.items()
}
if RAPORT["zad_4_2"]["0.1"]["loss_last"] is None:
    print("LR=0.1 -> NaN/inf: to tez wynik. Spisuje zamiast ukrywac.")
print("LR = 1e-4: typowo wolne schodzenie z loss. LR = 0.1: sprawdzac skoki/straty,")
print("nie zakladam z gory rozbieznosci - patrz konkretne krzywe.")




# --------- Komorka 45 ---------

# Mały zbiór treningowy - ułatwia przeuczenie
N_MALY = 500 if SZYBKI_TRYB else 1000
x_maly, y_maly = x_train[:N_MALY], y_train[:N_MALY]
EPOKI_REG = 3 if SZYBKI_TRYB else 60

def zbuduj_przeuczony():
    # Duży MLP bez regularyzacji: 784 -> 512 -> 512 -> 512 -> 10
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([keras.Input(shape=(28, 28)), layers.Flatten()])
    for _ in range(3):
        model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    return model

def trenuj_reg(model, epoki=EPOKI_REG):
    # Wspólna procedura treningu dla wszystkich wariantów regularyzacji
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(x_maly, y_maly, epochs=epoki, batch_size=64,
                     validation_data=(x_val, y_val), verbose=0).history

model_over = zbuduj_przeuczony()
print("parametry:", model_over.count_params())
h_over = trenuj_reg(model_over)
rysuj_historie(h_over, tytul="Bez regularyzacji (mały zbiór)")
print(f"train acc: {h_over['accuracy'][-1]:.3f}, val acc: {h_over['val_accuracy'][-1]:.3f}, "
      f"min val_loss w epoce {int(np.argmin(h_over['val_loss'])) + 1}")



# --------- Komorka 47 ---------

# 5.1. Ten sam duzy MLP, tylko dokladamy regularyzacje.
def zbuduj_regularyzowany(dropout=0.0, l2=0.0, batchnorm=False):
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([keras.Input(shape=(28, 28)), layers.Flatten()])
    reg = keras.regularizers.l2(l2) if l2 > 0 else None
    for _ in range(3):
        if batchnorm:
            model.add(layers.Dense(512, activation=None, kernel_regularizer=reg))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation("relu"))
        else:
            model.add(layers.Dense(512, activation="relu", kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation="softmax"))
    return model

h_drop = trenuj_reg(zbuduj_regularyzowany(dropout=0.5))
h_l2 = trenuj_reg(zbuduj_regularyzowany(l2=1e-3))
RAPORT["zad_5_1"] = {"dropout_val_acc_last": float(h_drop["val_accuracy"][-1]),
                      "l2_val_acc_last": float(h_l2["val_accuracy"][-1])}




# --------- Komorka 50 ---------

# 5.2. Hipoteza przed trenowaniem jest nizej w notebooku. Tutaj juz same pomiary.
h_bn = trenuj_reg(zbuduj_regularyzowany(batchnorm=True))
h_bn_drop = trenuj_reg(zbuduj_regularyzowany(batchnorm=True, dropout=0.3))
warianty_reg = {
    "brak": h_over, "dropout_0.5": h_drop, "L2_1e-3": h_l2,
    "batchnorm": h_bn, "BN+dropout_0.3": h_bn_drop,
}
fig, ax = plt.subplots(figsize=(10, 5))
for nazwa, h in warianty_reg.items():
    ax.plot(range(1, len(h["val_loss"])+1), h["val_loss"], label=nazwa)
ax.set(xlabel="epoka", ylabel="val_loss", title="5.2 Regularyzacja - porównanie 5 wariantów")
ax.grid(alpha=0.25); ax.legend()
plt.tight_layout(); zapisz_wykres()
tabela_regularyzacji = pd.DataFrame([
    {"wariant": nazwa,
     "best_val_accuracy": float(max(h["val_accuracy"])),
     "last_val_accuracy": float(h["val_accuracy"][-1]),
     "last_train_minus_val": float(h["accuracy"][-1] - h["val_accuracy"][-1]),
     "val_loss_min": float(min(h["val_loss"])),
     "val_loss_last": float(h["val_loss"][-1])}
    for nazwa, h in warianty_reg.items()
])
print(tabela_regularyzacji.round(4).to_string(index=False))
tabela_regularyzacji.to_csv(WYNIKI / "regularyzacja.csv", index=False)
RAPORT["zad_5_2"] = tabela_regularyzacji.to_dict(orient="records")
# Porownuje konkrety, a nie to co chcialem zobaczyc przed testem.
print("Hipoteza BN vs Dropout: weryfikacja w tabeli powyzej (last_train_minus_val).")




# --------- Komorka 53 ---------

# Demonstracja: zapis i wczytanie pełnego modelu
model_demo.save(str(MODELE / "model_demo.keras"))
wczytany = keras.models.load_model(str(MODELE / "model_demo.keras"))

p1 = model_demo.predict(x_test[:100], verbose=0)
p2 = wczytany.predict(x_test[:100], verbose=0)
print("Predykcje identyczne:", np.allclose(p1, p2, atol=1e-6))
print("Rozmiar pliku: %.1f kB" % (os.path.getsize(str(MODELE / "model_demo.keras")) / 1024))
print("Optymalizator wczytanego modelu:", type(wczytany.optimizer).__name__)



# --------- Komorka 55 ---------

# 6.1. Wagi i caly model to dwie rozne rzeczy. Sprawdzam obie.
model_do_zapisu = zbuduj_regularyzowany(dropout=0.5)
h_przed = trenuj_reg(model_do_zapisu, epoki=2)
sciezka_wag = MODELE / "wagi_dropout.weights.h5"
sciezka_model = MODELE / "model_dropout.keras"
model_do_zapisu.save_weights(sciezka_wag)
model_po_wagach = zbuduj_regularyzowany(dropout=0.5)
model_po_wagach.load_weights(sciezka_wag)
p_wagi_1 = model_do_zapisu.predict(x_test[:100], verbose=0)
p_wagi_2 = model_po_wagach.predict(x_test[:100], verbose=0)
wagi_zgodne = bool(np.allclose(p_wagi_1, p_wagi_2, atol=1e-6))
sprawdz(wagi_zgodne, "6.1: predykcje po wczytaniu wag takie same")
model_do_zapisu.save(sciezka_model)
model_wznowiony = keras.models.load_model(sciezka_model)
p_wszystko = model_wznowiony.predict(x_test[:100], verbose=0)
model_zgodny = bool(np.allclose(p_wagi_1, p_wszystko, atol=1e-6))
sprawdz(model_zgodny, "6.1: predykcje po wczytaniu pelnego modelu takie same")
h_po = model_wznowiony.fit(x_maly, y_maly, epochs=2, batch_size=64,
                            validation_data=(x_val, y_val), verbose=0)
acc_przed = float(h_przed["val_accuracy"][-1])
acc_po = float(h_po.history["val_accuracy"][-1])
print("Przed wznowieniem:", round(acc_przed, 4), "Po:", round(acc_po, 4))
RAPORT["zad_6_1"] = {"weights_predictions_equal": wagi_zgodne,
                       "full_model_predictions_equal": model_zgodny,
                       "val_acc_before_resume": acc_przed,
                       "val_acc_after_resume": acc_po,
                       "models_saved": [str(sciezka_wag), str(sciezka_model)]}




# --------- Komorka 60 ---------

EPOKI_DOD = 2 if SZYBKI_TRYB else 20
# BONUS. Callback sprawdza walidacje po epoce, a nie na treningu.
class ZatrzymajPrzyDokladnosci(keras.callbacks.Callback):
    def __init__(self, prog):
        super().__init__()
        self.prog = float(prog)
        self.learning_rates = []
        self.zatrzymano = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        obecny_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.learning_rates.append(obecny_lr)
        accuracy = logs.get("val_accuracy")
        if accuracy is not None and accuracy >= self.prog:
            print(f"Próg {self.prog:.2f} osiągnięty: val_acc={accuracy:.3f}, epoka {epoch+1}")
            self.zatrzymano = True
            self.model.stop_training = True

keras.utils.set_random_seed(SEED)
model_dod = keras.Sequential([
    keras.Input(shape=(28, 28)), layers.Flatten(),
    layers.Dense(256, activation="relu"), layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
batch_dod = 128
kroki_epoka = int(np.ceil(len(x_train)/batch_dod))
schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=EPOKI_DOD * kroki_epoka,
)
model_dod.compile(optimizer=keras.optimizers.Adam(learning_rate=schedule),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cb_prog = ZatrzymajPrzyDokladnosci(prog=0.89)
hist_dod = model_dod.fit(x_train, y_train, epochs=EPOKI_DOD, batch_size=batch_dod,
                         validation_data=(x_val, y_val), callbacks=[cb_prog], verbose=2)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(cb_prog.learning_rates)+1), cb_prog.learning_rates, "o-")
ax.set(xlabel="epoka", ylabel="learning rate", title="Bonus: CosineDecay i własny callback")
ax.grid(alpha=0.25)
plt.tight_layout(); zapisz_wykres()
RAPORT["bonus"] = {"zatrzymano_po_progu": cb_prog.zatrzymano,
                   "epochs": len(hist_dod.history["loss"]),
                   "best_val_accuracy": float(max(hist_dod.history["val_accuracy"]))}




# --------- Komorka 61 ---------

# Zapisuje finalne wyniki DOPIERO po wykonaniu wszystkich zadan.
import json
RAPORT["biblioteki"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                         "keras": keras.__version__, "tensorflow": tf.__version__ if tf is not None else "brak",
                         "backend": keras.backend.backend()}
RAPORT["wykresy"] = [str(p).replace("\\", "/") for p in sorted(WYKRESY.glob("*.png"))]
RAPORT["ostrzezenia"] = OSTRZEZENIA
RAPORT["status"] = "OK - brak zgloszonych problemow" if not OSTRZEZENIA else "UWAGA - sprawdz ostrzezenia"
(WYNIKI / "podsumowanie.json").write_text(json.dumps(RAPORT, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

# Kilka wnioskow z FAKTYCZNYCH liczb; nie z samych oczekiwan.
mini = RAPORT["zad_1_1"]
reg = RAPORT["zad_2_2"]
proby = RAPORT["zad_5_2"]
bez = next(z for z in proby if z["wariant"] == "brak")
drop = next(z for z in proby if z["wariant"] == "dropout_0.5")
lr = RAPORT["zad_4_2"]
wnioski = [
    "Lab03 - wyniki lokalne (tryb: " + RAPORT["tryb"] + ")",
    f"1.1 Fashion-MNIST: acc na teście = {mini['test_accuracy']:.3f}.",
    f"1.2 Najwięcej pomyłek: {RAPORT['zad_1_2']['most_confused_pair'][0]} -> {RAPORT['zad_1_2']['most_confused_pair'][1]}.",
    f"2. California Housing: MAE średniej = {mae_sr:.3f}, linear = {mae_lin:.3f}, MLP = {mae_nn:.3f}.",
    f"3. Callbacki: epok {liczba_epok_cb}, najlepsza {najlepsza_epoka_cb}, MAE checkpointu = {mae_cb:.3f}.",
    f"4. Optim: najlepsze val_accuracy = {tabela_opt.iloc[0]['optymalizator']} ({tabela_opt.iloc[0]['best_val_accuracy']:.3f}).",
    "4.2 Adam LR: loss 0.0001 = " + str(lr["0.0001"]["loss_last"]) +
    ", 0.1 = " + str(lr["0.1"]["loss_last"]) + " (None=NaN/inf).",
    f"5. Overfitting: luka train-val bez regular. = {bez['last_train_minus_val']:.3f}, z dropout = {drop['last_train_minus_val']:.3f}.",
    f"6. Zapis: predykcje po wadze identyczne = {wagi_zgodne}; po pelnym modelu = {model_zgodny}.",
    "Usprawnienie na później: augmentacja obrazów Fashion-MNIST lub AdamW z porównaniem walidacji.",
    "UWAGA: dane Fashion-MNIST test tylko do raportu, dobór parametrów na walidacji.",
    "Ostrzezenia: " + ("; ".join(OSTRZEZENIA) if OSTRZEZENIA else "brak"),
]
(WYNIKI / "wnioski.txt").write_text("\n".join(wnioski) + "\n", encoding="utf-8")
print("\n=== LAB03 ZAKONCZONE ===")
print("Wyniki:", WYNIKI.resolve())
print("Liczba zapisanych wykresow:", len(RAPORT["wykresy"]))
print("Status:", RAPORT["status"])
