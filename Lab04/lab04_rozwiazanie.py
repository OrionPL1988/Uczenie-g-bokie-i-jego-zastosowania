#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab04 - CNN i transfer learning. Pracuje na wspólnym środowisku z Lab02.

Przykłady:
  python lab04_rozwiazanie.py --fast      # test przepływu
  python lab04_rozwiazanie.py --balanced  # CPU, próba pośrednia
  python lab04_rozwiazanie.py --full      # liczby epok/danych z notebooka
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Lab04 CNN i transfer learning")
    tryb = parser.add_mutually_exclusive_group(required=True)
    tryb.add_argument("--fast", action="store_true", help="mały test działania")
    tryb.add_argument("--balanced", action="store_true", help="wyniki pośrednie na CPU")
    tryb.add_argument("--full", action="store_true", help="pełny przebieg zgodny z laboratorium")
    args = parser.parse_args()
    os.environ["DL_PROFILE"] = "fast" if args.fast else "balanced" if args.balanced else "full"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("MPLBACKEND", "Agg")  # .py: bez blokujących okienek z wykresami
    # %% [code] cell 3
    import time, json, re
    from pathlib import Path
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import keras
    from keras import layers
    from sklearn.metrics import confusion_matrix, classification_report

    # W notebooku można zmienić domyślne "fast" na "full".
    # W skrypcie .py argument --full ustawi DL_PROFILE automatycznie.
    PROFIL = os.environ.get("DL_PROFILE", "fast").lower()
    if PROFIL not in ("fast", "balanced", "full"):
        raise ValueError("DL_PROFILE musi być fast, balanced lub full")
    SZYBKI_TRYB = PROFIL == "fast"
    ZROWNOWAZONY = PROFIL == "balanced"
    SEED = 42
    keras.utils.set_random_seed(SEED)
    AUTOTUNE = tf.data.AUTOTUNE

    # Nie wrzucam modeli .keras na GitHub, więc zapisują się osobno albo w cache.
    KATALOG_LAB = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    WYNIKI = KATALOG_LAB / "wyniki" if PROFIL == "full" else KATALOG_LAB / "wyniki" / "testy" / PROFIL
    WYKRESY = WYNIKI / "wykresy"
    WYKRESY.mkdir(parents=True, exist_ok=True)
    # Każdy nowy przebieg czyści stare wykresy w swoim profilu.
    for stary in WYKRESY.glob("*.png"):
        stary.unlink()
    for stary in WYNIKI.glob("*.csv"):
        stary.unlink()
    for nazwa_starego in ("wnioski.txt", "classification_report.txt"):
        (WYNIKI / nazwa_starego).unlink(missing_ok=True)
    STAN_WYKRESU = {"numer": 0, "sekcja": "eksperyment"}
    WYKRESY_ZAPISANE = []
    if not hasattr(plt, "_lab04_show_original"):
        plt._lab04_show_original = plt.show
    ORYGINALNE_SHOW = plt._lab04_show_original

    def pokaz_i_zapisz(*args, **kwargs):
        # Przy wielu wykresach w jednej komórce zapisujemy je oddzielnie.
        figury = list(plt.get_fignums())
        for numer_fig in figury:
            fig = plt.figure(numer_fig)
            STAN_WYKRESU["numer"] += 1
            sufiks = re.sub(r"[^a-z0-9_]+", "_", STAN_WYKRESU["sekcja"].lower())
            plik = WYKRESY / f'{STAN_WYKRESU["numer"]:02d}_{sufiks}.png'
            fig.savefig(plik, dpi=145, bbox_inches="tight")
            WYKRESY_ZAPISANE.append(str(plik.relative_to(KATALOG_LAB)).replace("\\", "/"))
        wynik = ORYGINALNE_SHOW(*args, **kwargs)
        for numer_fig in figury:
            plt.close(numer_fig)
        return wynik

    plt.show = pokaz_i_zapisz

    RAPORT = {
        "tryb": {"fast": "szybki", "balanced": "posredni", "full": "pełny"}[PROFIL],
        "profil": PROFIL,
        "seed": SEED,
        "python": __import__("sys").version.split()[0],
        "tensorflow": tf.__version__, "keras": keras.__version__,
        "gpu": [str(x) for x in tf.config.list_physical_devices("GPU")],
        "czas_start": datetime.now().isoformat(timespec="seconds"),
        "ostrzezenia": [], "status": "w trakcie"
    }

    def zapisz_raport():
        # Raport cząstkowy: jeśli pełny trening przerwie się po paru godzinach, metryki nie przepadną.
        RAPORT["wykresy"] = WYKRESY_ZAPISANE
        with (WYNIKI / "podsumowanie.json").open("w", encoding="utf-8") as f:
            json.dump(RAPORT, f, indent=2, ensure_ascii=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x))

    def kontrola_historii(hist, etykieta):
        for nazwa, wartosci in hist.history.items():
            if not np.all(np.isfinite(np.asarray(wartosci, dtype=float))):
                raise RuntimeError(f"Niepoprawne wartości w treningu {etykieta}: {nazwa}")

    print("TensorFlow:", tf.__version__, "| Keras:", keras.__version__)
    print("PROFIL:", PROFIL, "| GPU:", RAPORT["gpu"] or "brak (CPU)")
    zapisz_raport()


    # %% [code] cell 4
    STAN_WYKRESU["sekcja"] = "sekcja_04"
    def rysuj_historie(historia, metryka="accuracy", tytul=""):
        # Rysuje stratę i metrykę (train vs walidacja) z obiektu History lub słownika
        h = historia.history if hasattr(historia, "history") else historia
        ep = np.arange(1, len(h["loss"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, m in enumerate(["loss", metryka]):
            ax[i].plot(ep, h[m], "o-", label="train")
            if "val_" + m in h:
                ax[i].plot(ep, h["val_" + m], "s-", label="walidacja")
            ax[i].set_xlabel("epoka"); ax[i].set_ylabel(m); ax[i].set_title(f"{tytul} - {m}")
            ax[i].legend(); ax[i].grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    def rysuj_macierz_pomylek(y_true, y_pred, nazwy, tytul):
        # Macierz pomyłek z liczbami w komórkach
        cm = confusion_matrix(y_true, y_pred, labels=range(len(nazwy)))
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(nazwy))); ax.set_xticklabels(nazwy, rotation=60, ha="right")
        ax.set_yticks(range(len(nazwy))); ax.set_yticklabels(nazwy)
        for i in range(len(nazwy)):
            for j in range(len(nazwy)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("klasa przewidziana"); ax.set_ylabel("klasa prawdziwa"); ax.set_title(tytul)
        plt.colorbar(im, ax=ax, label="liczba próbek"); plt.tight_layout(); plt.show()
        return cm

    # %% [code] cell 6
    STAN_WYKRESU["sekcja"] = "sekcja_06"
    # Wczytujemy Fashion-MNIST - użyjemy go w sekcjach 1 i 2
    (xf_tr_full, yf_tr_full), (xf_test, yf_test) = keras.datasets.fashion_mnist.load_data()
    if SZYBKI_TRYB:
        xf_tr_full, yf_tr_full = xf_tr_full[:2000], yf_tr_full[:2000]
        xf_test, yf_test = xf_test[:500], yf_test[:500]
    elif ZROWNOWAZONY:
        xf_tr_full, yf_tr_full = xf_tr_full[:12000], yf_tr_full[:12000]
        xf_test, yf_test = xf_test[:2000], yf_test[:2000]
    xf_tr_full = xf_tr_full.astype("float32") / 255.0
    xf_test = xf_test.astype("float32") / 255.0
    NAZWY_FASHION = ["T-shirt/top", "Spodnie", "Sweter", "Sukienka", "Płaszcz",
                     "Sandał", "Koszula", "Sneaker", "Torba", "Botek"]

    # Prosty obraz syntetyczny: jasny kwadrat na ciemnym tle - krawędzie są oczywiste
    kwadrat = np.zeros((28, 28), dtype="float32")
    kwadrat[8:20, 8:20] = 1.0
    obraz_fashion = xf_test[0]       # pierwszy obraz testowy
    print("klasa obrazu testowego:", NAZWY_FASHION[yf_test[0]])

    # %% [code] cell 7
    STAN_WYKRESU["sekcja"] = "sobel"
    def splot2d_prosty(obraz, filtr):
        # Splot (korelacja) bez paddingu, stride = 1: pętla po pozycjach wyjścia
        k = filtr.shape[0]
        h_out = obraz.shape[0] - k + 1          # wzór: n - k + 1
        w_out = obraz.shape[1] - k + 1
        wyjscie = np.zeros((h_out, w_out), dtype="float32")
        for i in range(h_out):
            for j in range(w_out):
                fragment = obraz[i:i + k, j:j + k]          # okno k x k
                wyjscie[i, j] = np.sum(fragment * filtr)    # iloczyn skalarny okna i filtra
        return wyjscie

    # Filtry Sobela: przybliżenie pochodnej obrazu w kierunku x (krawędzie pionowe) i y (poziome)
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")
    SOBEL_Y = SOBEL_X.T

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for r, (nazwa, img) in enumerate([("kwadrat", kwadrat), ("Fashion-MNIST", obraz_fashion)]):
        gx = splot2d_prosty(img, SOBEL_X)
        gy = splot2d_prosty(img, SOBEL_Y)
        modul = np.sqrt(gx ** 2 + gy ** 2)      # moduł gradientu = "siła" krawędzi
        for c, (t, im) in enumerate([("oryginał", img), ("Sobel X", gx), ("Sobel Y", gy), ("moduł gradientu", modul)]):
            h = axes[r, c].imshow(im, cmap="gray" if c in (0, 3) else "RdBu_r")
            axes[r, c].set_title(f"{nazwa}: {t}"); axes[r, c].axis("off")
            plt.colorbar(h, ax=axes[r, c], fraction=0.046)
    plt.suptitle("Filtry Sobela - splot liczony ręcznie (wyjście 26x26 z wejścia 28x28)")
    plt.tight_layout(); plt.show()

    # %% [code] cell 10
    STAN_WYKRESU["sekcja"] = "splot_tf"
    def splot2d(obraz, filtr, stride=1, padding=0):
        obraz, filtr = np.asarray(obraz), np.asarray(filtr)
        if obraz.ndim != 2 or filtr.ndim != 2 or min(stride, padding + 1) <= 0:
            raise ValueError("Oczekuję macierzy 2D, stride>0, padding>=0")
        if stride != int(stride) or padding != int(padding):
            raise ValueError("stride i padding muszą być całkowite")
        kh, kw = filtr.shape
        oh = (obraz.shape[0] + 2 * padding - kh) // stride + 1
        ow = (obraz.shape[1] + 2 * padding - kw) // stride + 1
        if oh <= 0 or ow <= 0:
            raise ValueError("Filtr jest większy niż obraz z paddingiem")
        obraz_pad = np.pad(obraz, padding, mode="constant")
        wynik = np.empty((oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                # Indeksy to już współrzędne po paddingu. Niczego nie cofamy.
                wycinek = obraz_pad[i * stride:i * stride + kh,
                                    j * stride:j * stride + kw]
                wynik[i, j] = np.sum(wycinek * filtr)
        return wynik

    wlasny = splot2d(obraz_fashion, SOBEL_X, stride=2, padding=1)
    tensor = tf.convert_to_tensor(obraz_fashion[None, ..., None], tf.float32)
    jadro = tf.convert_to_tensor(SOBEL_X[:, :, None, None], tf.float32)
    wzorzec = tf.nn.conv2d(tensor, jadro, strides=[1, 2, 2, 1],
                           padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
    roznica_splot = float(np.max(np.abs(wlasny - wzorzec.numpy()[0, :, :, 0])))
    print("1.1 kształt:", wlasny.shape, "| maks. różnica:", f"{roznica_splot:.3e}")
    assert wlasny.shape == (14, 14) and roznica_splot < 1e-5
    RAPORT["zad_1_1"] = {"ksztalt": list(wlasny.shape), "max_roznica_tf": roznica_splot}
    zapisz_raport()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, img, tytul in zip(axes, [obraz_fashion, wlasny, wzorzec.numpy()[0, :, :, 0]],
                              ["obraz 28x28", "NumPy stride=2 / pad=1", "TF conv2d"]):
        ax.imshow(img, cmap="RdBu_r" if "obraz" not in tytul else "gray")
        ax.set_title(tytul); ax.axis("off")
    plt.tight_layout(); plt.show()


    # %% [code] cell 12
    STAN_WYKRESU["sekcja"] = "sekcja_12"
    # Demonstracja: Keras liczy dokładnie według wzoru
    demo = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),   # 32x32x32, 3*3*3*32+32 = 896
        layers.MaxPooling2D(2),                                    # 16x16x32, 0 parametrów
    ])
    demo.summary()

    # %% [code] cell 15
    STAN_WYKRESU["sekcja"] = "sekcja_15"
    def wymiar_wyjscia(n, k, s=1, p=0):
        if s <= 0 or p < 0 or k <= 0:
            raise ValueError("Nieprawidłowy rozmiar, krok albo padding")
        return (n + 2 * p - k) // s + 1

    # Liczę ręcznie; dla "same" i s=2 wynik to ceil(n/2).
    n = wymiar_wyjscia(32, 5, 1, 0)         # 28
    moje = [("conv5_valid", (None, n, n, 16), 5*5*3*16 + 16)]
    n = wymiar_wyjscia(n, 2, 2, 0)       # 14
    moje.append(("pool2", (None, n, n, 16), 0))
    n = int(np.ceil(n / 2))                # 7
    moje.append(("conv3_same", (None, n, n, 32), 3*3*16*32 + 32))
    moje.append(("flatten", (None, n*n*32), 0))
    moje.append(("dense10", (None, 10), n*n*32*10 + 10))

    model_ksztalt = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, 5, strides=1, padding="valid", name="conv5_valid"),
        layers.MaxPooling2D(2, name="pool2"),
        layers.Conv2D(32, 3, strides=2, padding="same", name="conv3_same"),
        layers.Flatten(name="flatten"),
        layers.Dense(10, name="dense10"),
    ], name="sprawdzenie_reczne")

    for (nazwa, ksztalt, parametry), warstwa in zip(moje, model_ksztalt.layers):
        ksztalt_keras = tuple(warstwa.output.shape)
        print(f"{nazwa:14} -> {ksztalt} ({parametry} par.) | Keras {ksztalt_keras} ({warstwa.count_params()})")
        assert ksztalt == ksztalt_keras and parametry == warstwa.count_params()
    assert len(moje) == len(model_ksztalt.layers)
    RAPORT["zad_1_2"] = {"warstwy": [{"nazwa": n, "ksztalt": str(k), "parametry": p}
                                     for n, k, p in moje],
                         "parametry_lacznie": sum(p for _, _, p in moje)}
    zapisz_raport()


    # %% [code] cell 17
    STAN_WYKRESU["sekcja"] = "sekcja_17"
    # Dodajemy oś kanałów: (n, 28, 28) -> (n, 28, 28, 1)
    xf_tr4 = xf_tr_full[..., None]
    xf_te4 = xf_test[..., None]
    n_val_f = len(xf_tr4) // 6
    xf_train, yf_train = xf_tr4[:-n_val_f], yf_tr_full[:-n_val_f]
    xf_val, yf_val = xf_tr4[-n_val_f:], yf_tr_full[-n_val_f:]
    print("train:", xf_train.shape, "val:", xf_val.shape, "test:", xf_te4.shape)

    EPOKI_F = 2 if SZYBKI_TRYB else (5 if ZROWNOWAZONY else 12)

    # Baseline z L3: MLP 256-128
    keras.utils.set_random_seed(SEED)
    mlp_f = keras.Sequential([keras.Input(shape=(28, 28, 1)), layers.Flatten(),
                              layers.Dense(256, activation="relu"), layers.Dense(128, activation="relu"),
                              layers.Dense(10, activation="softmax")], name="mlp_L3")
    mlp_f.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    hist_mlp_f = mlp_f.fit(xf_train, yf_train, epochs=EPOKI_F, batch_size=128,
                           validation_data=(xf_val, yf_val), verbose=2,
                           callbacks=[keras.callbacks.TerminateOnNaN()])
    czas_mlp = time.time() - t0
    acc_mlp_f = mlp_f.evaluate(xf_te4, yf_test, verbose=0)[1]
    print(f"MLP: parametry = {mlp_f.count_params():,}, test acc = {acc_mlp_f:.4f}, czas = {czas_mlp:.1f} s")
    kontrola_historii(hist_mlp_f, "MLP baseline")
    RAPORT["fashion_mlp"] = {"parametry": mlp_f.count_params(), "test_acc": float(acc_mlp_f),
                             "train_n": len(xf_train), "test_n": len(xf_te4)}
    zapisz_raport()


    # %% [code] cell 20
    STAN_WYKRESU["sekcja"] = "fashion_cnn"
    keras.utils.set_random_seed(SEED)
    cnn_f = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dropout(0.3), layers.Dense(10, activation="softmax")
    ], name="cnn_fashion")
    cnn_f.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    hist_cnn_f = cnn_f.fit(xf_train, yf_train, validation_data=(xf_val, yf_val),
                           epochs=EPOKI_F, batch_size=128, verbose=2,
                           callbacks=[keras.callbacks.TerminateOnNaN()])
    czas_cnn = time.time() - t0
    kontrola_historii(hist_cnn_f, "Fashion CNN")
    acc_cnn_f = float(cnn_f.evaluate(xf_te4, yf_test, verbose=0)[1])
    tabela_fashion = pd.DataFrame([
        {"model": "MLP 256-128 (L3)", "parametry": mlp_f.count_params(),
         "test_accuracy": float(acc_mlp_f), "czas_s": round(czas_mlp, 1)},
        {"model": "CNN 32-64 + Flatten", "parametry": cnn_f.count_params(),
         "test_accuracy": acc_cnn_f, "czas_s": round(czas_cnn, 1)}
    ])
    print("\nMLP vs CNN (ten sam zbiór testowy):\n", tabela_fashion.to_string(index=False))
    print("Hipoteza o dokładności:", "potwierdzona" if acc_cnn_f > acc_mlp_f else "niepotwierdzona")
    print("CNN ma", "więcej" if cnn_f.count_params() > mlp_f.count_params() else "mniej", "parametrów.")
    tabela_fashion.to_csv(WYNIKI / "mlp_vs_cnn_fashion.csv", index=False)
    RAPORT["zad_2_1"] = {"mlp_acc": float(acc_mlp_f), "cnn_acc": acc_cnn_f,
                          "mlp_parametry": mlp_f.count_params(), "cnn_parametry": cnn_f.count_params(),
                          "czas_mlp_s": round(czas_mlp, 2), "czas_cnn_s": round(czas_cnn, 2),
                          "hipoteza_cnn_lepsze": bool(acc_cnn_f > acc_mlp_f)}
    zapisz_raport()
    rysuj_historie(hist_cnn_f, tytul="2.1 CNN Fashion-MNIST")


    # %% [code] cell 23
    STAN_WYKRESU["sekcja"] = "cifar_przyklady"
    (xc_tr_full, yc_tr_full), (xc_test, yc_test) = keras.datasets.cifar10.load_data()
    yc_tr_full, yc_test = yc_tr_full.ravel(), yc_test.ravel()   # etykiety mają kształt (n, 1) -> (n,)
    if SZYBKI_TRYB:
        xc_tr_full, yc_tr_full = xc_tr_full[:2000], yc_tr_full[:2000]
        xc_test, yc_test = xc_test[:500], yc_test[:500]
    elif ZROWNOWAZONY:
        xc_tr_full, yc_tr_full = xc_tr_full[:12000], yc_tr_full[:12000]
        xc_test, yc_test = xc_test[:2000], yc_test[:2000]

    # Surowe obrazy uint8 0..255 zachowujemy do transfer learningu (sekcja 6)
    xc_tr_raw, xc_te_raw = xc_tr_full, xc_test

    # Do własnych CNN: skalowanie do [0, 1]
    xc_tr = xc_tr_full.astype("float32") / 255.0
    xc_te = xc_test.astype("float32") / 255.0
    n_val_c = len(xc_tr) // 10
    xc_train, yc_train = xc_tr[:-n_val_c], yc_tr_full[:-n_val_c]
    xc_val, yc_val = xc_tr[-n_val_c:], yc_tr_full[-n_val_c:]

    NAZWY_CIFAR = ["samolot", "samochód", "ptak", "kot", "jeleń", "pies", "żaba", "koń", "statek", "ciężarówka"]
    print("train:", xc_train.shape, "val:", xc_val.shape, "test:", xc_te.shape)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(xc_train[i]); ax.set_title(NAZWY_CIFAR[yc_train[i]], fontsize=9); ax.axis("off")
    plt.suptitle("Przykładowe obrazy CIFAR-10"); plt.tight_layout(); plt.show()

    # %% [code] cell 24
    STAN_WYKRESU["sekcja"] = "cifar_augmentacje"
    # Potok augmentacji jako mały model Sequential
    augmentacja = keras.Sequential([
        layers.RandomFlip("horizontal"),                    # odbicie lewo-prawo
        layers.RandomTranslation(0.1, 0.1),                 # przesunięcie do 10% w pionie i poziomie
        layers.RandomRotation(0.05),                        # obrót do +-5% pełnego kąta (ok. +-18 stopni)
        layers.RandomZoom(0.1),                             # zoom +-10%
    ], name="augmentacja")

    # Podgląd: ten sam obraz po 8 losowych augmentacjach (training=True wymusza losowość)
    obraz = xc_train[:1]
    fig, axes = plt.subplots(1, 9, figsize=(16, 2.4))
    axes[0].imshow(obraz[0]); axes[0].set_title("oryginał"); axes[0].axis("off")
    for ax in axes[1:]:
        aug = augmentacja(obraz, training=True)
        ax.imshow(np.clip(keras.ops.convert_to_numpy(aug)[0], 0, 1)); ax.set_title("augmentacja"); ax.axis("off")
    plt.suptitle(f"Augmentacja obrazu klasy '{NAZWY_CIFAR[yc_train[0]]}'"); plt.tight_layout(); plt.show()

    # %% [code] cell 27
    STAN_WYKRESU["sekcja"] = "cifar_baseline"
    EPOKI_C = 2 if SZYBKI_TRYB else (5 if ZROWNOWAZONY else 15)

    keras.utils.set_random_seed(SEED)
    we = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(we)
    x = layers.MaxPooling2D(2)(x)                                                     # 16x16
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(2)(x)                                                     # 8x8
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(2)(x)                                                     # 4x4
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    wy = layers.Dense(10, activation="softmax")(x)
    cnn_base = keras.Model(we, wy, name="cnn_baseline")
    cnn_base.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("parametry:", f"{cnn_base.count_params():,}")

    t0 = time.time()
    hist_base = cnn_base.fit(xc_train, yc_train, epochs=EPOKI_C, batch_size=128,
                             validation_data=(xc_val, yc_val), verbose=2,
                             callbacks=[keras.callbacks.TerminateOnNaN()])
    acc_base = cnn_base.evaluate(xc_te, yc_test, verbose=0)[1]
    print(f"czas: {time.time() - t0:.0f} s, test acc: {acc_base:.4f}")
    rysuj_historie(hist_base, tytul="CNN baseline (CIFAR-10)")
    kontrola_historii(hist_base, "CNN baseline CIFAR")
    RAPORT["cifar_baseline"] = {"test_acc": float(acc_base), "parametry": cnn_base.count_params(),
                                "train_n": len(xc_train), "test_n": len(xc_te),
                                "best_val_loss_epoch": int(np.argmin(hist_base.history["val_loss"]) + 1)}
    zapisz_raport()


    # %% [code] cell 30
    STAN_WYKRESU["sekcja"] = "cifar_bn_aug"
    EPOKI_C31 = 2 if SZYBKI_TRYB else (8 if ZROWNOWAZONY else 40)
    keras.utils.set_random_seed(SEED)
    we_c = keras.Input(shape=(32, 32, 3))
    x = augmentacja(we_c)
    for i, filtry in enumerate((32, 64, 128)):
        x = layers.Conv2D(filtry, 3, padding="same", use_bias=False,
                          name=f"conv_b{i+1}_1")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filtry, 3, padding="same", use_bias=False,
                          name="ostatni_conv" if i == 2 else f"conv_b{i+1}_2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.2 + i*0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    wy_c = layers.Dense(10, activation="softmax")(x)
    model_cifar = keras.Model(we_c, wy_c, name="cnn_bn_dropout_aug")
    model_cifar.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("3.1 parametry CNN BN+aug:", model_cifar.count_params())
    cb_c = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            keras.callbacks.TerminateOnNaN()]
    t0 = time.time()
    hist_cifar = model_cifar.fit(xc_train, yc_train, validation_data=(xc_val, yc_val),
                                 epochs=EPOKI_C31, batch_size=128, verbose=2, callbacks=cb_c)
    czas_cifar = time.time() - t0
    kontrola_historii(hist_cifar, "CIFAR BN+aug")
    acc_cifar = float(model_cifar.evaluate(xc_te, yc_test, verbose=0)[1])
    print(f"3.1 CIFAR: baseline={acc_base:.3f}, BN+aug={acc_cifar:.3f}; "
          f"epok={len(hist_cifar.history['loss'])}")
    RAPORT["zad_3_1"] = {"baseline_test_acc": float(acc_base),
                          "bn_aug_test_acc": acc_cifar,
                          "bn_aug_parametry": model_cifar.count_params(),
                          "epoki": len(hist_cifar.history["loss"]),
                          "best_val_acc": float(max(hist_cifar.history["val_accuracy"])),
                          "czas_s": round(czas_cifar, 2)}
    zapisz_raport()
    rysuj_historie(hist_cifar, tytul="3.1 CNN BatchNorm + augmentacja")


    # %% [code] cell 33
    STAN_WYKRESU["sekcja"] = "sekcja_33"
    def blok_identycznosciowy(x, filtry):
        # Blok residualny bez zmiany wymiarów: wejście i wyjście mają 'filtry' kanałów
        skrot = x                                                    # gałąź skrótu: identyczność
        y = layers.Conv2D(filtry, 3, padding="same", use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(filtry, 3, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Add()([y, skrot])                                 # F(x) + x
        return layers.Activation("relu")(y)

    # Demonstracja na małym modelu: kształty muszą się zgadzać, inaczej Add zgłosi błąd
    we = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(we)   # 32 kanały
    x = blok_identycznosciowy(x, 32)
    wy = layers.GlobalAveragePooling2D()(x)
    keras.Model(we, wy, name="demo_resblock").summary()

    # Porównanie liczby parametrów głowy: Flatten vs GAP na mapie 8x8x128 i 10 klasach
    print("Flatten + Dense(10):", 8 * 8 * 128 * 10 + 10, "parametrów")
    print("GAP + Dense(10):    ", 128 * 10 + 10, "parametrów")

    # %% [code] cell 35
    STAN_WYKRESU["sekcja"] = "mini_resnet"
    def blok_rezydualny(x, filtry, stride=1):
        skrot = x
        y = layers.Conv2D(filtry, 3, strides=stride, padding="same", use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(filtry, 3, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        if stride != 1 or int(x.shape[-1]) != filtry:
            # Projekcja skrótu przy zmianie kanałów albo rozdzielczości.
            skrot = layers.Conv2D(filtry, 1, strides=stride, use_bias=False)(skrot)
            skrot = layers.BatchNormalization()(skrot)
        y = layers.Add()([y, skrot])
        return layers.Activation("relu")(y)

    keras.utils.set_random_seed(SEED)
    we_r = keras.Input(shape=(32, 32, 3))
    x = augmentacja(we_r)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = blok_rezydualny(x, 32)
    x = blok_rezydualny(x, 64, stride=2)
    x = blok_rezydualny(x, 128, stride=2)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    wy_r = layers.Dense(10, activation="softmax")(x)
    model_res = keras.Model(we_r, wy_r, name="mini_resnet")
    model_res.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    hist_res = model_res.fit(xc_train, yc_train, validation_data=(xc_val, yc_val),
                             epochs=EPOKI_C31, batch_size=128, verbose=2,
                             callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                                                      restore_best_weights=True),
                                        keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3),
                                        keras.callbacks.TerminateOnNaN()])
    czas_res = time.time() - t0
    kontrola_historii(hist_res, "mini-ResNet")
    acc_res = float(model_res.evaluate(xc_te, yc_test, verbose=0)[1])
    tabela_cifar_cnn = pd.DataFrame([
        {"model": "baseline", "parametry": cnn_base.count_params(), "test_accuracy": float(acc_base)},
        {"model": "BN+aug", "parametry": model_cifar.count_params(), "test_accuracy": acc_cifar},
        {"model": "mini-ResNet+GAP", "parametry": model_res.count_params(), "test_accuracy": acc_res}
    ])
    print(tabela_cifar_cnn.to_string(index=False))
    tabela_cifar_cnn.to_csv(WYNIKI / "cnn_cifar_modele.csv", index=False)
    RAPORT["zad_4_1"] = {"resnet_acc": acc_res, "resnet_parametry": model_res.count_params(),
                          "cnn_bn_parametry": model_cifar.count_params(),
                          "resnet_epoki": len(hist_res.history["loss"]),
                          "czas_s": round(czas_res, 2)}
    zapisz_raport()
    rysuj_historie(hist_res, tytul="4.1 Mini-ResNet (GAP)")


    # %% [code] cell 37
    STAN_WYKRESU["sekcja"] = "filtry_conv1"
    # Filtry pierwszej warstwy baseline'u
    W, b = cnn_base.get_layer("conv1").get_weights()   # W: (3, 3, 3, 32)
    print("kształt wag conv1:", W.shape)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
    for i, ax in enumerate(axes.flat):
        f = W[:, :, :, i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)   # skalowanie do [0, 1] dla każdego filtra
        ax.imshow(f, interpolation="nearest"); ax.axis("off"); ax.set_title(str(i), fontsize=7)
    plt.suptitle("Filtry 3x3x3 warstwy conv1 (CNN baseline) jako obrazy RGB"); plt.tight_layout(); plt.show()

    # %% [code] cell 39
    STAN_WYKRESU["sekcja"] = "mapy_conv1"
    # Mapy cech warstwy conv1 dla jednego obrazu testowego
    ekstraktor = keras.Model(inputs=cnn_base.inputs, outputs=cnn_base.get_layer("conv1").output)
    obraz = xc_te[3:4]                                  # batch z jednym obrazem
    mapy = keras.ops.convert_to_numpy(ekstraktor(obraz))[0]   # (32, 32, 32)

    fig, axes = plt.subplots(3, 9, figsize=(15, 5.5))
    axes[0, 0].imshow(obraz[0]); axes[0, 0].set_title(NAZWY_CIFAR[yc_test[3]]); axes[0, 0].axis("off")
    for k, ax in enumerate(axes.flat[1:]):
        ax.imshow(mapy[:, :, k], cmap="viridis"); ax.set_title(f"kanał {k}", fontsize=8); ax.axis("off")
    plt.suptitle("Mapy cech warstwy conv1 (32x32) - jasne = silna aktywacja"); plt.tight_layout(); plt.show()

    # %% [code] cell 42
    STAN_WYKRESU["sekcja"] = "plytkie_glebokie"
    # Ten sam obraz do płytkich i głębokich map; w predykcji aug jest wyłączona.
    warstwa_plytka = model_cifar.get_layer("conv_b1_1")
    warstwa_gleboka = model_cifar.get_layer("ostatni_conv")
    ekstraktor_dwoch = keras.Model(model_cifar.inputs,
                                  [warstwa_plytka.output, warstwa_gleboka.output])
    obraz_demo = xc_te[3:4]
    mapa_ply, mapa_gle = ekstraktor_dwoch(obraz_demo, training=False)
    mapa_ply = keras.ops.convert_to_numpy(mapa_ply)[0]
    mapa_gle = keras.ops.convert_to_numpy(mapa_gle)[0]
    fig, axes = plt.subplots(2, 8, figsize=(15, 4.7))
    for r, (mapy, nazwa) in enumerate([(mapa_ply, "płytkie"), (mapa_gle, "głębokie")]):
        for k in range(8):
            axes[r, k].imshow(mapy[:, :, k], cmap="viridis")
            axes[r, k].set_title(f"{nazwa} {mapy.shape[0]}x{mapy.shape[1]} | {k}", fontsize=8)
            axes[r, k].axis("off")
    plt.suptitle("5.1 Mapy cech: pierwsza i ostatnia warstwa splotowa")
    plt.tight_layout(); plt.show()
    print("Mapy płytkie", mapa_ply.shape, "| głębokie", mapa_gle.shape)
    RAPORT["zad_5_1"] = {"plytkie": list(mapa_ply.shape), "glebokie": list(mapa_gle.shape)}
    zapisz_raport()


    # %% [code] cell 45
    STAN_WYKRESU["sekcja"] = "sekcja_45"
    ROZMIAR = 96                                        # rozdzielczość wejścia MobileNetV2
    N_TL_TRAIN = 300 if SZYBKI_TRYB else (2000 if ZROWNOWAZONY else 10000)
    N_TL_TEST = 100 if SZYBKI_TRYB else (500 if ZROWNOWAZONY else 2000)
    BATCH_TL = 64
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    # Losowy, ale powtarzalny podzbiór (surowe obrazy uint8 0..255!)
    rng = np.random.default_rng(SEED)
    idx_tr = rng.permutation(len(xc_tr_raw))[:N_TL_TRAIN]
    idx_te = rng.permutation(len(xc_te_raw))[:N_TL_TEST]
    x_tl, y_tl = xc_tr_raw[idx_tr], yc_tr_full[idx_tr]
    x_tl_te, y_tl_te = xc_te_raw[idx_te], yc_test[idx_te]
    n_val_tl = len(x_tl) // 10
    x_tl_tr, y_tl_tr = x_tl[:-n_val_tl], y_tl[:-n_val_tl]
    x_tl_va, y_tl_va = x_tl[-n_val_tl:], y_tl[-n_val_tl:]

    def przygotuj(x, y):
        # uint8 32x32 -> float32 96x96 -> skala [-1, 1] wymagana przez MobileNetV2
        x = tf.image.resize(tf.cast(x, tf.float32), (ROZMIAR, ROZMIAR))
        return preprocess_input(x), y

    def zrob_ds(x, y, tasuj=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if tasuj:
            ds = ds.shuffle(len(x), seed=SEED)
        return ds.batch(BATCH_TL).map(przygotuj, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    ds_tl_tr = zrob_ds(x_tl_tr, y_tl_tr, tasuj=True)
    ds_tl_va = zrob_ds(x_tl_va, y_tl_va)
    ds_tl_te = zrob_ds(x_tl_te, y_tl_te)

    xb, yb = next(iter(ds_tl_tr))
    print("batch:", xb.shape, "| zakres po preprocess_input:", float(tf.reduce_min(xb)), "-", float(tf.reduce_max(xb)))
    print("train/val/test:", len(x_tl_tr), len(x_tl_va), len(x_tl_te))

    # %% [code] cell 46
    STAN_WYKRESU["sekcja"] = "sekcja_46"
    # Pretrenowana baza: include_top=False usuwa głowę klasyfikacyjną ImageNet (1000 klas)
    try:
        baza = keras.applications.MobileNetV2(input_shape=(ROZMIAR, ROZMIAR, 3),
                                              include_top=False, weights="imagenet")
    except Exception as exc:
        raise RuntimeError("MobileNetV2 wymaga wag ImageNet. Sprawdź internet / cache Keras; "
                           "nie zastępuj wag losowymi.") from exc
    baza.trainable = False                               # zamrażamy WSZYSTKIE wagi bazy
    print("warstw w bazie:", len(baza.layers))
    print("wyjście bazy:", baza.output.shape)            # (None, 3, 3, 1280) dla 96x96
    print(f"parametry bazy: {baza.count_params():,}, trenowalne: {sum(int(np.prod(w.shape)) for w in baza.trainable_weights)}")

    # %% [code] cell 50
    STAN_WYKRESU["sekcja"] = "mobilenet_zamrozony"
    EPOKI_TL1 = 1 if SZYBKI_TRYB else (2 if ZROWNOWAZONY else 6)
    keras.utils.set_random_seed(SEED)
    we_tl = keras.Input(shape=(ROZMIAR, ROZMIAR, 3))
    x = layers.RandomFlip("horizontal")(we_tl)
    x = baza(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    wy_tl = layers.Dense(10, activation="softmax")(x)
    model_tl = keras.Model(we_tl, wy_tl, name="mobilenetv2_transfer")
    model_tl.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    param_tl_zam = sum(int(np.prod(w.shape)) for w in model_tl.trainable_weights)
    print("6.1 trenowalne parametry:", param_tl_zam)
    assert param_tl_zam == 12810, "Przy zamrożonej bazie powinna trenować się tylko głowa 12810 par."
    t0 = time.time()
    hist_tl1 = model_tl.fit(ds_tl_tr, validation_data=ds_tl_va,
                             epochs=EPOKI_TL1, verbose=2,
                             callbacks=[keras.callbacks.TerminateOnNaN()])
    czas_tl1 = time.time() - t0
    kontrola_historii(hist_tl1, "MobileNet zamrożona")
    acc_tl1 = float(model_tl.evaluate(ds_tl_te, verbose=0)[1])
    print(f"6.1 test accuracy: {acc_tl1:.4f}")
    RAPORT["zad_6_1"] = {"trainable_parametry": param_tl_zam,
                          "test_acc": acc_tl1, "val_acc_best": float(max(hist_tl1.history["val_accuracy"])),
                          "czas_s": round(czas_tl1, 2)}
    zapisz_raport()
    rysuj_historie(hist_tl1, tytul="6.1 MobileNetV2 - zamrożona baza")


    # %% [code] cell 52
    STAN_WYKRESU["sekcja"] = "mobilenet_finetuning"
    EPOKI_TL2 = 1 if SZYBKI_TRYB else (2 if ZROWNOWAZONY else 6)
    N_ODMROZ = 30
    baza.trainable = True
    for warstwa in baza.layers[:-N_ODMROZ]:
        warstwa.trainable = False
    # trainable zmienione -> compile konieczny (inaczej optimizer używa starej listy wag).
    model_tl.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                     loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    param_tl_ft = sum(int(np.prod(w.shape)) for w in model_tl.trainable_weights)
    print("6.2 fine-tuning - trenowalne parametry:", param_tl_ft)
    start_ep = len(hist_tl1.history["loss"])
    t0 = time.time()
    hist_tl2 = model_tl.fit(ds_tl_tr, validation_data=ds_tl_va,
                             initial_epoch=start_ep, epochs=start_ep + EPOKI_TL2,
                             verbose=2, callbacks=[keras.callbacks.TerminateOnNaN()])
    czas_tl2 = time.time() - t0
    kontrola_historii(hist_tl2, "MobileNet fine-tuning")
    acc_tl2 = float(model_tl.evaluate(ds_tl_te, verbose=0)[1])
    print(f"6.2 test accuracy: przed={acc_tl1:.4f}, po={acc_tl2:.4f} "
          f"(zmiana {100*(acc_tl2-acc_tl1):+.2f} p.p.)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for panel, klucz in zip(ax, ("loss", "accuracy")):
        for typ in ("", "val_"):
            wartosci = hist_tl1.history[typ+klucz] + hist_tl2.history[typ+klucz]
            panel.plot(range(1, len(wartosci)+1), wartosci, "o-", label=typ+klucz)
        panel.axvline(start_ep + 0.5, color="black", linestyle="--", label="fine-tuning start")
        panel.set_xlabel("epoka"); panel.set_ylabel(klucz); panel.legend(); panel.grid(alpha=.3)
    plt.suptitle("6.2 MobileNetV2: zamrożona baza -> fine-tuning (połączone historie)")
    plt.tight_layout(); plt.show()
    RAPORT["zad_6_2"] = {"trainable_parametry": param_tl_ft,
                          "test_acc_po": acc_tl2,
                          "roznica_punktow_proc": 100*(acc_tl2-acc_tl1),
                          "czas_s": round(czas_tl2, 2)}
    if acc_tl2 <= acc_tl1:
        RAPORT["ostrzezenia"].append("Fine-tuning nie poprawił dokładności na tym przebiegu; nie wymuszam wniosku o poprawie.")
    zapisz_raport()


    # %% [code] cell 54
    STAN_WYKRESU["sekcja"] = "mobilenet_macierz_pomylek"
    # tf.data dla testu jest bez shuffle, więc etykiety zachowują kolejność.
    pred_tl = np.argmax(model_tl.predict(ds_tl_te, verbose=0), axis=1)
    cm_tl = rysuj_macierz_pomylek(y_tl_te, pred_tl, NAZWY_CIFAR,
                                  "6.3 MobileNetV2: macierz pomyłek (podzbiór CIFAR-10)")
    raport_klas = classification_report(y_tl_te, pred_tl, labels=list(range(10)),
                                        target_names=NAZWY_CIFAR, zero_division=0)
    print(raport_klas)
    (WYNIKI / "classification_report.txt").write_text(raport_klas, encoding="utf-8")

    pary = []
    for i in range(10):
        for j in range(i+1, 10):
            pary.append((int(cm_tl[i, j] + cm_tl[j, i]), NAZWY_CIFAR[i], NAZWY_CIFAR[j],
                         int(cm_tl[i, j]), int(cm_tl[j, i])))
    pary.sort(reverse=True)
    print("Najczęstsze pary (oba kierunki):")
    for ile, a, b, ab, ba in pary[:2]:
        print(f"  {a} <-> {b}: łącznie {ile} (od {a} do {b}: {ab}, odwrotnie: {ba})")

    N_CIFAR_TRAIN = len(xc_train)
    N_CIFAR_TEST = len(xc_te)
    N_TL_UCZENIE = len(x_tl_tr)
    N_TL_TESTOWY = len(x_tl_te)
    model_tabela = pd.DataFrame([
        {"model": "CNN baseline", "parametry": cnn_base.count_params(), "test_accuracy": float(acc_base),
         "train_n": N_CIFAR_TRAIN, "test_n": N_CIFAR_TEST},
        {"model": "CNN BN+augmentacja", "parametry": model_cifar.count_params(), "test_accuracy": acc_cifar,
         "train_n": N_CIFAR_TRAIN, "test_n": N_CIFAR_TEST},
        {"model": "mini-ResNet+GAP", "parametry": model_res.count_params(), "test_accuracy": acc_res,
         "train_n": N_CIFAR_TRAIN, "test_n": N_CIFAR_TEST},
        {"model": "MobileNetV2 zamrożona", "parametry": model_tl.count_params(), "test_accuracy": acc_tl1,
         "train_n": N_TL_UCZENIE, "test_n": N_TL_TESTOWY},
        {"model": "MobileNetV2 fine-tuning", "parametry": model_tl.count_params(), "test_accuracy": acc_tl2,
         "train_n": N_TL_UCZENIE, "test_n": N_TL_TESTOWY},
    ])
    model_tabela.to_csv(WYNIKI / "cifar_podsumowanie_modeli.csv", index=False)
    print("\nKOŃCOWA TABELA (uwaga: różna liczba danych testowych):\n", model_tabela.to_string(index=False))
    RAPORT["zad_6_3"] = {"dwie_pary_pomylek": [{"klasy": [a, b], "razem": liczba}
                                                   for liczba, a, b, _, _ in pary[:2]],
                          "porownanie_test_na_roznych_podzbiorach": True,
                          "test_n_cnn": N_CIFAR_TEST, "test_n_tl": N_TL_TESTOWY}
    zapisz_raport()


    # %% [code] cell 59
    STAN_WYKRESU["sekcja"] = "gradcam"
    def grad_cam(model, obraz, nazwa_warstwy="ostatni_conv"):
        if obraz.ndim == 3:
            obraz = obraz[None, ...]
        podmodel = keras.Model(inputs=model.inputs,
                               outputs=[model.get_layer(nazwa_warstwy).output, model.output])
        tensor_x = tf.convert_to_tensor(obraz, dtype=tf.float32)
        with tf.GradientTape() as tape:
            mapy_cech, prob = podmodel(tensor_x, training=False)
            klasa = tf.argmax(prob[0])
            wynik_klasy = tf.gather(prob[0], klasa)
        gradient = tape.gradient(wynik_klasy, mapy_cech)
        if gradient is None:
            raise RuntimeError("Grad-CAM: brak gradientu do ostatniej warstwy splotowej")
        alpha = tf.reduce_mean(gradient, axis=(1, 2), keepdims=True)
        mapa = tf.reduce_sum(mapy_cech * alpha, axis=-1)[0]
        mapa = tf.nn.relu(mapa)
        mapa = mapa / (tf.reduce_max(mapa) + 1e-8)
        mapa = tf.image.resize(mapa[..., None], (obraz.shape[1], obraz.shape[2]))[..., 0]
        wynik = mapa.numpy()
        assert np.isfinite(wynik).all()
        return wynik, int(klasa.numpy())

    fig, axes = plt.subplots(6, 3, figsize=(10, 15))
    for i in range(6):
        obraz = xc_te[i]
        mapa, klasa_pred = grad_cam(model_cifar, obraz)
        axes[i, 0].imshow(obraz)
        axes[i, 0].set_title(f"prawda: {NAZWY_CIFAR[int(yc_test[i])]} | pred: {NAZWY_CIFAR[klasa_pred]}", fontsize=9)
        axes[i, 1].imshow(mapa, cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title("Grad-CAM", fontsize=9)
        axes[i, 2].imshow(obraz)
        axes[i, 2].imshow(mapa, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[i, 2].set_title("nałożenie", fontsize=9)
        for ax in axes[i]: ax.axis("off")
    plt.suptitle("Bonus: Grad-CAM (6 obrazów CIFAR-10)")
    plt.tight_layout(); plt.show()
    RAPORT["bonus_gradcam"] = {"obrazy": 6, "nazwa_warstwy": "ostatni_conv"}
    zapisz_raport()


    # %% [code] cell 61
    RAPORT["czas_koniec"] = datetime.now().isoformat(timespec="seconds")
    RAPORT["status"] = "OK - wszystkie sekcje przeszły" if not RAPORT["ostrzezenia"] else "ZAKOŃCZONO - sprawdź ostrzeżenia"
    for zad, metryka, prog in [("zad_2_1", "cnn_acc", .91), ("zad_3_1", "bn_aug_test_acc", .80),
                               ("zad_4_1", "resnet_acc", .80)]:
        # Progi oceny prowadzącej dotyczą pełnego trybu. Nie osądzamy po 2 epokach.
        if PROFIL == "full" and float(RAPORT[zad][metryka]) < prog:
            RAPORT["ostrzezenia"].append(f"{zad}: accuracy poniżej progu {prog:.2f}; to wynik eksperymentu, nie błąd kodu")
    if PROFIL == "full" and RAPORT["zad_6_1"]["val_acc_best"] < .75:
        RAPORT["ostrzezenia"].append("6.1: MobileNetV2 val_accuracy poniżej 0.75")
    if PROFIL != "full":
        RAPORT["ostrzezenia"].append("Tryb fast/balanced: nie opisuj tych metryk jako wyników pełnego laboratorium")
    RAPORT["status"] = "OK - pełny przebieg wykonany" if not RAPORT["ostrzezenia"] else "ZAKOŃCZONO - zweryfikuj ostrzeżenia"

    wnioski = [
        "Lab04 - wyniki z lokalnego przebiegu",
        f"Profil: {PROFIL} (tryb pełny tylko dla --full)",
        f"1.1 Splot: różnica NumPy/TF = {RAPORT['zad_1_1']['max_roznica_tf']:.3e}",
        f"1.2 Parametry ręcznie vs Keras: OK, suma = {RAPORT['zad_1_2']['parametry_lacznie']}",
        f"2.1 MLP acc = {acc_mlp_f:.3f} ({mlp_f.count_params()} par.), CNN acc = {acc_cnn_f:.3f} ({cnn_f.count_params()} par.)",
        f"3.1 CIFAR baseline acc = {acc_base:.3f}, CNN BN+aug = {acc_cifar:.3f}",
        f"4.1 Mini-ResNet acc = {acc_res:.3f}, parametry = {model_res.count_params()}",
        f"5.1 Płytkie mapy = {mapa_ply.shape}, głębokie = {mapa_gle.shape}",
        f"6.1 Transfer zamrożony acc = {acc_tl1:.3f}; 6.2 fine-tuning acc = {acc_tl2:.3f}",
        "6.3 W tabeli osobno podano wielkości zbiorów train/test. Transfer używa mniejszego podzbioru testowego.",
        "Najczęściej mylone pary: " + ", ".join(f"{a} i {b} ({c})" for c, a, b, _, _ in pary[:2]),
        "Usprawnienie do rozważenia (NIE testowane): wejście 128x128 lub delikatniejsza augmentacja; wymaga ponownej walidacji.",
        "Ostrzeżenia: " + ("; ".join(RAPORT["ostrzezenia"]) if RAPORT["ostrzezenia"] else "brak"),
    ]
    (WYNIKI / "wnioski.txt").write_text("\n".join(wnioski) + "\n", encoding="utf-8")
    zapisz_raport()
    print("\nKONIEC LAB04:", RAPORT["status"])
    print("Raport:", WYNIKI / "podsumowanie.json")
    print("Wykresy:", len(WYKRESY_ZAPISANE), "|", WYKRESY)


if __name__ == "__main__":
    main()
