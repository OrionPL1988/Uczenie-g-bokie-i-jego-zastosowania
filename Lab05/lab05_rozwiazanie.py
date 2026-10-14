# -*- coding: utf-8 -*-
"""Lab05: PyTorch na danych tabelarycznych; uruchom w VS Code/PowerShell.

Korzystamy z kodu prowadzącej i rozwiązań z notebooka.
Dane nie są podmieniane na syntetyczne poza specjalnie oznaczonymi demonstracjami wycieku.
"""
import os
import sys
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab05 - PyTorch, data leakage, CV i analiza wyników')
    grupa = parser.add_mutually_exclusive_group()
    grupa.add_argument('--fast', action='store_true', help='krótki przebieg techniczny (nie do oceny)')
    grupa.add_argument('--full', action='store_true', help='pełny przebieg do sprawozdania (domyślnie)')
    parser.add_argument('--bonus', action='store_true', help='dodatkowa zagnieżdżona CV na szumie')
    parser.add_argument('--threads', type=int, default=None, help='liczba wątków PyTorch na CPU')
    args = parser.parse_args()
    os.environ['DL_FAST'] = '1' if args.fast else '0'
    os.environ['DL_BONUS'] = '1' if args.bonus else '0'
    if args.threads is not None:
        os.environ['DL_TORCH_THREADS'] = str(args.threads)
    os.environ['MPLBACKEND'] = 'Agg'

# W notebooku __file__ nie istnieje, ale w zwykłym .py wynik zapisujemy obok skryptu.


# ====================
# Sekcja źródłowa notebooka, komórka 04
# ====================
# Importy - wszystkie biblioteki są domyślnie dostępne w Colab (bez pip install)
import os, time, copy, random, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_breast_cancer, fetch_california_housing, make_classification
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold, GroupKFold,
                                     RepeatedStratifiedKFold, cross_val_score, cross_validate,
                                     ParameterSampler, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             roc_curve, precision_recall_curve, mean_absolute_error,
                             mean_squared_error, r2_score)

warnings.filterwarnings('ignore', category=UserWarning)   # mniej szumu w wyjściu (np. ostrzeżenia o nazwach cech)
pd.set_option('display.precision', 4)                       # czytelniejsze tabele
plt.rcParams['figure.dpi'] = 100
print('torch', torch.__version__)
# Lokalny zapis wykresów. W notebooku figury też się wyświetlą, w .py zapiszą się na dysk.
from pathlib import Path
import re, unicodedata
try:
    from IPython.display import display as _ipython_display
except ImportError:
    _ipython_display = None

def display(o):
    if '__file__' not in globals() and _ipython_display is not None:
        return _ipython_display(o)
    print(o.to_string() if hasattr(o, 'to_string') else str(o))

LAB_ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DL_MODE = 'fast' if os.environ.get('DL_FAST', '0') == '1' else 'full'
OUTPUT_DIR = LAB_ROOT / f'wyniki_{DL_MODE}'
MODELE_DIR = OUTPUT_DIR / 'modele'
(OUTPUT_DIR / 'wykresy').mkdir(parents=True, exist_ok=True)
MODELE_DIR.mkdir(parents=True, exist_ok=True)

_plt_show_real = plt.show
_wykres_counter = 0

def _plt_show_save(*args, **kwargs):
    global _wykres_counter
    for nr in plt.get_fignums():
        fig = plt.figure(nr)
        _wykres_counter += 1
        nazwa = (fig._suptitle.get_text() if fig._suptitle else '')
        if not nazwa and fig.axes:
            nazwa = fig.axes[0].get_title()
        nazwa = unicodedata.normalize('NFKD', nazwa).encode('ascii', errors='ignore').decode()
        nazwa = re.sub(r'[^a-zA-Z0-9]+', '_', nazwa).strip('_').lower()[:48] or 'wykres'
        sciezka = OUTPUT_DIR / 'wykresy' / f'{_wykres_counter:02d}_{nazwa}.png'
        fig.savefig(sciezka, dpi=135, bbox_inches='tight')
        print('[WYKRES]', sciezka.relative_to(LAB_ROOT))
    if '__file__' in globals():
        plt.close('all')
    else:
        _plt_show_real(*args, **kwargs)

plt.show = _plt_show_save

# ====================
# Sekcja źródłowa notebooka, komórka 05
# ====================
# Urządzenie: GPU, jeśli jest dostępne
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Urządzenie:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Brak GPU - rozważ SZYBKI_TRYB = True (patrz niżej).')

# Tryb szybki: True przy pracy bez GPU (mniej epok, foldów i danych)
SZYBKI_TRYB = os.environ.get("DL_FAST", "0") == "1"
# SZYBKI_TRYB = True   # <- odkomentuj, jeśli pracujesz na CPU
if SZYBKI_TRYB:
    torch.set_num_threads(2)    # oszczędnie na słabszych maszynach

# Wszystkie "koszty obliczeniowe" w jednym miejscu
EPOKI_MAX   = 30  if SZYBKI_TRYB else 300   # górny limit epok (early stopping zwykle kończy wcześniej)
CIERPLIWOSC = 5   if SZYBKI_TRYB else 20    # patience dla early stopping
EPOKI_DEMO  = 15  if SZYBKI_TRYB else 150   # trening bez early stopping w demonstracji 2.3
EPOKI_REG   = 15  if SZYBKI_TRYB else 200   # limit epok dla regresji
N_FOLDS     = 3   if SZYBKI_TRYB else 5     # liczba foldów w CV
N_POWT      = 1   if SZYBKI_TRYB else 3     # liczba powtórzeń CV
N_ITER      = 3   if SZYBKI_TRYB else 15    # iteracje random search
N_BOOT      = 200 if SZYBKI_TRYB else 2000  # próby bootstrap
N_PERM      = 3   if SZYBKI_TRYB else 20    # powtórzenia permutacji w permutation importance
N_TREES     = 50  if SZYBKI_TRYB else 300   # liczba drzew w lasach losowych
N_CAL       = 2000 if SZYBKI_TRYB else None # rozmiar podzbioru California Housing (None = cały)
print('SZYBKI_TRYB =', SZYBKI_TRYB)
if DEVICE.type == 'cpu' and not SZYBKI_TRYB:
    torch.set_num_threads(int(os.environ.get('DL_TORCH_THREADS', '4')))  # Xeon: 4 fizyczne rdzenie

# ====================
# Sekcja źródłowa notebooka, komórka 07
# ====================
SEED = 42

def ustaw_seed(seed=SEED):
    # Ustawia ziarna we wszystkich używanych generatorach liczb losowych
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # deterministyczne algorytmy cuDNN
    torch.backends.cudnn.benchmark = False      # bez autotuningu (który bywa niedeterministyczny)

ustaw_seed()
rng = np.random.default_rng(SEED)   # generator numpy do demonstracji w sekcji 5

# ====================
# Sekcja źródłowa notebooka, komórka 09
# ====================
bc = load_breast_cancer(as_frame=True)          # dane jako DataFrame
X_bc = bc.data                                   # 30 cech
y_bc = (1 - bc.target).astype(int)               # PRZEKODOWANIE: 1 = złośliwy (malignant)
y_bc.name = 'zlosliwy'

print('Kształt X:', X_bc.shape)
print('Liczność klas (1 = złośliwy):')
print(y_bc.value_counts().rename({0: 'łagodny (0)', 1: 'złośliwy (1)'}))
print(f'Odsetek klasy pozytywnej (prewalencja): {y_bc.mean():.3f}')
display(X_bc.iloc[:5, :8])

# ====================
# Sekcja źródłowa notebooka, komórka 10
# ====================
# Skale cech różnią się o rzędy wielkości - dlatego standaryzacja jest dla sieci konieczna
opis = X_bc.describe().T[['mean', 'std', 'min', 'max']]
display(opis.sort_values('mean', ascending=False).head(6))
display(opis.sort_values('mean').head(4))

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(range(X_bc.shape[1]), X_bc.std().values)          # odchylenie standardowe każdej cechy
ax.set_yscale('log')                                      # skala log - inaczej małe cechy byłyby niewidoczne
ax.set_xlabel('Indeks cechy'); ax.set_ylabel('Odchylenie standardowe (skala log)')
ax.set_title('Breast Cancer: rozrzut skal cech przed standaryzacją')
plt.tight_layout(); plt.show()

# ====================
# Sekcja źródłowa notebooka, komórka 12
# ====================
# Krok 1: odcinamy test (20%) - od tej chwili leży "w sejfie"
X_dev, X_test, y_dev, y_test = train_test_split(
    X_bc, y_bc, test_size=0.20, stratify=y_bc, random_state=SEED)
# Krok 2: z części deweloperskiej wydzielamy walidację (0.25 * 0.8 = 20% całości)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED)

for nazwa, yy in [('train', y_train), ('val', y_val), ('test', y_test)]:
    print(f'{nazwa:5s}: n = {len(yy):3d}, odsetek złośliwych = {yy.mean():.3f}')

# Krok 3: skaler dopasowany WYŁĄCZNIE na train
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)   # fit był na train -> transform
X_val_s   = scaler.transform(X_val)     # tylko transform
X_test_s  = scaler.transform(X_test)    # tylko transform

print('\nŚrednie cech po standaryzacji (train):', X_train_s.mean(0)[:4].round(3), '...')
print('Średnie cech po standaryzacji (val):  ', X_val_s.mean(0)[:4].round(3), '... (nie są dokładnie 0 - i tak ma być!)')

# ====================
# Sekcja źródłowa notebooka, komórka 15
# ====================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        # float32 na wejściu, y zawsze jako kolumna - BCE i MSE tego oczekują
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        assert len(self.X) == len(self.y), 'X i y mają różną liczbę rekordów'

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# shuffle tylko train. Walidacja i test muszą zachować kolejność rekordów.
train_loader = DataLoader(TabularDataset(X_train_s, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TabularDataset(X_val_s, y_val), batch_size=256, shuffle=False)
test_loader = DataLoader(TabularDataset(X_test_s, y_test), batch_size=256, shuffle=False)

# ====================
# Sekcja źródłowa notebooka, komórka 16
# ====================
# Komórka testowa - nie modyfikuj
xb, yb = next(iter(train_loader))
print('X batch:', xb.shape, xb.dtype)
print('y batch:', yb.shape, yb.dtype)
print('Liczba batchy w train_loader:', len(train_loader))
assert xb.shape[1] == 30 and yb.shape[1] == 1 and xb.dtype == torch.float32

# ====================
# Sekcja źródłowa notebooka, komórka 18
# ====================
class MLP(nn.Module):
    # MLP dla danych tabelarycznych: [Linear -> ReLU -> Dropout] x len(hidden) -> Linear(n_out)
    def __init__(self, n_in, hidden=(64, 32), dropout=0.2, n_out=1):
        super().__init__()
        warstwy, d = [], n_in
        for h in hidden:                                  # kolejne warstwy ukryte
            warstwy += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        warstwy.append(nn.Linear(d, n_out))               # wyjście: logit (klasyfikacja) lub wartość (regresja)
        self.net = nn.Sequential(*warstwy)

    def forward(self, x):
        return self.net(x)                                # BEZ sigmoidy - zwracamy logity

ustaw_seed()
model_demo = MLP(n_in=30).to(DEVICE)
print(model_demo)
n_param = sum(p.numel() for p in model_demo.parameters())
print('Liczba parametrów:', n_param, '= (30*64+64) + (64*32+32) + (32*1+1)')

# ====================
# Sekcja źródłowa notebooka, komórka 20
# ====================
def trenuj_epoke(model, loader, loss_fn, optimizer):
    # Jedna epoka uczenia; zwraca średnią stratę treningową
    model.train()                                          # Dropout aktywny
    suma, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)              # batch na GPU/CPU
        optimizer.zero_grad()                              # zerujemy gradienty z poprzedniego kroku
        loss = loss_fn(model(xb), yb)                      # forward + strata
        loss.backward()                                    # backprop (autograd)
        optimizer.step()                                   # aktualizacja wag
        suma += loss.item() * len(xb); n += len(xb)        # ważenie rozmiarem batcha
    return suma / n

@torch.no_grad()                                           # bez grafu obliczeń - szybciej, mniej pamięci
def ewaluuj(model, loader, loss_fn):
    # Średnia strata na zbiorze (walidacyjnym lub testowym)
    model.eval()                                           # Dropout wyłączony
    suma, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        suma += loss_fn(model(xb), yb).item() * len(xb); n += len(xb)
    return suma / n

@torch.no_grad()
def przewiduj_torch(model, X, batch_size=1024):
    # Surowe wyjścia sieci (logity lub wartości regresji) jako wektor numpy
    model.eval()
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
    wyj = [model(X_t[i:i + batch_size].to(DEVICE)).cpu() for i in range(0, len(X_t), batch_size)]
    return torch.cat(wyj).numpy().ravel()

def proba_torch(model, X):
    # Prawdopodobieństwo klasy 1 = sigmoida z logitu
    return torch.sigmoid(torch.as_tensor(przewiduj_torch(model, X))).numpy()

# ====================
# Sekcja źródłowa notebooka, komórka 22
# ====================
ustaw_seed()
model_demo = MLP(n_in=30, hidden=(64, 32), dropout=0.2).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model_demo.parameters(), lr=1e-3, weight_decay=1e-4)

hist_tr, hist_val = [], []
t0 = time.time()
for ep in range(EPOKI_DEMO):
    hist_tr.append(trenuj_epoke(model_demo, train_loader, loss_fn, opt))
    hist_val.append(ewaluuj(model_demo, val_loader, loss_fn))
print(f'Czas: {time.time() - t0:.1f} s; najniższa strata val = {min(hist_val):.4f} w epoce {int(np.argmin(hist_val)) + 1}')

plt.figure(figsize=(7, 4))
plt.plot(range(1, EPOKI_DEMO + 1), hist_tr, label='train')
plt.plot(range(1, EPOKI_DEMO + 1), hist_val, label='val')
plt.axvline(int(np.argmin(hist_val)) + 1, ls='--', c='gray', label='minimum straty val')
plt.xlabel('Epoka'); plt.ylabel('BCE'); plt.yscale('log')
plt.title('Krzywe uczenia MLP bez early stopping'); plt.legend(); plt.tight_layout(); plt.show()

# ====================
# Sekcja źródłowa notebooka, komórka 25
# ====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_state = None
        self.best_epoch = -1
        self.licznik = 0

    def krok(self, val_loss, model, epoka):
        # deepcopy obowiązkowo; samo state_dict trzyma referencje do wag
        if np.isfinite(val_loss) and val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoka)
            self.best_state = copy.deepcopy(model.state_dict())
            self.licznik = 0
        else:
            self.licznik += 1
        return self.licznik >= self.patience

# ====================
# Sekcja źródłowa notebooka, komórka 26
# ====================
# Komórka testowa - nie modyfikuj
_m = nn.Linear(2, 1)
_es = EarlyStopping(patience=2)
_straty = [1.0, 0.8, 0.9, 0.7, 0.75, 0.72]
_stop = []
for _ep, _l in enumerate(_straty):
    with torch.no_grad():
        _m.weight.fill_(float(_ep))          # wagi = numer epoki, żeby sprawdzić kopię
    _stop.append(_es.krok(_l, _m, _ep))
print('Decyzje stop:', _stop)
assert _stop == [False, False, False, False, False, True], 'zła logika licznika'
assert _es.best_epoch == 3 and abs(_es.best_loss - 0.7) < 1e-9
assert float(_es.best_state['weight'][0, 0]) == 3.0, 'best_state nie jest kopią (deepcopy!)'
print('EarlyStopping działa poprawnie.')

# ====================
# Sekcja źródłowa notebooka, komórka 27
# ====================
def trenuj(model, train_loader, val_loader, loss_fn, lr=1e-3, weight_decay=1e-4,
           max_epoki=EPOKI_MAX, cierpliwosc=CIERPLIWOSC, verbose=True):
    # Pełny trening z early stopping; zwraca model z NAJLEPSZYMI wagami i historię
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    es = EarlyStopping(patience=cierpliwosc, min_delta=1e-4)
    hist = {'train_loss': [], 'val_loss': []}
    for ep in range(max_epoki):
        hist['train_loss'].append(trenuj_epoke(model, train_loader, loss_fn, optimizer))
        hist['val_loss'].append(ewaluuj(model, val_loader, loss_fn))
        if es.krok(hist['val_loss'][-1], model, ep):     # decyzja o zatrzymaniu
            break
    if es.best_state is not None:
        model.load_state_dict(es.best_state)            # przywracamy najlepsze wagi
    hist['best_epoch'] = es.best_epoch
    if verbose:
        print(f'Zatrzymano po {ep + 1} epokach; najlepsza epoka {es.best_epoch + 1}, '
              f'val_loss = {es.best_loss:.4f}')
    return model, hist

# ====================
# Sekcja źródłowa notebooka, komórka 29
# ====================
ustaw_seed()
mlp_bc = MLP(n_in=30, hidden=(64, 32), dropout=0.2).to(DEVICE)
t0 = time.time()
mlp_bc, hist_bc = trenuj(mlp_bc, train_loader, val_loader, nn.BCEWithLogitsLoss(), lr=1e-3, weight_decay=1e-4)
print(f'Czas treningu: {time.time() - t0:.1f} s')

plt.figure(figsize=(7, 4))
plt.plot(hist_bc['train_loss'], label='train')
plt.plot(hist_bc['val_loss'], label='val')
plt.axvline(hist_bc['best_epoch'], ls='--', c='gray', label='przywrócona epoka')
plt.xlabel('Epoka (od 0)'); plt.ylabel('BCE'); plt.yscale('log')
plt.title('MLP z early stopping'); plt.legend(); plt.tight_layout(); plt.show()

# Zapis i ponowne wczytanie najlepszych wag
torch.save(mlp_bc.state_dict(), str(MODELE_DIR / 'mlp_breast_cancer_best.pt'))
mlp_wczytany = MLP(n_in=30, hidden=(64, 32), dropout=0.2).to(DEVICE)       # ta sama architektura
mlp_wczytany.load_state_dict(torch.load(str(MODELE_DIR / 'mlp_breast_cancer_best.pt'), map_location=DEVICE))
roznica = np.abs(przewiduj_torch(mlp_bc, X_val_s) - przewiduj_torch(mlp_wczytany, X_val_s)).max()
print('Maksymalna różnica predykcji oryginał vs wczytany:', roznica)

# ====================
# Sekcja źródłowa notebooka, komórka 31
# ====================
# Baseline: każdy model dopasowany WYŁĄCZNIE na train; porównanie na walidacji
modele_klas = {
    'LogReg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),   # skaler w środku pipeline
    'RandomForest': RandomForestClassifier(n_estimators=N_TREES, random_state=SEED),
    'HistGB': HistGradientBoostingClassifier(random_state=SEED),
}
p_val, p_test = {}, {}
for nazwa, m in modele_klas.items():
    m.fit(X_train, y_train)                         # surowe X - pipeline LogReg sam skaluje
    p_val[nazwa] = m.predict_proba(X_val)[:, 1]     # prawdopodobieństwo klasy 1 (złośliwy)
    p_test[nazwa] = m.predict_proba(X_test)[:, 1]
p_val['MLP'] = proba_torch(mlp_bc, X_val_s)         # sieć używa danych ustandaryzowanych skalerem z train
p_test['MLP'] = proba_torch(mlp_bc, X_test_s)

def metryki_klas(y, p, prog=0.5):
    # Zestaw metryk dla klasyfikacji binarnej (p = prawdopodobieństwo klasy 1)
    yhat = (p >= prog).astype(int)
    return {'ROC-AUC': roc_auc_score(y, p), 'PR-AUC (AP)': average_precision_score(y, p),
            'Brier': brier_score_loss(y, p), 'Accuracy': accuracy_score(y, yhat),
            'Recall': recall_score(y, yhat), 'Precision': precision_score(y, yhat, zero_division=0),
            'F1': f1_score(y, yhat)}

tab_val = pd.DataFrame({n: metryki_klas(y_val, p) for n, p in p_val.items()}).T
print('Metryki na WALIDACJI (próg 0.5):')
display(tab_val.round(4))

# ====================
# Sekcja źródłowa notebooka, komórka 33
# ====================
# Macierz pomyłek i raport klasyfikacji dla MLP na walidacji (próg 0.5)
yhat_val_mlp = (p_val['MLP'] >= 0.5).astype(int)
print(classification_report(y_val, yhat_val_mlp, target_names=['łagodny (0)', 'złośliwy (1)'], digits=3))

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_val, yhat_val_mlp),
                       display_labels=['łagodny', 'złośliwy']).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('MLP - macierz pomyłek (walidacja, próg 0.5)')
ax.set_xlabel('Predykcja'); ax.set_ylabel('Prawda')
plt.tight_layout(); plt.show()

# ====================
# Sekcja źródłowa notebooka, komórka 34
# ====================
# Krzywe ROC i PR dla wszystkich modeli (walidacja)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for nazwa, p in p_val.items():
    fpr, tpr, _ = roc_curve(y_val, p)
    axes[0].plot(fpr, tpr, label=f'{nazwa} (AUC={roc_auc_score(y_val, p):.3f})')
    prec, rec, _ = precision_recall_curve(y_val, p)
    axes[1].plot(rec, prec, label=f'{nazwa} (AP={average_precision_score(y_val, p):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='losowy')
axes[0].set_xlabel('FPR (1 - specyficzność)'); axes[0].set_ylabel('TPR (czułość)'); axes[0].set_title('Krzywe ROC (walidacja)')
axes[1].axhline(y_val.mean(), color='k', ls='--', lw=1, label=f'losowy (AP = prewalencja = {y_val.mean():.2f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('Krzywe precision-recall (walidacja)')
for a in axes: a.legend(loc='lower right' if a is axes[0] else 'lower left', fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.show()

# ====================
# Sekcja źródłowa notebooka, komórka 37
# ====================
# Wybór progu tylko na walidacji. Test zostaje na końcową ocenę.
prec_val, rec_val, prog_vals = precision_recall_curve(y_val, p_val['MLP'])
mask = rec_val[:-1] >= 0.98   # ostatnia wartość PR nie ma odpowiadającego progu
if mask.any():
    kandydaci = np.flatnonzero(mask)
    # Maks. precision; przy remisie wyższy próg -> mniej fałszywych alarmów
    najlepsza_p = prec_val[kandydaci].max()
    kandydaci = kandydaci[np.isclose(prec_val[kandydaci], najlepsza_p)]
    prog_opt = float(prog_vals[kandydaci[-1]])
else:
    prog_opt = 0.5

fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.plot(prog_vals, prec_val[:-1], label='precision (val)')
ax.plot(prog_vals, rec_val[:-1], label='recall (val)')
ax.axvline(prog_opt, color='tab:red', ls='--', label=f'próg wybrany = {prog_opt:.3f}')
ax.axvline(0.5, color='gray', ls=':', label='próg domyślny = 0.5')
ax.axhline(0.98, color='tab:green', ls=':', label='cel recall = 0.98')
ax.set(xlabel='próg decyzji', ylabel='metryka', title='3.1 Wybór progu na walidacji')
ax.set_ylim(0, 1.05); ax.legend(loc='lower left', fontsize=8)
plt.tight_layout(); plt.show()

wyniki_progow = []
for opis, prog in [('domyślny', 0.5), ('dobrany na val', prog_opt)]:
    yh = (p_test['MLP'] >= prog).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yh, labels=[0, 1]).ravel()
    wyniki_progow.append({'wariant': opis, 'prog': prog, 'recall': recall_score(y_test, yh),
                          'precision': precision_score(y_test, yh, zero_division=0),
                          'specificity': tn / (tn + fp), 'TN': int(tn), 'FP': int(fp),
                          'FN': int(fn), 'TP': int(tp)})
tab_progi = pd.DataFrame(wyniki_progow)
print('Progi - ocena na teście (próg dobrany WYŁĄCZNIE na val):')
display(tab_progi.round(4))
if tab_progi.loc[1, 'recall'] < 0.98:
    print('Recall < 0.98 na teście; walidacja 114 osób daje mało precyzyjny dobór progu.')
print('W praktyce warto mieć margines bezpieczeństwa: cel na walidacji > 0.98, większy zbiór.')

# ====================
# Sekcja źródłowa notebooka, komórka 40
# ====================
# Platt kalibrujemy na logitach z walidacji; nie uczymy niczego na teście.
logity_val = przewiduj_torch(mlp_bc, X_val_s).reshape(-1, 1)
logity_test = przewiduj_torch(mlp_bc, X_test_s).reshape(-1, 1)
platt = LogisticRegression(max_iter=2000, random_state=SEED)
platt.fit(logity_val, np.asarray(y_val))
p_platt_test = platt.predict_proba(logity_test)[:, 1]

pred_kal = {n: p_test[n] for n in ['MLP', 'LogReg', 'RandomForest']}
pred_kal['MLP + Platt'] = p_platt_test
fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='idealna kalibracja')
for nazwa, prob in pred_kal.items():
    odsetek, p_sr = calibration_curve(y_test, prob, n_bins=5, strategy='quantile')
    brier = brier_score_loss(y_test, prob)
    ax.plot(p_sr, odsetek, marker='o', label=f'{nazwa} (Brier {brier:.4f})')
ax.set(xlabel='średnia przewidywana p', ylabel='rzeczywisty odsetek klasy 1',
       title='3.2 Kalibracja na teście (Platt dopasowany na val)')
ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout(); plt.show()

tab_kal = pd.DataFrame([{'model': nazwa,
                         'Brier_test': brier_score_loss(y_test, prob),
                         'ROC_AUC_test': roc_auc_score(y_test, prob)}
                        for nazwa, prob in pred_kal.items()])
print('Brier i ROC-AUC przed/po kalibracji:')
display(tab_kal.round(5))
print('Platt to monotoniczna transformacja logitów: porządek wyników i AUC pozostają bez zmian.')

# ====================
# Sekcja źródłowa notebooka, komórka 42
# ====================
cal = fetch_california_housing(as_frame=True)
X_cal, y_cal = cal.data, cal.target
if N_CAL is not None and len(X_cal) > N_CAL:              # tryb szybki: losowy podzbiór
    idx = np.random.default_rng(SEED).choice(len(X_cal), N_CAL, replace=False)
    X_cal, y_cal = X_cal.iloc[idx], y_cal.iloc[idx]
print('Kształt:', X_cal.shape, '| cel min/max:', round(float(y_cal.min()), 3), round(float(y_cal.max()), 3))
print('Odsetek obserwacji z celem >= 5.0 (ucięcie):', round(float((y_cal >= 5.0).mean()), 4))

# Podział 60/20/20 - analogicznie jak wyżej
Xc_dev, Xc_test, yc_dev, yc_test = train_test_split(X_cal, y_cal, test_size=0.2, random_state=SEED)
Xc_train, Xc_val, yc_train, yc_val = train_test_split(Xc_dev, yc_dev, test_size=0.25, random_state=SEED)
sc_cal = StandardScaler().fit(Xc_train)                    # skaler tylko na train
Xc_train_s, Xc_val_s, Xc_test_s = (sc_cal.transform(a) for a in (Xc_train, Xc_val, Xc_test))

plt.figure(figsize=(6, 3.5))
plt.hist(yc_train, bins=50, color='steelblue')
plt.xlabel('Mediana wartości domu [100 tys. USD]'); plt.ylabel('Liczba dzielnic')
plt.title('Rozkład celu (train) - zwróć uwagę na pik przy 5.0'); plt.tight_layout(); plt.show()

# ====================
# Sekcja źródłowa notebooka, komórka 43
# ====================
# Baseline'y regresji (dopasowanie na train, ocena na teście w zadaniu 4.1)
modele_reg = {'Dummy (średnia)': DummyRegressor(),
              'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
              'HistGB': HistGradientBoostingRegressor(random_state=SEED)}
pred_reg_test = {}
for nazwa, m in modele_reg.items():
    m.fit(Xc_train, yc_train)
    pred_reg_test[nazwa] = m.predict(Xc_test)

def metryki_reg(y, yhat):
    return {'MAE': mean_absolute_error(y, yhat), 'RMSE': math.sqrt(mean_squared_error(y, yhat)),
            'R2': r2_score(y, yhat)}

display(pd.DataFrame({n: metryki_reg(yc_test, p) for n, p in pred_reg_test.items()}).T.round(4))

# ====================
# Sekcja źródłowa notebooka, komórka 45
# ====================
# Regresja - bez standaryzacji y (wartości pozostają w 100 tys. USD).
train_cal_loader = DataLoader(TabularDataset(Xc_train_s, yc_train), batch_size=64, shuffle=True)
val_cal_loader = DataLoader(TabularDataset(Xc_val_s, yc_val), batch_size=256, shuffle=False)
test_cal_loader = DataLoader(TabularDataset(Xc_test_s, yc_test), batch_size=256, shuffle=False)
ustaw_seed(SEED)
mlp_cal = MLP(n_in=8, hidden=(128, 64), dropout=0.1).to(DEVICE)
t0 = time.time()
mlp_cal, hist_cal = trenuj(mlp_cal, train_cal_loader, val_cal_loader,
                          nn.MSELoss(), lr=1e-3, weight_decay=1e-4,
                          max_epoki=EPOKI_REG, cierpliwosc=CIERPLIWOSC, verbose=True)
print(f'Regresja MLP: {time.time()-t0:.1f} s, najlepsza epoka {hist_cal["best_epoch"]+1}')
pred_reg_test['MLP'] = przewiduj_torch(mlp_cal, Xc_test_s)
tab_reg = pd.DataFrame({n: metryki_reg(yc_test, pred) for n, pred in pred_reg_test.items()}).T
print('Regresja - wszystkie modele, wyłącznie test:')
display(tab_reg.round(4))

p = pred_reg_test['MLP']; y = np.asarray(yc_test)
reszty = y - p
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
axes[0].scatter(y, p, s=8, alpha=0.22)
lims = [min(y.min(), p.min()), max(y.max(), p.max())]
axes[0].plot(lims, lims, 'k--', lw=1)
axes[0].set(xlabel='y prawdziwe', ylabel='y pred', title='Wartość vs predykcja')
axes[1].scatter(p, reszty, s=8, alpha=0.2)
axes[1].axhline(0, color='k', ls='--', lw=1)
axes[1].set(xlabel='predykcja', ylabel='reszta: y - pred', title='Reszty vs predykcja')
axes[2].hist(reszty, bins=45, color='steelblue')
axes[2].set(xlabel='reszta', ylabel='liczba próbek', title='Rozkład reszt')
fig.suptitle('4.1 California Housing - analiza reszt MLP', fontsize=12)
plt.tight_layout(); plt.show()
print(f'Odsetek uciętych y >= 5.0 w teście: {np.mean(y >= 5.0):.2%}; to wpływa na reszty.')

# ====================
# Sekcja źródłowa notebooka, komórka 49
# ====================
cv5 = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_POWT, random_state=SEED)   # te same foldy dla obu procedur

# Z WYCIEKIEM: skaler "widzi" wszystkie obserwacje, także przyszłe foldy testowe
X_dev_leak = StandardScaler().fit_transform(X_dev)
auc_leak = cross_val_score(LogisticRegression(max_iter=5000), X_dev_leak, y_dev, cv=cv5, scoring='roc_auc')

# POPRAWNIE: skaler jest krokiem pipeline'u -> dopasowywany w każdym foldzie tylko na jego train
pipe_ok = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
auc_ok = cross_val_score(pipe_ok, X_dev, y_dev, cv=cv5, scoring='roc_auc')

print(f'Skaler na całości: ROC-AUC = {auc_leak.mean():.4f} +/- {auc_leak.std():.4f}')
print(f'Skaler w pipeline: ROC-AUC = {auc_ok.mean():.4f} +/- {auc_ok.std():.4f}')
print(f'Średnia różnica (parami, te same foldy): {np.mean(auc_leak - auc_ok):+.5f}')

# ====================
# Sekcja źródłowa notebooka, komórka 51
# ====================
n_szum, p_szum, k_szum = 100, (2000 if SZYBKI_TRYB else 5000), 20
X_szum = rng.normal(size=(n_szum, p_szum))              # cechy: czysty szum
y_szum = rng.permutation(np.repeat([0, 1], n_szum // 2))  # etykiety niezależne od cech (50/50)
cv_szum = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Z WYCIEKIEM: selekcja cech na CAŁYM zbiorze (używa etykiet foldów testowych!)
selektor = SelectKBest(f_classif, k=k_szum).fit(X_szum, y_szum)
X_szum_sel = selektor.transform(X_szum)
acc_szum_leak = cross_val_score(LogisticRegression(max_iter=2000), X_szum_sel, y_szum, cv=cv_szum)
print(f'Selekcja przed CV:  accuracy = {acc_szum_leak.mean():.3f}  (prawdziwa wartość: 0.5!)')

# ====================
# Sekcja źródłowa notebooka, komórka 53
# ====================
# Poprawnie: selektor wewnątrz Pipeline, więc fit odbywa się osobno w foldzie.
pipe_selekcja = Pipeline([('select', SelectKBest(f_classif, k=20)),
                          ('clf', LogisticRegression(max_iter=2000))])
acc_szum_ok = cross_val_score(pipe_selekcja, X_szum, y_szum, cv=cv_szum, scoring='accuracy').mean()
print(f'Selekcja w CV: accuracy = {acc_szum_ok:.3f} (vs wyciek {acc_szum_leak.mean():.3f})')

wiersze_szum = []
for k in [5, 20, 100, 500]:
    # Celowo błędne: wybór cech widzi y z całego zbioru, w tym fold testowy.
    X_bad = SelectKBest(f_classif, k=k).fit_transform(X_szum, y_szum)
    acc_bad = cross_val_score(LogisticRegression(max_iter=2000), X_bad, y_szum,
                              cv=cv_szum, scoring='accuracy').mean()
    # Prawidłowe: SelectKBest dopasowuje się tylko na foldzie treningowym.
    pipe_good = Pipeline([('select', SelectKBest(f_classif, k=k)),
                          ('clf', LogisticRegression(max_iter=2000))])
    acc_good = cross_val_score(pipe_good, X_szum, y_szum,
                               cv=cv_szum, scoring='accuracy').mean()
    wiersze_szum.append({'k': k, 'z_wyciekiem': acc_bad, 'poprawnie': acc_good})
tab_szum = pd.DataFrame(wiersze_szum)
display(tab_szum.round(3))
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.plot(tab_szum['k'], tab_szum['z_wyciekiem'], '-o', label='wyciek: selekcja przed CV')
ax.plot(tab_szum['k'], tab_szum['poprawnie'], '-s', label='pipeline: selekcja w CV')
ax.axhline(0.5, ls='--', c='gray', label='losowanie (0.5)')
ax.set(xscale='log', xlabel='k - liczba wybranych cech', ylabel='accuracy (CV)',
       title='5.1 Wyciek danych przy selekcji cech')
ax.legend(); plt.tight_layout(); plt.show()
najsilniejszy = tab_szum.loc[(tab_szum['z_wyciekiem'] - tab_szum['poprawnie']).idxmax()]
print(f'Największy obserwowany rozjazd: k={int(najsilniejszy.k)}. Selekcja na całości "zna" testowe etykiety.')

# ====================
# Sekcja źródłowa notebooka, komórka 55
# ====================
# Dane niezbalansowane (ok. 10% klasy 1) o umiarkowanie trudnym sygnale
X_imb, y_imb = make_classification(n_samples=600, n_features=10, n_informative=3, n_redundant=2,
                                   weights=[0.9, 0.1], class_sep=0.6, flip_y=0.05, random_state=SEED)

def oversample(X, y, gen):
    # Losowe powielanie klasy 1 do liczności klasy 0 (z powtórzeniami)
    i_min, i_maj = np.where(y == 1)[0], np.where(y == 0)[0]
    extra = gen.choice(i_min, size=len(i_maj) - len(i_min), replace=True)
    idx = np.concatenate([i_maj, i_min, extra])
    return X[idx], y[idx]

cv_imb = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED)

# Z WYCIEKIEM: oversampling całego zbioru, potem CV
X_os, y_os = oversample(X_imb, y_imb, np.random.default_rng(SEED))
f1_os_leak = cross_val_score(rf, X_os, y_os, cv=cv_imb, scoring='f1')

# POPRAWNIE: oversampling tylko w części treningowej foldu; fold testowy w oryginalnym rozkładzie
f1_os_ok = []
gen = np.random.default_rng(SEED)
for tr, te in cv_imb.split(X_imb, y_imb):
    Xtr, ytr = oversample(X_imb[tr], y_imb[tr], gen)
    rf.fit(Xtr, ytr)
    f1_os_ok.append(f1_score(y_imb[te], rf.predict(X_imb[te])))

print(f'Prewalencja klasy 1: {y_imb.mean():.3f}')
print(f'Oversampling przed CV: F1 = {f1_os_leak.mean():.3f}')
print(f'Oversampling w foldzie: F1 = {np.mean(f1_os_ok):.3f}')

# ====================
# Sekcja źródłowa notebooka, komórka 57
# ====================
# Umiarkowany sygnał: AUC ok. 0.85
X_w, y_w = make_classification(n_samples=1000, n_features=8, n_informative=3, n_redundant=0,
                               class_sep=0.3, flip_y=0.15, random_state=SEED)
cecha_po_diagnozie = y_w + rng.normal(0, 0.4, size=len(y_w))     # skutek etykiety + szum
X_wl = np.column_stack([X_w, cecha_po_diagnozie])
nazwy_wl = [f'x{i}' for i in range(8)] + ['skierowanie_po_diagnozie']

Xa, Xb, ya, yb = train_test_split(X_wl, y_w, test_size=0.3, stratify=y_w, random_state=SEED)
hgb_bez = HistGradientBoostingClassifier(random_state=SEED).fit(Xa[:, :8], ya)
hgb_z = HistGradientBoostingClassifier(random_state=SEED).fit(Xa, ya)
print(f'ROC-AUC bez cechy-pośrednika: {roc_auc_score(yb, hgb_bez.predict_proba(Xb[:, :8])[:, 1]):.3f}')
print(f'ROC-AUC Z cechą-pośrednikiem: {roc_auc_score(yb, hgb_z.predict_proba(Xb)[:, 1]):.3f}  <- "zbyt dobre, by było prawdziwe"')

# Diagnostyka: permutation importance na zbiorze wstrzymanym
pi_wl = permutation_importance(hgb_z, Xb, yb, scoring='roc_auc', n_repeats=N_PERM, random_state=SEED)
kol = ['crimson' if n == 'skierowanie_po_diagnozie' else 'steelblue' for n in nazwy_wl]
plt.figure(figsize=(7, 3.5))
plt.barh(nazwy_wl, pi_wl.importances_mean, xerr=pi_wl.importances_std, color=kol)
plt.xlabel('Spadek ROC-AUC po permutacji cechy'); plt.title('Jedna cecha dominuje - sygnał ostrzegawczy wycieku')
plt.tight_layout(); plt.show()

# ====================
# Sekcja źródłowa notebooka, komórka 59
# ====================
n_pac, m_prob, d_g = (40 if SZYBKI_TRYB else 80), 5, 20
odcisk = rng.normal(size=(n_pac, d_g))                               # "odcisk palca" każdej pacjentki
X_g = np.repeat(odcisk, m_prob, axis=0) + rng.normal(0, 0.3, size=(n_pac * m_prob, d_g))  # 5 podobnych próbek na osobę
y_pac = rng.permutation(np.repeat([0, 1], n_pac // 2))               # etykieta losowa, niezależna od cech
y_g = np.repeat(y_pac, m_prob)
grupy = np.repeat(np.arange(n_pac), m_prob)                          # identyfikator pacjentki

modele_g = {'RandomForest': RandomForestClassifier(n_estimators=N_TREES, random_state=SEED),
            'kNN (k=1)': KNeighborsClassifier(n_neighbors=1)}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)              # losowy podział WIERSZY
acc_kf = {n: cross_val_score(m, X_g, y_g, cv=kf).mean() for n, m in modele_g.items()}
for n, a in acc_kf.items():
    print(f'{n:13s} KFold po wierszach: accuracy = {a:.3f}   (prawdziwa wartość: 0.5)')

# ====================
# Sekcja źródłowa notebooka, komórka 61
# ====================
# Tu jednostką podziału jest pacjent, nie pojedynczy wiersz.
gkf = GroupKFold(n_splits=5)
acc_gkf = {n: cross_val_score(m, X_g, y_g, groups=grupy, cv=gkf,
                              scoring='accuracy').mean() for n, m in modele_g.items()}
for tr, te in gkf.split(X_g, y_g, groups=grupy):
    assert not set(grupy[tr]).intersection(grupy[te]), 'Wykryto pacjenta w train i test!'
kolizje_kf = sum(bool(set(grupy[tr]).intersection(grupy[te]))
                 for tr, te in kf.split(X_g, y_g))
print(f'GroupKFold: brak przecinających się pacjentów. KFold: kolizje w {kolizje_kf}/5 foldach.')
print('KFold vs GroupKFold:', {n: (round(acc_kf[n], 3), round(acc_gkf[n], 3)) for n in modele_g})

etykiety = list(modele_g.keys())
pos = np.arange(len(etykiety)); szer = 0.35
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.bar(pos - szer / 2, [acc_kf[n] for n in etykiety], width=szer, label='KFold: po wierszach')
ax.bar(pos + szer / 2, [acc_gkf[n] for n in etykiety], width=szer, label='GroupKFold: po osobach')
ax.axhline(0.5, ls='--', color='gray', label='przypadek: 0.5')
ax.set(xticks=pos, xticklabels=etykiety, ylabel='accuracy', ylim=(0, 1.05),
       title='5.2 Zależne rekordy: wyciek między pacjentami')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print('W CV po wierszach model rozpoznaje pacjentów, a nie niezależny sygnał kliniczny.')

# ====================
# Sekcja źródłowa notebooka, komórka 63
# ====================
# Przykład kompletnego pipeline'u sklearn: wszystkie kroki "uczone z danych" są wewnątrz
pipe_pelny = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),    # (tu brak braków danych, ale krok jest bezpieczny)
    ('scaler', StandardScaler()),
    ('select', SelectKBest(f_classif, k=15)),
    ('clf', LogisticRegression(max_iter=5000)),
])
wyn = cross_validate(pipe_pelny, X_dev, y_dev, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                     scoring=['roc_auc', 'balanced_accuracy'], return_estimator=True)
print('ROC-AUC per fold:', wyn['test_roc_auc'].round(4))
# Każdy fold wybrał (nieco) inne cechy - to normalne i to właśnie musi zostać uwzględnione w ocenie
for i, est in enumerate(wyn['estimator'][:3]):
    wybrane = np.array(X_dev.columns)[est.named_steps['select'].get_support()]
    print(f'fold {i}: {sorted(wybrane)[:5]} ...')

# ====================
# Sekcja źródłowa notebooka, komórka 64
# ====================
class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    # Sieć MLP z PyTorch jako estymator sklearn: można jej używać w Pipeline, cross_val_score,
    # permutation_importance. Early stopping na wewnętrznej części danych przekazanych do fit.
    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=32,
                 max_epoki=EPOKI_MAX, cierpliwosc=CIERPLIWOSC, val_frac=0.15, random_state=SEED):
        # sklearn wymaga, by __init__ tylko zapisywał parametry (bez logiki) - inaczej clone() nie zadziała
        self.hidden = hidden; self.dropout = dropout; self.lr = lr; self.weight_decay = weight_decay
        self.batch_size = batch_size; self.max_epoki = max_epoki; self.cierpliwosc = cierpliwosc
        self.val_frac = val_frac; self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)                      # wymagane przez sklearn (np. scoring='roc_auc')
        ustaw_seed(self.random_state)                     # powtarzalność (uwaga: ustawia globalne ziarna)
        # Wewnętrzny podział TYLKO z danych treningowych -> early stopping bez wycieku
        X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=self.val_frac, stratify=y,
                                                  random_state=self.random_state)
        tl = DataLoader(TabularDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        vl = DataLoader(TabularDataset(X_es, y_es), batch_size=1024)
        self.model_ = MLP(X.shape[1], self.hidden, self.dropout).to(DEVICE)
        self.model_, self.historia_ = trenuj(self.model_, tl, vl, nn.BCEWithLogitsLoss(), lr=self.lr,
                                             weight_decay=self.weight_decay, max_epoki=self.max_epoki,
                                             cierpliwosc=self.cierpliwosc, verbose=False)
        return self

    def predict_proba(self, X):
        p = proba_torch(self.model_, np.asarray(X, dtype=np.float32))
        return np.column_stack([1 - p, p])                # kolumny w kolejności self.classes_

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

# Szybki test: sieć w pipeline ze skalerem, dopasowanie na train, ocena na walidacji
pipe_mlp = make_pipeline(StandardScaler(), TorchMLPClassifier())
pipe_mlp.fit(X_train, y_train)
print('Pipeline(Scaler, TorchMLP) - ROC-AUC na walidacji:',
      round(roc_auc_score(y_val, pipe_mlp.predict_proba(X_val)[:, 1]), 4))

# ====================
# Sekcja źródłowa notebooka, komórka 66
# ====================
# Demonstracja: powtórzona stratyfikowana CV sieci w pipeline (część deweloperska, test nietknięty)
cv_rep = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_POWT, random_state=SEED)
t0 = time.time()
wyn_mlp_cv = cross_validate(make_pipeline(StandardScaler(), TorchMLPClassifier()), X_dev, y_dev, cv=cv_rep,
                            scoring=['roc_auc', 'average_precision', 'balanced_accuracy'])
print(f'Czas: {time.time() - t0:.1f} s, liczba dopasowań: {len(wyn_mlp_cv["test_roc_auc"])}')
for m in ['roc_auc', 'average_precision', 'balanced_accuracy']:
    a = wyn_mlp_cv['test_' + m]
    print(f'{m:18s}: {a.mean():.4f} +/- {a.std(ddof=1):.4f} (SD)')

# ====================
# Sekcja źródłowa notebooka, komórka 67
# ====================
# Demonstracja: bootstrap 95% CI dla ROC-AUC MLP na zbiorze testowym
def bootstrap_ci(y, p, metryka, B=N_BOOT, seed=SEED):
    g = np.random.default_rng(seed); y = np.asarray(y); wyn = []
    for _ in range(B):
        idx = g.integers(0, len(y), len(y))           # losowanie ze zwracaniem
        if len(np.unique(y[idx])) < 2:                # AUC wymaga obu klas w próbie
            continue
        wyn.append(metryka(y[idx], p[idx]))
    return np.percentile(wyn, [2.5, 97.5]), np.array(wyn)

(lo, hi), boot = bootstrap_ci(y_test, p_test['MLP'], roc_auc_score)
print(f'MLP test ROC-AUC = {roc_auc_score(y_test, p_test["MLP"]):.4f}, 95% CI bootstrap: [{lo:.4f}, {hi:.4f}]')
plt.figure(figsize=(6, 3.5))
plt.hist(boot, bins=40, color='slategray')
plt.axvline(lo, c='r', ls='--', label='2.5%'); plt.axvline(hi, c='r', ls='--', label='97.5%')
plt.xlabel('ROC-AUC w próbie bootstrap'); plt.ylabel('Liczność'); plt.title('Rozkład bootstrap ROC-AUC (test)')
plt.legend(); plt.tight_layout(); plt.show()

# ====================
# Sekcja źródłowa notebooka, komórka 69
# ====================
def przedzialy(a, n_test, n_train, poziom=0.95):
    # Średnia, CI zwykły i CI z korektą na zależność foldów.
    a = np.asarray(a, dtype=float)
    J = len(a)
    s2 = a.var(ddof=1)
    tq = stats.t.ppf(0.5 + poziom / 2, J - 1)
    naiw = tq * math.sqrt(s2 / J)
    kor = tq * math.sqrt((1 / J + n_test / n_train) * s2)
    return a.mean(), (a.mean() - naiw, a.mean() + naiw), (a.mean() - kor, a.mean() + kor)

sf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
wiersze_cv = []
for numer, (tr, te) in enumerate(sf.split(X_dev, y_dev), start=1):
    X_tr_fold, X_te_fold = X_dev.iloc[tr], X_dev.iloc[te]
    y_tr_fold, y_te_fold = y_dev.iloc[tr], y_dev.iloc[te]

    # Najpierw wewnętrzny podział, potem fit skalera na części faktycznie uczącej.
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr_fold, y_tr_fold, test_size=0.15, stratify=y_tr_fold,
        random_state=SEED + numer)
    sk = StandardScaler().fit(X_fit)
    X_fit_s = sk.transform(X_fit)
    X_es_s = sk.transform(X_es)
    X_te_s = sk.transform(X_te_fold)

    ustaw_seed(SEED + numer)
    model_fold = MLP(n_in=30).to(DEVICE)
    tl = DataLoader(TabularDataset(X_fit_s, y_fit), batch_size=32, shuffle=True)
    vl = DataLoader(TabularDataset(X_es_s, y_es), batch_size=256, shuffle=False)
    model_fold, h = trenuj(model_fold, tl, vl, nn.BCEWithLogitsLoss(),
                           max_epoki=EPOKI_MAX, cierpliwosc=CIERPLIWOSC,
                           verbose=False)
    a_mlp = roc_auc_score(y_te_fold, proba_torch(model_fold, X_te_s))
    lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    lr_model.fit(X_tr_fold, y_tr_fold)
    a_lr = roc_auc_score(y_te_fold, lr_model.predict_proba(X_te_fold)[:, 1])
    wiersze_cv.append({'fold': numer, 'ROC_AUC_MLP': a_mlp,
                       'ROC_AUC_LogReg': a_lr, 'roznica': a_mlp - a_lr,
                       'epoka_best': int(h['best_epoch'] + 1),
                       'n_train': len(tr), 'n_test': len(te)})

tab_cv = pd.DataFrame(wiersze_cv)
print('6.1 Wyniki ręcznej CV:')
display(tab_cv.round(5))
n_te = tab_cv['n_test'].mean()
n_tr = tab_cv['n_train'].mean()
sr_mlp, ci_naiwny, ci_kor = przedzialy(tab_cv['ROC_AUC_MLP'], n_te, n_tr)
sr_diff, _, ci_diff = przedzialy(tab_cv['roznica'], n_te, n_tr)
ci_kor_uciete = (float(max(0, ci_kor[0])), float(min(1, ci_kor[1])))
print(f'MLP średnie AUC = {sr_mlp:.4f}')
print(f'  CI zwykły 95%:     [{ci_naiwny[0]:.4f}, {ci_naiwny[1]:.4f}]')
print(f'  CI Nadeau-Bengio:  [{ci_kor[0]:.4f}, {ci_kor[1]:.4f}]')
if ci_kor_uciete != ci_kor:
    print(f'  Po obcięciu do [0,1]: {ci_kor_uciete} (przybliżenie t blisko 1 jest zgrubne).')
print(f'Różnica MLP - LogReg = {sr_diff:+.4f}, skorygowany CI 95%: [{ci_diff[0]:+.4f}, {ci_diff[1]:+.4f}]')
print('Hipoteza o różnicy: ' + ('CI zawiera 0 - brak podstaw do stwierdzenia przewagi MLP.'
       if ci_diff[0] <= 0 <= ci_diff[1] else 'CI nie obejmuje 0 - zaobserwowano różnicę w tej procedurze.'))

# ====================
# Sekcja źródłowa notebooka, komórka 72
# ====================
# Permutation importance na teście - opis wpływu cech, nie wybór modelu po teście.
pi_mlp = permutation_importance(pipe_mlp, X_test, y_test, scoring='roc_auc',
                                n_repeats=N_PERM, random_state=SEED)
pi_hgb = permutation_importance(modele_klas['HistGB'], X_test, y_test,
                                scoring='roc_auc', n_repeats=N_PERM, random_state=SEED)
nazwy_cech = np.asarray(X_test.columns)
top_mlp = np.argsort(pi_mlp.importances_mean)[-10:][::-1]
top_hgb = np.argsort(pi_hgb.importances_mean)[-10:][::-1]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
for ax, pi, ids, nazwa in [(axes[0], pi_mlp, top_mlp, 'MLP'),
                           (axes[1], pi_hgb, top_hgb, 'HistGB')]:
    ax.barh(nazwy_cech[ids][::-1], pi.importances_mean[ids][::-1],
            xerr=pi.importances_std[ids][::-1])
    ax.set(xlabel='spadek ROC-AUC', title=f'7.1 Ważność permutacyjna: {nazwa}')
plt.tight_layout(); plt.show()
czesci_wspolne = sorted(set(nazwy_cech[top_mlp]) & set(nazwy_cech[top_hgb]))
print('Wspólne cechy top-10:', czesci_wspolne)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(X_train[nazwy_cech[top_mlp]].corr(), cmap='coolwarm', center=0,
            annot=True, fmt='.1f', annot_kws={'size': 8}, ax=ax)
ax.set_title('7.1 Korelacje cech z top-10 MLP (train)')
plt.tight_layout(); plt.show()
print('Wysoka korelacja rozdziela wagę między cechy zastępcze; ważność pojedynczej cechy może być mała.')
print('Przy AUC bliskim 1 część permutacji nie zmienia rankingu, więc spadki są niewielkie.')

# ====================
# Sekcja źródłowa notebooka, komórka 75
# ====================
from scipy.stats import loguniform

przestrzen_rs = {
    'hidden': [(32,), (64, 32), (128, 64)],
    'dropout': [0.0, 0.1, 0.3],
    'lr': loguniform(1e-4, 1e-2),
    'weight_decay': loguniform(1e-6, 1e-2),
    'batch_size': [16, 32, 64],
}
cv_rs = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
wiersze_rs = []
for i, params in enumerate(ParameterSampler(przestrzen_rs, n_iter=N_ITER, random_state=SEED), 1):
    print(f'Random search {i}/{N_ITER}: {params}')
    m = make_pipeline(StandardScaler(), TorchMLPClassifier(**params))
    oceny = cross_val_score(m, X_dev, y_dev, cv=cv_rs, scoring='roc_auc')
    wiersze_rs.append({'nr': i, **params,
                       'auc_cv_mean': float(np.mean(oceny)),
                       'auc_cv_sd': float(np.std(oceny, ddof=1))})
wyniki_rs = pd.DataFrame(wiersze_rs).sort_values('auc_cv_mean', ascending=False).reset_index(drop=True)
print('8.1 Wszystkie konfiguracje (sortowane wg CV):')
display(wyniki_rs.round(5))

# Najlepszy wybór na CV dev. Test wykorzystujemy raz na końcu procedury.
best = wyniki_rs.iloc[0]
param_keys = list(przestrzen_rs)
najlepsze_param = {
    'hidden': tuple(int(v) for v in best['hidden']),
    'dropout': float(best['dropout']),
    'lr': float(best['lr']),
    'weight_decay': float(best['weight_decay']),
    'batch_size': int(best['batch_size']),
}
# Pandas oddaje np.int64 - DataLoader potrzebuje natywnego int.
pipe_best = make_pipeline(StandardScaler(), TorchMLPClassifier(**najlepsze_param))
pipe_best.fit(X_dev, y_dev)
p_best_test = pipe_best.predict_proba(X_test)[:, 1]
auc_rs_test = roc_auc_score(y_test, p_best_test)
(ci_rs_low, ci_rs_high), boot_rs = bootstrap_ci(y_test, p_best_test, roc_auc_score, B=N_BOOT)
print(f'Najlepsze CV = {best.auc_cv_mean:.4f}, TEST = {auc_rs_test:.4f}, '
      f'bootstrap 95% [{ci_rs_low:.4f}, {ci_rs_high:.4f}]')
print('Różnice między CV i testem nie są dowodem poprawy bez analizy niepewności.')

# ====================
# Sekcja źródłowa notebooka, komórka 80
# ====================
# Bonus opcjonalny: 5x5 zagnieżdżone CV i 30 wariantów to sporo dopasowań.
if os.environ.get('DL_BONUS', '0') == '1':
    from sklearn.model_selection import GridSearchCV
    pipe_bonus = Pipeline([('select', SelectKBest(f_classif)),
                           ('clf', LogisticRegression(max_iter=2000))])
    grid_bonus = {'select__k': [5, 10, 20, 50, 100],
                  'clf__C': np.logspace(-3, 2, 6)}
    n_cv_bonus = 3 if SZYBKI_TRYB else 5
    inner = StratifiedKFold(n_splits=n_cv_bonus, shuffle=True, random_state=SEED)
    outer = StratifiedKFold(n_splits=n_cv_bonus, shuffle=True, random_state=1)
    search_bonus = GridSearchCV(pipe_bonus, grid_bonus, cv=inner, scoring='accuracy')
    search_bonus.fit(X_szum, y_szum)
    # Osobny estymator dla każdej iteracji outer (sklearn klonuje go automatycznie).
    nested_scores = cross_val_score(search_bonus, X_szum, y_szum, cv=outer, scoring='accuracy')
    bonus_wynik = {'best_cv': float(search_bonus.best_score_),
                   'nested_cv_mean': float(np.mean(nested_scores)),
                   'nested_cv_std': float(np.std(nested_scores, ddof=1)),
                   'random_chance': 0.5, 'outer_folds': n_cv_bonus}
    print('Bonus: najlepszy inner-CV =', round(bonus_wynik['best_cv'], 4),
          '| nested-CV =', round(bonus_wynik['nested_cv_mean'], 4), '| prawda = 0.5')
    print('Wybór maksimum ze zbioru ocen CV daje optymizm selekcji; outer CV ocenia cały proces.')
else:
    bonus_wynik = {'status': 'pominięto', 'instrukcja': 'uruchom skrypt z --bonus'}
    print('Bonus pominięty (opcjonalny). Uruchom --bonus, jeśli chcesz go policzyć.')

# ====================
# Sekcja źródłowa notebooka, komórka 82
# ====================
# Raport końcowy: podsumowanie z realnych wartości, bez wpisywania wyników "na sztywno".
import json
from datetime import datetime

def _num(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, (float, int, str, bool)) or x is None: return x
    if isinstance(x, tuple): return [_num(v) for v in x]
    if isinstance(x, list): return [_num(v) for v in x]
    if isinstance(x, dict): return {str(k): _num(v) for k, v in x.items()}
    return str(x)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tab_test = pd.DataFrame({n: metryki_klas(y_test, p) for n, p in p_test.items()}).T
# To jest tabela raportowa: bez wyboru najlepszego modelu po teście.
tab_val.to_csv(OUTPUT_DIR / 'klasyfikacja_val.csv', encoding='utf-8-sig')
tab_test.to_csv(OUTPUT_DIR / 'klasyfikacja_test.csv', encoding='utf-8-sig')
tab_reg.to_csv(OUTPUT_DIR / 'regresja_test.csv', encoding='utf-8-sig')
tab_cv.to_csv(OUTPUT_DIR / 'zadanie_6_1_cv.csv', index=False, encoding='utf-8-sig')
wyniki_rs.to_csv(OUTPUT_DIR / 'zadanie_8_1_random_search.csv', index=False, encoding='utf-8-sig')
tab_progi.to_csv(OUTPUT_DIR / 'zadanie_3_1_progi.csv', index=False, encoding='utf-8-sig')
tab_kal.to_csv(OUTPUT_DIR / 'zadanie_3_2_kalibracja.csv', index=False, encoding='utf-8-sig')
tab_szum.to_csv(OUTPUT_DIR / 'zadanie_5_1_wyciek_selekcja.csv', index=False, encoding='utf-8-sig')
pd.DataFrame([{'model': n, 'KFold': acc_kf[n], 'GroupKFold': acc_gkf[n]}
              for n in modele_g]).to_csv(OUTPUT_DIR / 'zadanie_5_2_grupy.csv', index=False,
                                         encoding='utf-8-sig')

podsumowanie = {
    'data_uruchomienia': datetime.now().astimezone().isoformat(),
    'tryb': 'fast' if SZYBKI_TRYB else 'full',
    'bonus': bonus_wynik,
    'python_torch': torch.__version__,
    'device': str(DEVICE),
    'seed': SEED,
    'n_breast_cancer': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    'n_california': {'train': len(Xc_train), 'val': len(Xc_val), 'test': len(Xc_test)},
    'early_stopping_best_epoka': hist_bc['best_epoch'] + 1,
    'mlp_roc_auc_test': roc_auc_score(y_test, p_test['MLP']),
    'mlp_roc_auc_test_bootstrap_95': [float(lo), float(hi)],
    'klasyfikacja_val': tab_val.round(6).to_dict(orient='index'),
    'klasyfikacja_test': tab_test.round(6).to_dict(orient='index'),
    'regresja_test': tab_reg.round(6).to_dict(orient='index'),
    'prog_opt_na_walidacji': float(prog_opt),
    'progi_test': tab_progi.to_dict(orient='records'),
    'platt_brier_przed': float(brier_score_loss(y_test, p_test['MLP'])),
    'platt_brier_po': float(brier_score_loss(y_test, p_platt_test)),
    'wyciek_selekcja_acc_przed_cv': float(acc_szum_leak.mean()),
    'wyciek_selekcja_acc_pipeline': float(acc_szum_ok),
    'wyciek_grupy': {n: {'kfold': float(acc_kf[n]), 'groupkfold': float(acc_gkf[n])}
                     for n in modele_g},
    '6_1_auc_mlp_avg': float(sr_mlp),
    '6_1_ci_naive': list(ci_naiwny),
    '6_1_ci_Nadeau_Bengio': list(ci_kor),
    '6_1_ci_Nadeau_Bengio_przycięty': list(ci_kor_uciete),
    '6_1_roznica_mlp_minus_logreg': float(sr_diff),
    '6_1_ci_roznicy': list(ci_diff),
    '6_1_ci_roznicy_obejmuje_zero': bool(ci_diff[0] <= 0 <= ci_diff[1]),
    '7_1_wspolne_top10_cechy': czesci_wspolne,
    '8_1_cv_best_auc': float(best.auc_cv_mean),
    '8_1_test_auc': float(auc_rs_test),
    '8_1_test_auc_ci_bootstrap': [float(ci_rs_low), float(ci_rs_high)],
    '8_1_najlepsze_parametry': najlepsze_param,
    'liczba_wykresow': len(list((OUTPUT_DIR / 'wykresy').glob('*.png'))),
}
(OUTPUT_DIR / 'podsumowanie.json').write_text(
    json.dumps(_num(podsumowanie), ensure_ascii=False, indent=2), encoding='utf8')

wnioski = [
    f'TRYB: {podsumowanie["tryb"]}. Wersja FAST służy wyłącznie do testu działania.',
    f'Breast Cancer: ROC-AUC MLP na teście = {podsumowanie["mlp_roc_auc_test"]:.4f}.',
    f'Próg {prog_opt:.4f} dobrano na walidacji, nie na teście.',
    f'Kalibracja MLP (Brier): przed {podsumowanie["platt_brier_przed"]:.4f}, po {podsumowanie["platt_brier_po"]:.4f}.',
    f'Wyciek przez selekcję cech: {acc_szum_leak.mean():.3f} (błędnie) vs {acc_szum_ok:.3f} (pipeline).',
    f'MLP vs LogReg, 95% CI różnicy z korektą Nadeau-Bengio: [{ci_diff[0]:+.4f}, {ci_diff[1]:+.4f}].',
    ('CI różnicy obejmuje 0 - nie wykazano przewagi MLP.' if ci_diff[0] <= 0 <= ci_diff[1]
     else 'CI różnicy nie obejmuje 0 w tej procedurze.'),
    f'Regresja: MAE MLP = {tab_reg.loc["MLP", "MAE"]:.4f}; MAE HistGB = {tab_reg.loc["HistGB", "MAE"]:.4f}.',
    'California Housing: cel jest ucięty blisko wartości 5, więc analiza reszt wymaga ostrożności.',
    'Cechy silnie skorelowane mogą zastępować się w permutation importance.',
    f'Random search: najlepsze CV AUC = {best.auc_cv_mean:.4f}, test AUC = {auc_rs_test:.4f}.',
    'Wyniku testu nie użyto do wyboru architektury, progu ani hiperparametrów.',
]
(OUTPUT_DIR / 'wnioski.txt').write_text('\n'.join(wnioski) + '\n', encoding='utf8')
print('\nZapisano raport:', OUTPUT_DIR.resolve())
print('Wykresów:', podsumowanie['liczba_wykresow'])
print('Podsumowanie:', OUTPUT_DIR / 'podsumowanie.json')
