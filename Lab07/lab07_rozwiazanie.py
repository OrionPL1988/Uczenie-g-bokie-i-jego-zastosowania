#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab07 - VAE / GAN. Zadania odpowiadaja komorkom notebooka prowadzacej."""
import base64
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


class _Tee:
    def __init__(self, terminal, recorder):
        self.terminal, self.recorder = terminal, recorder
        self.encoding = getattr(terminal, 'encoding', 'utf-8')

    def write(self, text):
        self.terminal.write(text)
        self.recorder.log.write(text)
        self.recorder.pending.append(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.recorder.log.flush()

    def isatty(self):
        return False


class WynikiLab:
    """Pliki wynikowe + zapis faktycznych wyjsc komorek do .ipynb."""
    def __init__(self, lab_dir, profile, cfg, capture=False):
        self.lab_dir = Path(lab_dir).resolve()
        self.root = self.lab_dir / ('wyniki_' + profile)
        if (self.root / 'podsumowanie.json').exists() and not cfg['overwrite']:
            raise FileExistsError(
                f'W {self.root} sa juz wyniki. Zachowaj ich kopie albo uzyj --overwrite. '
                'W notebooku ustaw NADPISZ = True.')
        self.root.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.root / 'wykresy'
        self.fig_dir.mkdir(exist_ok=True)
        self.models = self.root / 'modele'
        self.models.mkdir(exist_ok=True)
        if cfg['overwrite']:
            # Czyszcze tylko katalog konkretnego profilu. FAST nie dotyka FULL.
            for p in self.fig_dir.glob('*.png'):
                p.unlink()
            for p in self.root.glob('*.csv'):
                p.unlink()
        self.report = {
            'profil': profile, 'konfiguracja': dict(cfg),
            'status': 'W_TRAKCIE', 'etap': 'start', 'seed': cfg['seed'],
            'czas_start': datetime.now(timezone.utc).isoformat(),
            'testy': {}, 'sekcje_wykonane': [], 'ostrzezenia': [], 'wykresy': [],
            'biblioteki': {'python': platform.python_version()},
            'uwagi_metodyczne': [
                'VAE: walidacja to 10% puli treningowej, nie oficjalny test.',
                'GAN i klasyfikator: trening na puli Fashion-MNIST danego profilu.',
                'Eksperyment beta: ten sam seed i budzet, niezalezne modele.',
                'Entropy/confidence/FID-lite nie sa dowodem braku mode collapse.',
                'Czasy pierwszych epok obejmuja kompilacje grafu.'],
        }
        for package in ('tensorflow', 'tensorflow-cpu', 'keras', 'numpy', 'pandas',
                        'scipy', 'scikit-learn', 'matplotlib', 'torch'):
            try:
                self.report['biblioteki'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        self.started = time.perf_counter()
        self.pending = []
        self.active = None
        self.counter = 0
        self.template = None
        self.cell_map = {}
        self.capture = capture
        self.log = (self.root / 'przebieg.log').open('w', encoding='utf-8')
        self.original_stdout = sys.stdout
        template_path = self.lab_dir / 'Lab07_uzupelniony.ipynb'
        if capture and template_path.exists():
            self.template = json.loads(template_path.read_text(encoding='utf-8'))
            for idx, cell in enumerate(self.template['cells']):
                if cell['cell_type'] == 'code':
                    cell['outputs'], cell['execution_count'] = [], None
                    source_id = cell.get('metadata', {}).get('lab07_original_cell')
                    if source_id is not None:
                        self.cell_map[source_id] = idx
            # Konfiguracja w wyeksportowanym notebooku ma zgadzac sie z przebiegiem.
            for cell in self.template['cells']:
                if cell.get('metadata', {}).get('lab07_config'):
                    cell['source'] = [
                        f"PROFIL = {profile!r}\n", f"BONUS = {cfg['bonus']!r}\n",
                        f"PYTORCH = {cfg['pytorch']!r}\n", f"SEED = {cfg['seed']}\n",
                        f"WATKI = {cfg['threads']}\n", 'NADPISZ = False\n',
                    ]
            self.template['metadata'].setdefault('language_info', {})['version'] = platform.python_version()
            self.template['metadata']['lab07_run'] = {
                'method': 'real stdout/tables/images captured from matching script cells',
                'profile': profile,
            }
        elif capture:
            self.warn('Brak Lab07_uzupelniony.ipynb obok skryptu: eksport notebooka pominiemy.')
        if capture:
            sys.stdout = _Tee(self.original_stdout, self)
        for filename in ('lab07_rozwiazanie.py', 'Lab07_uzupelniony.ipynb'):
            p = self.lab_dir / filename
            if p.exists():
                self.report.setdefault('sha256_zrodel', {})[filename] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.save()

    def warn(self, message):
        if message not in self.report['ostrzezenia']:
            self.report['ostrzezenia'].append(message)

    def _flush_stream(self):
        text = ''.join(self.pending)
        self.pending.clear()
        if text and self.template is not None and self.active is not None:
            self.template['cells'][self.active]['outputs'].append({
                'output_type': 'stream', 'name': 'stdout', 'text': text.splitlines(True)})

    def mark(self, source_id):
        self._flush_stream()
        self.active = self.cell_map.get(source_id)
        if self.template is not None and self.active is not None:
            self.counter += 1
            self.template['cells'][self.active]['execution_count'] = self.counter
        self.report['etap'] = 'komorka_' + str(source_id)
        self.save()
        print(f'\n--- Komorka {source_id} ---', flush=True)

    def save(self):
        self.report['czas_dotychczas_s'] = round(time.perf_counter() - self.started, 2)
        self._flush_stream()
        tmp = self.root / 'podsumowanie.json.tmp'
        tmp.write_text(json.dumps(self.report, ensure_ascii=False, indent=2,
                                  allow_nan=False, default=self._json_value), encoding='utf-8')
        os.replace(tmp, self.root / 'podsumowanie.json')
        if self.template is not None:
            dest = self.root / 'Lab07_z_wynikami.ipynb'
            tmp = self.root / 'Lab07_z_wynikami.ipynb.tmp'
            tmp.write_text(json.dumps(self.template, ensure_ascii=False, indent=1), encoding='utf-8')
            os.replace(tmp, dest)
        self.log.flush()

    @staticmethod
    def _json_value(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f'Nie umiem zapisac {type(x)} do JSON')

    def figure(self, fig, slug):
        import matplotlib.pyplot as plt
        self._flush_stream()
        number = len(self.report['wykresy']) + 1
        path = self.fig_dir / f'{number:02d}_{slug}.png'
        fig.savefig(path, dpi=135, bbox_inches='tight')
        self.report['wykresy'].append('wykresy/' + path.name)
        if self.template is not None and self.active is not None:
            self.template['cells'][self.active]['outputs'].append({
                'output_type': 'display_data', 'metadata': {},
                'data': {'image/png': base64.b64encode(path.read_bytes()).decode('ascii'),
                         'text/plain': ['<Figure: ' + slug + '>']}})
        elif not self.capture:
            from IPython.display import display, Image
            display(Image(filename=str(path)))
        plt.close(fig)
        self.save()

    def table(self, df, filename):
        df.to_csv(self.root / filename, index=False, encoding='utf-8')
        print(df.to_string(index=False), flush=True)
        self.save()

    def history(self, history, name):
        import numpy as np
        import pandas as pd
        h = history.history if hasattr(history, 'history') else history
        for key, vals in h.items():
            if not np.isfinite(np.asarray(vals, dtype=float)).all():
                raise FloatingPointError(f'{name}: NaN/inf w historii {key}')
        df = pd.DataFrame(h)
        df.insert(0, 'epoka', range(1, len(df) + 1))
        df.to_csv(self.root / ('historia_' + name + '.csv'), index=False)
        self.report.setdefault('historie', {})[name] = {
            'epoki': len(df), 'ostatnie': {k: float(v[-1]) for k, v in h.items() if v},
            'csv': 'historia_' + name + '.csv'}
        self.save()

    def done(self, section):
        if section not in self.report['sekcje_wykonane']:
            self.report['sekcje_wykonane'].append(section)
        self.save()

    def close(self, error=None):
        if error is not None:
            self.report['status'] = 'PRZERWANO' if isinstance(error, KeyboardInterrupt) else 'BLAD'
            self.report['blad'] = type(error).__name__ + ': ' + str(error)
            if self.template is not None and self.active is not None:
                self._flush_stream()
                self.template['cells'][self.active]['outputs'].append({
                    'output_type': 'error', 'ename': type(error).__name__, 'evalue': str(error),
                    'traceback': traceback.format_exception(type(error), error, error.__traceback__)})
        self.report['czas_eksperymentu_s'] = round(time.perf_counter() - self.started, 2)
        self.report['czas_koniec'] = datetime.now(timezone.utc).isoformat()
        self.save()
        if self.capture:
            sys.stdout = self.original_stdout
        self.log.close()


def pobierz_argumenty():
    import argparse
    p=argparse.ArgumentParser(description='Lab07: VAE, DCGAN, opcjonalnie FID-lite i PyTorch')
    g=p.add_mutually_exclusive_group(required=True)
    for profile in ('fast','balanced','full'):
        g.add_argument('--'+profile,action='store_true')
    p.add_argument('--bonus',action='store_true',help='FID-lite w cechach klasyfikatora')
    p.add_argument('--pytorch',action='store_true',help='dodatkowy GAN z sekcji 4; wymaga torch')
    p.add_argument('--threads',type=int,default=4,help='liczba watkow CPU, domyslnie 4')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite',action='store_true',help='nadpisz wyniki TEGO profilu')
    a=p.parse_args()
    if a.threads<1 or a.seed<0:
        p.error('threads >= 1 i seed >= 0')
    return a


def wykonaj(a):
    PROFIL='fast' if a.fast else 'balanced' if a.balanced else 'full'
    BONUS,PYTORCH,SEED,WATKI,NADPISZ=a.bonus,a.pytorch,a.seed,a.threads,a.overwrite
    TRYB_SKRYPTU=True
    rt=None
    try:

        # %% Komorka 3
        import os
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import tensorflow as tf
        import keras
        from keras import layers, ops
        from scipy.stats import norm

        if keras.backend.backend() != 'tensorflow':
            raise RuntimeError('Potrzebny backend TensorFlow. Zrestartuj kernel notebooka.')
        try:
            tf.config.threading.set_intra_op_parallelism_threads(WATKI)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            print('TF byl juz uruchomiony; liczby watkow nie zmieniono. Restart kernela pozwoli ja ustawic.')

        if '__file__' in globals():
            KATALOG_LAB = Path(__file__).resolve().parent
        else:
            # Wynikowy notebook moze byc otwarty wewnatrz wyniki_full.
            KATALOG_LAB = next((p for p in (Path.cwd(), *Path.cwd().parents)
                                if (p/'lab07_rozwiazanie.py').exists()), Path.cwd())
        CFG = dict(profile=PROFIL, bonus=BONUS, pytorch=PYTORCH, seed=SEED,
                   threads=WATKI, overwrite=NADPISZ)
        rt = WynikiLab(KATALOG_LAB, PROFIL, CFG, capture=TRYB_SKRYPTU)
        RAPORT = rt.report
        if TRYB_SKRYPTU:
            rt.mark(3)
        print('TensorFlow:', tf.__version__, '| Keras:', keras.__version__, '| backend:', keras.backend.backend())

        # %% Komorka 4
        rt.mark(4)
        if PROFIL not in ('fast', 'balanced', 'full'):
            raise ValueError('PROFIL: fast, balanced albo full')
        keras.utils.set_random_seed(SEED)
        SZYBKI_TRYB = PROFIL == 'fast'
        PROFILE = {
            'fast': dict(n=2000, test=2000, vae=2, krotko=1, gan=2, torch=1, latent=64, batch=64, clf=1, gen=1000),
            'balanced': dict(n=12000, test=2500, vae=8, krotko=4, gan=8, torch=3, latent=128, batch=128, clf=3, gen=2000),
            'full': dict(n=60000, test=10000, vae=30, krotko=10, gan=30, torch=10, latent=128, batch=128, clf=5, gen=5000),
        }
        cfg = PROFILE[PROFIL]
        N_TRAIN, N_TEST = cfg['n'], cfg['test']
        EPOKI_VAE, EPOKI_KROTKO = cfg['vae'], cfg['krotko']
        EPOKI_GAN, EPOKI_TORCH = cfg['gan'], cfg['torch']
        LATENT_GAN, BATCH_GAN, N_GEN = cfg['latent'], cfg['batch'], cfg['gen']
        RAPORT['limity'] = dict(cfg)
        RAPORT['gpu_tensorflow'] = [str(g) for g in tf.config.list_physical_devices('GPU')]
        RAPORT['opcjonalne'] = {'pytorch': 'oczekuje' if PYTORCH else 'pominieto',
                               'fid_lite': 'oczekuje' if BONUS else 'pominieto'}
        if PROFIL != 'full':
            rt.warn('FAST/BALANCED to test przeplywu. Nie traktuj tych pomiarow jako finalnego Lab07.')
        if PYTORCH:
            try:
                import torch
            except ImportError as e:
                raise RuntimeError('Opcja --pytorch wymaga: pip install -r requirements_pytorch.txt') from e
        print('Profil:', PROFIL, '| parametry:', cfg, '| GPU TF:', RAPORT['gpu_tensorflow'] or 'brak, CPU')

        class KontrolaEpoki(keras.callbacks.Callback):
            def __init__(self, etykieta):
                super().__init__()
                self.etykieta = etykieta
                self.czasy = []
            def on_epoch_begin(self, epoch, logs=None):
                self.t0 = time.perf_counter()
            def on_epoch_end(self, epoch, logs=None):
                dt = time.perf_counter() - self.t0
                self.czasy.append(dt)
                for key, value in (logs or {}).items():
                    if not np.isfinite(float(value)):
                        raise FloatingPointError(f'{self.etykieta}: {key} nie jest skonczone')
                RAPORT.setdefault('postep', {})[self.etykieta] = {
                    'epoka': epoch + 1, 'czas_epoki_s': round(dt, 3),
                    'metryki': {k: float(v) for k, v in (logs or {}).items()}}
                rt.save()

        # Male pule watkow zapobiegaja tworzeniu kolejnych wielkich pul dla kazdego fit().
        def ds_obrazow(x, batch=128, shuffle=False, seed=SEED):
            ds = tf.data.Dataset.from_tensor_slices(x)
            if shuffle:
                ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
            options = tf.data.Options()
            options.threading.private_threadpool_size = max(1, WATKI)
            options.threading.max_intra_op_parallelism = 1
            return ds.batch(batch).with_options(options).prefetch(1)

        def rysuj_metryke(hist, key, title, slug, ylabel='wartosc na obraz'):
            h = hist.history if hasattr(hist, 'history') else hist
            fig, ax = plt.subplots(figsize=(7.2, 4.1))
            ep = np.arange(1, len(h[key]) + 1)
            ax.plot(ep, h[key], 'o-', label='train')
            if 'val_' + key in h:
                ax.plot(ep, h['val_' + key], 's--', label='walidacja')
            ax.set(xlabel='epoka', ylabel=ylabel, title=title)
            ax.legend(); ax.grid(alpha=.25)
            fig.tight_layout(); rt.figure(fig, slug)

        rt.save()

        # %% Komorka 5
        rt.mark(5)
        # Keras sam korzysta z cache; zadnych zastepczych danych po bledzie pobierania.
        try:
            (xm_tr, ym_tr), (xm_te, ym_te) = keras.datasets.mnist.load_data()
            (xf_tr, yf_tr), (xf_te, yf_te) = keras.datasets.fashion_mnist.load_data()
        except Exception as e:
            raise RuntimeError('Nie udalo sie wczytac MNIST/Fashion-MNIST. Sprawdz internet i cache ~/.keras/datasets.') from e
        x_mnist = (xm_tr[:N_TRAIN].astype('float32') / 255.0)[..., None]
        y_mnist = ym_tr[:N_TRAIN]
        x_mnist_te = (xm_te[:N_TEST].astype('float32') / 255.0)[..., None]
        y_mnist_te = ym_te[:N_TEST]
        x_fash = (xf_tr[:N_TRAIN].astype('float32') / 127.5 - 1.0)[..., None]
        y_fash = yf_tr[:N_TRAIN]
        x_fash_te = (xf_te[:N_TEST].astype('float32') / 127.5 - 1.0)[..., None]
        y_fash_te = yf_te[:N_TEST]
        KLASY_FASHION = ['T-shirt', 'spodnie', 'sweter', 'sukienka', 'plaszcz',
                         'sandal', 'koszula', 'sneaker', 'torba', 'botek']
        # Zmiana metodyczna wzgledem szablonu: test nie jest walidacja podczas fit().
        idx = np.random.default_rng(SEED).permutation(len(x_mnist))
        n_val = len(idx) // 10
        idx_val, idx_train = idx[:n_val], idx[n_val:]
        x_vae_train, x_vae_val = x_mnist[idx_train], x_mnist[idx_val]
        assert len(np.intersect1d(idx_val, idx_train)) == 0
        assert x_mnist.shape[1:] == (28, 28, 1) and x_fash.shape[1:] == (28, 28, 1)
        assert 0 <= x_mnist.min() <= x_mnist.max() <= 1
        assert -1 <= x_fash.min() <= x_fash.max() <= 1
        RAPORT['dane'] = {'mnist_pula_train': len(x_mnist), 'vae_train': len(x_vae_train),
                          'vae_val': len(x_vae_val), 'mnist_test': len(x_mnist_te),
                          'fashion_train': len(x_fash), 'fashion_test': len(x_fash_te),
                          'dane': 'MNIST i Fashion-MNIST z keras.datasets'}
        RAPORT['testy']['zakresy_i_podzial'] = True
        print('MNIST train/val/test:', len(x_vae_train), len(x_vae_val), len(x_mnist_te))
        print('Fashion train/test:', len(x_fash), len(x_fash_te))
        rt.done('dane')

        # %% Komorka 6
        rt.mark(6)
        def plansza(obrazy, n_wierszy, n_kolumn, tytul='', zakres=(0, 1), rozmiar=1.0, slug='plansza'):
            obrazy = np.asarray(obrazy).reshape(-1, 28, 28)[:n_wierszy*n_kolumn]
            # Jeden obraz zbiorczy; brak zachodzacych na siebie tytulow podwykresow.
            canvas = np.full((28*n_wierszy, 28*n_kolumn), zakres[0], dtype='float32')
            for i, im in enumerate(obrazy):
                r, c = divmod(i, n_kolumn)
                canvas[r*28:(r+1)*28, c*28:(c+1)*28] = im
            fig, ax = plt.subplots(figsize=(n_kolumn*rozmiar, n_wierszy*rozmiar + .45))
            ax.imshow(canvas, cmap='gray', vmin=zakres[0], vmax=zakres[1])
            ax.set_title(tytul, fontsize=11, pad=12); ax.axis('off')
            fig.tight_layout(); rt.figure(fig, slug)

        plansza(x_mnist[:16], 2, 8, 'MNIST - dane VAE', slug='dane_mnist')
        plansza(x_fash[:16], 2, 8, 'Fashion-MNIST - dane GAN', zakres=(-1, 1), slug='dane_fashion')

        # %% Komorka 10
        rt.mark(10)
        @keras.saving.register_keras_serializable(package='Lab07')
        class Sampling(layers.Layer):
            def __init__(self, seed=42, **kwargs):
                super().__init__(**kwargs)
                self.seed = seed
                self.seed_generator = keras.random.SeedGenerator(seed)
            def call(self, inputs):
                z_mean, z_log_var = inputs
                eps = keras.random.normal(ops.shape(z_mean), seed=self.seed_generator)
                # log_var to log wariancji, nie odchylenia. Stad 0.5 w exp().
                return z_mean + ops.exp(0.5*z_log_var) * eps
            def get_config(self):
                return {**super().get_config(), 'seed': self.seed}

        mu_t = np.full((20000, 2), 3.0, dtype='float32')
        lv_t = np.full((20000, 2), np.log(.25), dtype='float32')
        z_t = ops.convert_to_numpy(Sampling(seed=SEED)([mu_t, lv_t]))
        assert np.allclose(z_t.mean(0), 3, atol=.025)
        assert np.allclose(z_t.std(0), .5, atol=.025)
        RAPORT['zad_2_1'] = {'srednia': z_t.mean(0), 'std': z_t.std(0)}
        RAPORT['testy']['sampling'] = True
        print('Sampling - srednia:', z_t.mean(0), '| std:', z_t.std(0))
        rt.done('2.1')

        # %% Komorka 12
        rt.mark(12)
        LATENT_VAE = 2

        def zbuduj_enkoder_vae(latent_dim=LATENT_VAE):
            wej = keras.Input(shape=(28, 28, 1))
            h = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(wej)   # 14x14x32
            h = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(h)     # 7x7x64
            h = layers.Flatten()(h)
            h = layers.Dense(16, activation="relu")(h)
            z_mean = layers.Dense(latent_dim, name="z_mean")(h)         # średnia rozkładu q(z|x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)   # log wariancji (dowolna liczba rzeczywista)
            z = Sampling(seed=SEED, name="z")([z_mean, z_log_var])                 # próbka z rozkładu (trik reparametryzacji)
            return keras.Model(wej, [z_mean, z_log_var, z], name="vae_enkoder")

        def zbuduj_dekoder_vae(latent_dim=LATENT_VAE):
            wej = keras.Input(shape=(latent_dim,))
            h = layers.Dense(7 * 7 * 64, activation="relu")(wej)
            h = layers.Reshape((7, 7, 64))(h)
            h = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(h)   # 14x14
            h = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(h)   # 28x28
            wyj = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(h)
            return keras.Model(wej, wyj, name="vae_dekoder")

        enc_vae = zbuduj_enkoder_vae()
        dec_vae = zbuduj_dekoder_vae()
        enc_vae.summary()

        # %% Komorka 15
        rt.mark(15)
        class VAE(keras.Model):
            def __init__(self, enkoder, dekoder, beta=1.0, **kwargs):
                super().__init__(**kwargs)
                self.enkoder, self.dekoder, self.beta = enkoder, dekoder, float(beta)
                self.loss_tracker = keras.metrics.Mean(name='loss')
                self.rec_tracker = keras.metrics.Mean(name='rec_loss')
                self.kl_tracker = keras.metrics.Mean(name='kl_loss')
            @property
            def metrics(self):
                return [self.loss_tracker, self.rec_tracker, self.kl_tracker]
            def call(self, x, training=False):
                _, _, z = self.enkoder(x, training=training)
                return self.dekoder(z, training=training)
            def oblicz_straty(self, data, training):
                z_mean, z_log_var, z = self.enkoder(data, training=training)
                rek = self.dekoder(z, training=training)
                # BCE redukuje kanal, zostaje (batch, 28, 28). Sumujemy piksele.
                rec_loss = ops.mean(ops.sum(keras.losses.binary_crossentropy(data, rek), axis=(1, 2)))
                kl_loss = ops.mean(ops.sum(-.5*(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)), axis=1))
                return rec_loss + self.beta*kl_loss, rec_loss, kl_loss
            def _aktualizuj(self, total, rec, kl, batch):
                for metric, value in zip(self.metrics, (total, rec, kl)):
                    # Ostatni, mniejszy batch nie dostaje wagi calego batcha.
                    metric.update_state(value, sample_weight=ops.cast(batch, 'float32'))
                return {m.name: m.result() for m in self.metrics}
            def train_step(self, data):
                data = keras.utils.unpack_x_y_sample_weight(data)[0]
                with tf.GradientTape() as tape:
                    total, rec, kl = self.oblicz_straty(data, training=True)
                    tf.debugging.assert_all_finite(total, 'VAE: strata NaN/inf')
                grads = tape.gradient(total, self.trainable_weights)
                if any(g is None for g in grads):
                    raise RuntimeError('VAE: gradient nie dochodzi do wszystkich wag')
                for g in grads:
                    tf.debugging.assert_all_finite(g, 'VAE: gradient NaN/inf')
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return self._aktualizuj(total, rec, kl, ops.shape(data)[0])
            def test_step(self, data):
                data = keras.utils.unpack_x_y_sample_weight(data)[0]
                total, rec, kl = self.oblicz_straty(data, training=False)
                tf.debugging.assert_all_finite(total, 'VAE: walidacja NaN/inf')
                return self._aktualizuj(total, rec, kl, ops.shape(data)[0])

        # Test skali: stale p=0.5 daje 784*ln(2), a N(0,I) ma KL=0.
        zeros = ops.zeros((2, 28, 28, 1))
        half = ops.ones_like(zeros)*.5
        bce_test = float(ops.convert_to_numpy(ops.mean(ops.sum(keras.losses.binary_crossentropy(zeros, half), axis=(1, 2)))))
        assert np.isclose(bce_test, 784*np.log(2), rtol=1e-5)
        zero_kl = ops.sum(-.5*(1 + ops.zeros((2,2)) - ops.zeros((2,2)) - ops.exp(ops.zeros((2,2)))))
        assert np.isclose(float(ops.convert_to_numpy(zero_kl)), 0)
        RAPORT['testy']['skala_bce_i_kl'] = True
        print('BCE dla p=0.5:', bce_test, '| KL N(0,I): 0')

        # %% Komorka 16
        rt.mark(16)
        vae = VAE(enc_vae, dec_vae, beta=1.0)
        # Zbuduj model przed save_weights; samo oblicz_straty nie wywoluje VAE.call.
        _ = vae(x_vae_train[:2], training=False)
        vae.compile(optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)
        t0 = time.perf_counter()
        hist_vae = vae.fit(ds_obrazow(x_vae_train, shuffle=True),
                          validation_data=ds_obrazow(x_vae_val), epochs=EPOKI_VAE,
                          verbose=2, callbacks=[KontrolaEpoki('vae')])
        rt.history(hist_vae, 'vae')
        RAPORT['zad_2_2'] = {'czas_s': round(time.perf_counter()-t0, 2),
                            'parametry': vae.count_params(),
                            'train_ostatnie': {k: float(v[-1]) for k,v in hist_vae.history.items()},
                            'ocena_test': vae.evaluate(ds_obrazow(x_mnist_te), return_dict=True, verbose=0)}
        vae.save_weights(rt.models / 'vae_ostatni.weights.h5')
        # Te modele pozwalaja pozniej odtworzyc rysunki bez ponownego treningu.
        enc_vae.save(rt.models / 'vae_enkoder.keras')
        dec_vae.save(rt.models / 'vae_dekoder.keras')
        if hist_vae.history['kl_loss'][-1] < .01:
            rt.warn('KL VAE bliskie zeru: obejrzyj rekonstrukcje, mozliwy posterior collapse lub za krotki trening.')
        rt.done('2.2')

        # %% Komorka 17
        rt.mark(17)
        for key, title in [('loss','VAE - strata calkowita'), ('rec_loss','VAE - rekonstrukcja BCE'), ('kl_loss','VAE - KL')]:
            rysuj_metryke(hist_vae, key, title, 'vae_'+key)

        # %% Komorka 19
        rt.mark(19)
        z_mean_te, z_log_var_te, _ = enc_vae.predict(x_mnist_te, batch_size=128, verbose=0)
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        sc = ax.scatter(z_mean_te[:,0], z_mean_te[:,1], c=y_mnist_te, s=4, alpha=.65, cmap='tab10')
        ax.legend(*sc.legend_elements(), title='cyfra', fontsize=7)
        ax.set(xlabel='z1 (srednia)', ylabel='z2 (srednia)', title='VAE - z_mean oficjalnego zbioru testowego')
        fig.tight_layout(); rt.figure(fig, 'vae_latent_test')
        n = 15
        q = np.linspace(.05, .95, n)
        siatka_z = np.array([[norm.ppf(a),norm.ppf(b)] for b in q[::-1] for a in q], dtype='float32')
        gen = dec_vae.predict(siatka_z, verbose=0)
        plansza(gen, n, n, 'VAE - siatka kwantyli N(0,1)', rozmiar=.6, slug='vae_siatka_kwantyli')
        RAPORT['latent_vae'] = {'srednia_mu': z_mean_te.mean(0), 'std_mu': z_mean_te.std(0),
                               'srednia_sigma': np.exp(.5*z_log_var_te).mean(0)}
        print('Latent VAE:', RAPORT['latent_vae']); rt.save()

        # %% Komorka 20
        rt.mark(20)
        rek_1 = vae.predict(x_mnist_te[:10], verbose=0)
        rek_2 = vae.predict(x_mnist_te[:10], verbose=0)
        plansza(np.concatenate([x_mnist_te[:10],rek_1,rek_2]), 3, 10,
                'Oryginal / rekonstrukcja 1 / rekonstrukcja 2', slug='vae_rekonstrukcje')
        RAPORT['roznica_dwoch_losowan_vae'] = float(np.abs(rek_1-rek_2).mean())
        print('Srednia roznica dwoch losowan:', RAPORT['roznica_dwoch_losowan_vae']); rt.save()

        # %% Komorka 22
        rt.mark(22)
        z_los = np.random.default_rng(SEED).normal(size=(32, LATENT_VAE)).astype('float32')
        plansza(dec_vae.predict(z_los, verbose=0), 4, 8, 'VAE - 32 nowe probki', slug='vae_losowe_probki')

        # %% Komorka 24
        rt.mark(24)
        ts = np.linspace(0, 1, 10, dtype='float32')
        if not np.any(y_mnist_te == 1) or not np.any(y_mnist_te == 8):
            raise ValueError('W probce testowej brakuje cyfry 1 albo 8')
        x_a = x_mnist_te[np.flatnonzero(y_mnist_te == 1)[0]]
        x_b = x_mnist_te[np.flatnonzero(y_mnist_te == 8)[0]]
        wiersz_piksele = np.array([(1-t)*x_a + t*x_b for t in ts]).reshape(10,28,28)
        z_a_vae, _, _ = enc_vae.predict(x_a[None], verbose=0)
        z_b_vae, _, _ = enc_vae.predict(x_b[None], verbose=0)
        z_interp = (1-ts[:,None])*z_a_vae + ts[:,None]*z_b_vae
        wiersz_latent = dec_vae.predict(z_interp, verbose=0).reshape(10,28,28)
        assert np.allclose(wiersz_piksele[0], x_a[...,0])
        assert np.allclose(wiersz_piksele[-1], x_b[...,0])
        plansza(np.concatenate([wiersz_piksele, wiersz_latent]), 2, 10,
                '1 -> 8: piksele (gora), latent VAE (dol)', slug='vae_interpolacja')
        RAPORT['zad_2_3'] = {'ksztalt_piksele': list(wiersz_piksele.shape),
                            'ksztalt_latent': list(wiersz_latent.shape)}
        RAPORT['testy']['interpolacja_vae'] = True
        rt.done('2.3')

        # %% Komorka 26
        rt.mark(26)
        wyniki_beta = []
        for beta in (.1, 1., 4.):
            # Ten sam start, nowy model i optimizer. Nie kontynuujemy poprzedniego VAE.
            keras.utils.set_random_seed(SEED)
            en_b, de_b = zbuduj_enkoder_vae(), zbuduj_dekoder_vae()
            v_b = VAE(en_b, de_b, beta=beta)
            _ = v_b(x_vae_train[:2], training=False)
            v_b.compile(optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)
            h_b = v_b.fit(ds_obrazow(x_vae_train, shuffle=True),
                         validation_data=ds_obrazow(x_vae_val), epochs=EPOKI_KROTKO,
                         verbose=2, callbacks=[KontrolaEpoki('beta_'+str(beta))])
            rt.history(h_b, 'beta_'+str(beta).replace('.','_'))
            mu_b, lv_b, _ = en_b.predict(x_mnist_te, batch_size=128, verbose=0)
            results = {'beta': beta, 'rec_loss': float(h_b.history['rec_loss'][-1]),
                       'kl_loss': float(h_b.history['kl_loss'][-1]),
                       'val_rec_loss': float(h_b.history['val_rec_loss'][-1]),
                       'val_kl_loss': float(h_b.history['val_kl_loss'][-1]),
                       'std_mu_z1_test': float(mu_b.std(0)[0]), 'std_mu_z2_test': float(mu_b.std(0)[1]),
                       'mean_sigma_test': float(np.exp(.5*lv_b).mean()), 'epoki': EPOKI_KROTKO}
            wyniki_beta.append(results)
            fig, ax = plt.subplots(figsize=(6.2,4.8))
            sc = ax.scatter(mu_b[:,0], mu_b[:,1], c=y_mnist_te, s=3, alpha=.6, cmap='tab10')
            ax.set(xlabel='z1', ylabel='z2', title=f'VAE beta={beta} - przestrzen latentna (test)')
            ax.legend(*sc.legend_elements(), fontsize=6, title='cyfra')
            fig.tight_layout(); rt.figure(fig, 'beta_'+str(beta).replace('.','_')+'_latent')
            del v_b, en_b, de_b
        rt.table(pd.DataFrame(wyniki_beta), 'porownanie_beta.csv')
        rec = np.array([r['rec_loss'] for r in wyniki_beta])
        kl = np.array([r['kl_loss'] for r in wyniki_beta])
        trend_rec = bool(np.all(np.diff(rec) >= 0))
        trend_kl = bool(np.all(np.diff(kl) <= 0))
        werdykt_beta = ('zgodna z kierunkiem hipotezy' if trend_rec and trend_kl else
                        'czesciowo zgodna' if trend_rec or trend_kl else 'niepotwierdzona w tym przebiegu')
        RAPORT['zad_2_4'] = {'wyniki': wyniki_beta, 'hipoteza': werdykt_beta,
                            'uwaga': 'Jedno ziarno; tabela i chmury latentu wymagaja interpretacji. Brak testu istotnosci.'}
        print('Beta:', werdykt_beta)
        rt.done('2.4')

        # %% Komorka 28
        rt.mark(28)
        def zbuduj_dyskryminator():
            wej = keras.Input(shape=(28, 28, 1))
            h = layers.Conv2D(64, 4, strides=2, padding="same")(wej)          # 14x14x64
            h = layers.LeakyReLU(negative_slope=0.2)(h)
            h = layers.Conv2D(128, 4, strides=2, padding="same")(h)           # 7x7x128
            h = layers.LeakyReLU(negative_slope=0.2)(h)
            h = layers.Flatten()(h)
            h = layers.Dropout(0.3)(h)                                        # osłabia D, stabilizuje grę
            logit = layers.Dense(1)(h)                                        # logit (bez sigmoidy!)
            return keras.Model(wej, logit, name="dyskryminator")

        D = zbuduj_dyskryminator()
        D.summary()

        # %% Komorka 30
        rt.mark(30)
        def zbuduj_generator(latent_dim=LATENT_GAN):
            wej = keras.Input(shape=(latent_dim,))
            h = layers.Dense(7*7*128, use_bias=False)(wej)
            h = layers.BatchNormalization()(h)
            h = layers.LeakyReLU(negative_slope=.2)(h)
            h = layers.Reshape((7,7,128))(h)
            h = layers.Conv2DTranspose(64,4,strides=2,padding='same',use_bias=False)(h)
            h = layers.BatchNormalization()(h)
            h = layers.LeakyReLU(negative_slope=.2)(h)
            wyj = layers.Conv2DTranspose(1,4,strides=2,padding='same',activation='tanh')(h)
            return keras.Model(wej, wyj, name='generator')

        G = zbuduj_generator()
        z_test = keras.random.normal((4,LATENT_GAN), seed=SEED)
        x_test_g = G(z_test, training=False)
        assert G.output_shape == (None,28,28,1)
        assert tuple(D(x_test_g).shape) == (4,1)
        assert np.isfinite(ops.convert_to_numpy(x_test_g)).all()
        assert np.max(np.abs(ops.convert_to_numpy(x_test_g))) <= 1
        RAPORT['zad_3_1'] = {'parametry_G': G.count_params(), 'parametry_D': D.count_params(),
                            'latent_gan': LATENT_GAN, 'wyjscie_G': list(G.output_shape)}
        RAPORT['testy']['ksztalty_G_D'] = True
        G.summary(); rt.done('3.1')

        # %% Komorka 32
        rt.mark(32)
        class GAN(keras.Model):
            def __init__(self, generator, discriminator, latent_dim, **kwargs):
                super().__init__(**kwargs)
                self.generator, self.discriminator, self.latent_dim = generator, discriminator, latent_dim
                self.seed_generator = keras.random.SeedGenerator(SEED)
                self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
                self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
                self.d_real_tracker = keras.metrics.Mean(name='D_real')
                self.d_fake_tracker = keras.metrics.Mean(name='D_fake')
            def compile(self, d_optimizer, g_optimizer, loss_fn):
                super().compile(jit_compile=False)
                self.d_optimizer, self.g_optimizer, self.loss_fn = d_optimizer, g_optimizer, loss_fn
                self.d_optimizer.build(self.discriminator.trainable_variables)
                self.g_optimizer.build(self.generator.trainable_variables)
            @property
            def metrics(self):
                return [self.d_loss_tracker,self.g_loss_tracker,self.d_real_tracker,self.d_fake_tracker]
            def call(self, z, training=False):
                return self.generator(z, training=training)
            def train_step(self, real):
                real = keras.utils.unpack_x_y_sample_weight(real)[0]
                batch = ops.shape(real)[0]
                jedynki, zera = ops.ones((batch,1)), ops.zeros((batch,1))
                z = keras.random.normal((batch,self.latent_dim), seed=self.seed_generator)
                with tf.GradientTape() as tape_d:
                    fake = self.generator(z, training=True)
                    # Do D trafia obraz, ale gradient tej straty nie ma trenowac G.
                    logit_real = self.discriminator(real, training=True)
                    logit_fake = self.discriminator(tf.stop_gradient(fake), training=True)
                    d_loss = self.loss_fn(.9*jedynki, logit_real) + self.loss_fn(zera, logit_fake)
                grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_weights)
                if any(g is None for g in grads_d):
                    raise RuntimeError('GAN: brak gradientu D')
                for grad in grads_d:
                    tf.debugging.assert_all_finite(grad, 'GAN gradient D NaN/inf')
                self.d_optimizer.apply_gradients(zip(grads_d,self.discriminator.trainable_weights))

                # Osobna tasma i nowy szum. Tym razem etykieta 1, bez smoothingu.
                z = keras.random.normal((batch,self.latent_dim), seed=self.seed_generator)
                with tf.GradientTape() as tape_g:
                    fake_g = self.generator(z, training=True)
                    g_loss = self.loss_fn(jedynki, self.discriminator(fake_g, training=True))
                grads_g = tape_g.gradient(g_loss, self.generator.trainable_weights)
                if any(g is None for g in grads_g):
                    raise RuntimeError('GAN: brak gradientu G')
                for grad in grads_g:
                    tf.debugging.assert_all_finite(grad, 'GAN gradient G NaN/inf')
                self.g_optimizer.apply_gradients(zip(grads_g,self.generator.trainable_weights))
                tf.debugging.assert_all_finite(d_loss, 'GAN loss D NaN/inf')
                tf.debugging.assert_all_finite(g_loss, 'GAN loss G NaN/inf')
                values = [d_loss,g_loss,ops.mean(ops.sigmoid(logit_real)),ops.mean(ops.sigmoid(logit_fake))]
                for metric, value in zip(self.metrics, values):
                    metric.update_state(value, sample_weight=ops.cast(batch,'float32'))
                return {m.name:m.result() for m in self.metrics}

        # %% Komorka 34
        rt.mark(34)
        class MonitorGAN(keras.callbacks.Callback):
            def __init__(self, latent_dim, n=16):
                super().__init__()
                self.z = keras.random.normal((n,latent_dim), seed=123)
                self.migawki = []
            def on_epoch_end(self, epoch, logs=None):
                obr = ops.convert_to_numpy(self.model.generator(self.z, training=False))
                if not np.isfinite(obr).all():
                    raise FloatingPointError('GAN: NaN/inf w wygenerowanych obrazach')
                self.migawki.append(obr)
                # To same wagi, nie pelny stan optimizerow. Nie udajemy automatycznego resume.
                if (epoch+1)%5 == 0 or epoch+1 == EPOKI_GAN:
                    self.model.generator.save_weights(rt.models/'generator_ostatni.weights.h5')
                    self.model.discriminator.save_weights(rt.models/'dyskryminator_ostatni.weights.h5')
                np.savez_compressed(rt.models/'migawki_gan.npz', obrazy=np.asarray(self.migawki), z=ops.convert_to_numpy(self.z))

        options = tf.data.Options()
        options.threading.private_threadpool_size = max(1,WATKI)
        options.threading.max_intra_op_parallelism = 1
        ds_gan = (tf.data.Dataset.from_tensor_slices(x_fash)
                  .shuffle(len(x_fash),seed=SEED,reshuffle_each_iteration=True)
                  .batch(BATCH_GAN,drop_remainder=True).with_options(options).prefetch(1))
        keras.utils.set_random_seed(SEED)
        G, D = zbuduj_generator(), zbuduj_dyskryminator()
        gan = GAN(G,D,LATENT_GAN)
        _ = gan(ops.zeros((2,LATENT_GAN)), training=False)
        gan.compile(keras.optimizers.Adam(2e-4,beta_1=.5), keras.optimizers.Adam(2e-4,beta_1=.5),
                    keras.losses.BinaryCrossentropy(from_logits=True))
        monitor = MonitorGAN(LATENT_GAN)
        t0 = time.perf_counter()
        hist_gan = gan.fit(ds_gan,epochs=EPOKI_GAN,verbose=2,callbacks=[monitor,KontrolaEpoki('gan')])
        rt.history(hist_gan,'gan')
        RAPORT['zad_3_2'] = {'czas_s': round(time.perf_counter()-t0,2),
                            'epoki': len(hist_gan.history['d_loss']),
                            'ostatnie': {k:float(v[-1]) for k,v in hist_gan.history.items()}}
        G.save(rt.models/'generator.keras'); D.save(rt.models/'dyskryminator.keras')
        if hist_gan.history['D_fake'][-1] < .05:
            rt.warn('D_fake < 0.05: D silnie odrzuca probki G. Sprawdz plansze i przebieg uczenia.')
        rt.done('3.2')

        # %% Komorka 35
        rt.mark(35)
        h = hist_gan.history
        fig, ax = plt.subplots(figsize=(7.2,4.2))
        ep = np.arange(1,len(h['d_loss'])+1)
        ax.plot(ep,h['d_loss'],'o-',label='D'); ax.plot(ep,h['g_loss'],'s-',label='G')
        ax.set(xlabel='epoka',ylabel='BCE',title='GAN - straty G i D'); ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout(); rt.figure(fig,'gan_straty')
        fig, ax = plt.subplots(figsize=(7.2,4.2))
        ax.plot(ep,h['D_real'],'o-',label='D(x), prawdziwe')
        ax.plot(ep,h['D_fake'],'s-',label='D(G(z)), generowane')
        ax.axhline(.5,ls='--',label='0.5 - odniesienie')
        ax.set(xlabel='epoka',ylabel='srednie prawdopodobienstwo',title='GAN - wyjscie dyskryminatora')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); rt.figure(fig,'gan_diagnostyki_D')

        # %% Komorka 36
        rt.mark(36)
        epoki_do_pokazu = sorted(set(np.linspace(0,len(monitor.migawki)-1,min(6,len(monitor.migawki))).astype(int)))
        canvas = np.concatenate([np.concatenate([monitor.migawki[e][c,:,:,0] for c in range(8)],axis=1) for e in epoki_do_pokazu],axis=0)
        fig, ax = plt.subplots(figsize=(9,1.1*len(epoki_do_pokazu)+.6))
        ax.imshow(canvas,cmap='gray',vmin=-1,vmax=1)
        ax.set_xticks([]); ax.set_yticks([28*i+13.5 for i in range(len(epoki_do_pokazu))],
                                       labels=['epoka '+str(e+1) for e in epoki_do_pokazu])
        ax.set_title('GAN - ewolucja tych samych 8 wektorow szumu'); fig.tight_layout()
        rt.figure(fig,'gan_ewolucja_probek')
        plansza(ops.convert_to_numpy(G(keras.random.normal((32,LATENT_GAN),seed=7),training=False)),
                4,8,'DCGAN - 32 nowe probki',zakres=(-1,1),slug='gan_nowe_probki')

        # %% Komorka 38
        rt.mark(38)
        def slerp(za,zb,t):
            za,zb = np.asarray(za,dtype='float64'),np.asarray(zb,dtype='float64')
            norma = np.linalg.norm(za)*np.linalg.norm(zb)
            if norma < 1e-12:
                return (1-t)*za + t*zb
            omega = np.arccos(np.clip(np.dot(za,zb)/norma,-1,1))
            if abs(np.sin(omega)) < 1e-7:
                # Slerp dla wektorow rownoleglych/antyrownoleglych jest zle uwarunkowany.
                return (1-t)*za + t*zb
            return np.sin((1-t)*omega)/np.sin(omega)*za + np.sin(t*omega)/np.sin(omega)*zb

        rng = np.random.default_rng(3)
        z_a,z_b = rng.normal(size=LATENT_GAN),rng.normal(size=LATENT_GAN)
        ts = np.linspace(0,1,10)
        z_lin = np.asarray([(1-t)*z_a + t*z_b for t in ts],dtype='float32')
        z_slerp = np.asarray([slerp(z_a,z_b,t) for t in ts],dtype='float32')
        assert np.allclose(z_slerp[0],z_a) and np.allclose(z_slerp[-1],z_b)
        normy_lin, normy_slerp = np.linalg.norm(z_lin,axis=1), np.linalg.norm(z_slerp,axis=1)
        ob_lin = ops.convert_to_numpy(G(z_lin,training=False))
        ob_slerp = ops.convert_to_numpy(G(z_slerp,training=False))
        plansza(np.concatenate([ob_lin,ob_slerp]),2,10,'GAN - interpolacja liniowa (gora) i slerp (dol)',
                zakres=(-1,1),slug='gan_interpolacje')
        fig,ax=plt.subplots(figsize=(7.2,4.1))
        ax.plot(ts,normy_lin,'o-',label='liniowa'); ax.plot(ts,normy_slerp,'s-',label='slerp')
        ax.axhline(np.sqrt(LATENT_GAN),ls='--',label='sqrt(d)')
        ax.set(xlabel='t',ylabel='norma z(t)',title='Norma szumu podczas interpolacji')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); rt.figure(fig,'gan_normy_interpolacji')
        RAPORT['zad_3_3']={'normy_lin':normy_lin,'normy_slerp':normy_slerp,
                          'uwaga':'Losowe konce maja rozne normy, slerp nie musi miec idealnie stalego promienia.'}
        RAPORT['testy']['slerp_konce']=True
        rt.done('3.3')

        # %% Komorka 40
        rt.mark(40)
        keras.utils.set_random_seed(SEED)
        wej=keras.Input(shape=(28,28,1))
        h=layers.Conv2D(32,3,activation='relu',padding='same')(wej)
        h=layers.MaxPooling2D()(h)
        h=layers.Conv2D(64,3,activation='relu',padding='same')(h)
        h=layers.MaxPooling2D()(h)
        h=layers.Flatten()(h)
        cechy=layers.Dense(64,activation='relu',name='cechy')(h)
        wyj=layers.Dense(10,activation='softmax')(layers.Dropout(.3)(cechy))
        klasyfikator=keras.Model(wej,wyj,name='klasyfikator_fashion')
        klasyfikator.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'],jit_compile=False)
        options=tf.data.Options(); options.threading.private_threadpool_size=max(1,WATKI)
        ds_clf=(tf.data.Dataset.from_tensor_slices((x_fash,y_fash)).shuffle(len(x_fash),seed=SEED)
                .batch(128).with_options(options).prefetch(1))
        t0=time.perf_counter()
        hist_clf=klasyfikator.fit(ds_clf,epochs=cfg['clf'],verbose=2,callbacks=[KontrolaEpoki('klasyfikator')])
        rt.history(hist_clf,'klasyfikator')
        clf_eval=klasyfikator.evaluate(x_fash_te,y_fash_te,batch_size=128,return_dict=True,verbose=0)
        RAPORT['klasyfikator']={**{k:float(v) for k,v in clf_eval.items()},'czas_s':round(time.perf_counter()-t0,2)}
        klasyfikator.save(rt.models/'klasyfikator_fashion.keras')
        print('Klasyfikator - test:',RAPORT['klasyfikator'])
        if clf_eval['accuracy'] < .8:
            rt.warn('Klasyfikator ma test accuracy < 0.80. Entropie i FID-lite trzeba interpretowac szczegolnie ostroznie.')
        rt.save()

        # %% Komorka 41
        rt.mark(41)
        def entropia_znorm(p):
            p=np.asarray(p,dtype='float64')
            p=p[p>0]
            return float(-(p*np.log(p)).sum()/np.log(10))

        z_gen=np.random.default_rng(SEED+100).normal(size=(N_GEN,LATENT_GAN)).astype('float32')
        x_gen=G.predict(z_gen,batch_size=128,verbose=0)
        p_gen=klasyfikator.predict(x_gen,batch_size=128,verbose=0)
        p_real=klasyfikator.predict(x_fash_te,batch_size=128,verbose=0)
        # Histogram to twarde klasy argmax, nie srednia softmaxow.
        hist_gen=np.bincount(np.argmax(p_gen,axis=1),minlength=10)
        hist_real=np.bincount(np.argmax(p_real,axis=1),minlength=10)
        entropia_gen=entropia_znorm(hist_gen/hist_gen.sum())
        entropia_real=entropia_znorm(hist_real/hist_real.sum())
        pewnosc_gen,pewnosc_real=float(p_gen.max(1).mean()),float(p_real.max(1).mean())
        assert np.isfinite(p_gen).all() and np.allclose(p_gen.sum(1),1,atol=1e-5)
        RAPORT['zad_3_4']={'n_gen':N_GEN,'n_real':len(x_fash_te),'entropia_gen':entropia_gen,
                          'entropia_real':entropia_real,'pewnosc_gen':pewnosc_gen,'pewnosc_real':pewnosc_real,
                          'histogram_gen':hist_gen,'histogram_real_predykcje':hist_real,
                          'uwaga':'Brak automatycznego werdyktu mode collapse. Sprawdz obrazy i jakosc klasyfikatora.'}
        rt.table(pd.DataFrame({'klasa':KLASY_FASHION,'liczba_gen':hist_gen,'udzial_gen':hist_gen/N_GEN,
                              'liczba_real_pred':hist_real,'udzial_real_pred':hist_real/len(x_fash_te)}),'roznorodnosc_klas.csv')
        fig,ax=plt.subplots(figsize=(9,4.6)); pos=np.arange(10)
        ax.bar(pos-.2,hist_gen/N_GEN,width=.4,label='GAN')
        ax.bar(pos+.2,hist_real/len(x_fash_te),width=.4,label='real - klasy przewidziane')
        ax.set_xticks(pos,labels=KLASY_FASHION,rotation=35,ha='right')
        ax.set(ylabel='udzial',title='Roznorodnosc klas: GAN vs prawdziwe obrazy')
        ax.legend(); fig.tight_layout(); rt.figure(fig,'gan_histogram_klas')
        print('Entropia GAN/real:',entropia_gen,entropia_real,'| pewnosc:',pewnosc_gen,pewnosc_real)
        if hist_gen.max()/N_GEN > .5:
            rt.warn('Ponad polowa probek GAN ma te sama przewidywana klase: sprawdz mode collapse i bledy klasyfikatora.')
        rt.done('3.4')

        # %% Komorka 43
        rt.mark(43)
        from sklearn.neighbors import NearestNeighbors
        ekstr_nn=keras.Model(klasyfikator.input,klasyfikator.get_layer('cechy').output)
        f_tr=ekstr_nn.predict(x_fash,batch_size=128,verbose=0)
        x_pr=ops.convert_to_numpy(G(keras.random.normal((8,LATENT_GAN),seed=21),training=False))
        odl,idx=NearestNeighbors(n_neighbors=1).fit(f_tr).kneighbors(ekstr_nn.predict(x_pr,verbose=0))
        plansza(np.concatenate([x_pr,x_fash[idx[:,0]]]),2,8,
                'GAN (gora) / najblizszy treningowy obraz w cechach klasyfikatora (dol)',
                zakres=(-1,1),slug='gan_najblizsi_sasiedzi')
        RAPORT['sasiedzi']={'odleglosci':odl[:,0],'indeksy_train':idx[:,0],
                           'uwaga':'8 par nie dowodzi braku zapamietywania. Odleglosc w cechach != identycznosc pikseli.'}
        print('Odleglosci NN:',odl[:,0]); rt.save()

        # %% Komorka 46
        rt.mark(46)
        if PYTORCH:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            torch.manual_seed(SEED)
            torch.set_num_threads(WATKI)
            device=torch.device('cpu')
            # Dane te same co w Keras; jedynie NHWC -> NCHW.
            ds_t=TensorDataset(torch.from_numpy(np.transpose(x_fash,(0,3,1,2)).copy()),
                               torch.from_numpy(y_fash.astype('int64')))
            loader=DataLoader(ds_t,batch_size=128,shuffle=True,drop_last=True,num_workers=0,
                              generator=torch.Generator().manual_seed(SEED))
            LAT_T=64
            class GenT(nn.Module):
                def __init__(self,latent=64):
                    super().__init__()
                    self.fc=nn.Sequential(nn.Linear(latent,128*7*7,bias=False),nn.BatchNorm1d(128*7*7),nn.LeakyReLU(.2))
                    self.conv=nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias=False),nn.BatchNorm2d(64),
                                            nn.LeakyReLU(.2),nn.ConvTranspose2d(64,1,4,2,1),nn.Tanh())
                def forward(self,z):
                    return self.conv(self.fc(z).view(-1,128,7,7))
            class DisT(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net=nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(64,128,4,2,1),
                                           nn.LeakyReLU(.2),nn.Flatten(),nn.Dropout(.3),nn.Linear(128*7*7,1))
                def forward(self,x):
                    return self.net(x)
            G_t,D_t=GenT().to(device),DisT().to(device)
            opt_G=torch.optim.Adam(G_t.parameters(),lr=2e-4,betas=(.5,.999))
            opt_D=torch.optim.Adam(D_t.parameters(),lr=2e-4,betas=(.5,.999))
            kryterium=nn.BCEWithLogitsLoss()
            print('PyTorch:',torch.__version__,'| CPU | latent=',LAT_T)
        else:
            print('Sekcja 4 opcjonalna: pomijam. Do wlaczenia sluzy --pytorch.')

        # %% Komorka 48
        rt.mark(48)
        if PYTORCH:
            straty_D,straty_G=[],[]
            torch_epochs=[]
            z_staly=torch.randn(16,LAT_T,device=device)
            t0=time.perf_counter()
            for epoka in range(EPOKI_TORCH):
                G_t.train(); D_t.train(); te=time.perf_counter()
                ld_epoch,lg_epoch=[],[]
                for real,_ in loader:
                    real=real.to(device); b=real.size(0)
                    jedynki,zera=torch.ones(b,1,device=device),torch.zeros(b,1,device=device)
                    z=torch.randn(b,LAT_T,device=device)
                    fake=G_t(z)
                    opt_D.zero_grad(set_to_none=True)
                    loss_D=kryterium(D_t(real),.9*jedynki)+kryterium(D_t(fake.detach()),zera)
                    if not bool(torch.isfinite(loss_D)):
                        raise FloatingPointError('PyTorch D: NaN/inf')
                    loss_D.backward(); opt_D.step()
                    # Nie zmieniamy wag D krokiem G; graf przez D nadal prowadzi do fake.
                    for p in D_t.parameters(): p.requires_grad_(False)
                    opt_G.zero_grad(set_to_none=True)
                    loss_G=kryterium(D_t(fake),jedynki)
                    if not bool(torch.isfinite(loss_G)):
                        raise FloatingPointError('PyTorch G: NaN/inf')
                    loss_G.backward(); opt_G.step()
                    for p in D_t.parameters(): p.requires_grad_(True)
                    ld,lg=float(loss_D.detach()),float(loss_G.detach())
                    straty_D.append(ld); straty_G.append(lg); ld_epoch.append(ld); lg_epoch.append(lg)
                torch_epochs.append({'epoka':epoka+1,'d_loss':float(np.mean(ld_epoch)),
                                     'g_loss':float(np.mean(lg_epoch)), 'czas_s':time.perf_counter()-te})
                print('PyTorch epoka:',torch_epochs[-1],flush=True)
                torch.save({'G':G_t.state_dict(),'D':D_t.state_dict(),'opt_G':opt_G.state_dict(),
                            'opt_D':opt_D.state_dict(),'epoka':epoka+1},rt.models/'gan_torch_ostatni.pt')
            RAPORT['zad_4_1']={'czas_s':round(time.perf_counter()-t0,2),'epoki':EPOKI_TORCH,
                              'latent':LAT_T,'batch_size':128,'ostatnie':torch_epochs[-1],
                              'uwaga':'Nie jest to benchmark bibliotek 1:1: inne latent/budzet/inicjalizacja.'}
            RAPORT['opcjonalne']['pytorch']='wykonano'
            rt.table(pd.DataFrame(torch_epochs),'gan_pytorch_epoki.csv'); rt.done('4.1')

        # %% Komorka 49
        rt.mark(49)
        if PYTORCH:
            G_t.eval()
            with torch.no_grad():
                probki_t=G_t(z_staly).cpu().numpy().transpose(0,2,3,1)
            fig,ax=plt.subplots(figsize=(8,4))
            ax.plot(straty_D,label='D',alpha=.7);ax.plot(straty_G,label='G',alpha=.7)
            ax.set(xlabel='krok (batch)',ylabel='BCE',title='PyTorch GAN - straty');ax.legend()
            fig.tight_layout();rt.figure(fig,'pytorch_gan_straty')
            plansza(probki_t,2,8,'PyTorch DCGAN - ustalony szum',zakres=(-1,1),slug='pytorch_gan_probki')

        # %% Komorka 53
        rt.mark(53)
        RAPORT['biblioteki']['backend_keras']=keras.backend.backend()
        print(json.dumps(RAPORT['biblioteki'],indent=2,ensure_ascii=False))
        print('Seed:',SEED,'| profil:',PROFIL,'| watki CPU:',WATKI)
        rt.save()

        # %% Komorka 54
        rt.mark(54)
        def raport_powtorzen(wyniki,nazwa='metryka'):
            w=np.asarray(wyniki,dtype=float)
            if not len(w) or not np.isfinite(w).all():
                raise ValueError('Brak poprawnych wynikow do podsumowania')
            if len(w)==1:
                return f'{nazwa} = {w[0]:.4f} (n=1; odchylenie standardowe nieoszacowane)'
            return f'{nazwa} = {w.mean():.4f} +/- {w.std(ddof=1):.4f} (n={len(w)})'
        print('W tym przebiegu jest jedno ziarno. Nie wpisujemy fikcyjnych powtorzen do raportu.')

        # %% Komorka 59
        rt.mark(59)
        if BONUS:
            from scipy.linalg import sqrtm
            def frechet(f1,f2):
                f1,f2=np.asarray(f1,dtype='float64'),np.asarray(f2,dtype='float64')
                if f1.ndim!=2 or f2.ndim!=2 or f1.shape[1]!=f2.shape[1] or min(len(f1),len(f2))<2:
                    raise ValueError('FD wymaga dwoch macierzy cech z co najmniej 2 obrazami')
                if not np.isfinite(f1).all() or not np.isfinite(f2).all():
                    raise ValueError('FD: NaN/inf w cechach')
                mu1,mu2=f1.mean(0),f2.mean(0)
                c1,c2=np.cov(f1,rowvar=False),np.cov(f2,rowvar=False)
                root=sqrtm(c1@c2)
                # sqrtm moze dac drobny blad urojony dla numerycznie osobliwej kowariancji.
                if np.iscomplexobj(root):
                    rel_im=float(np.linalg.norm(root.imag)/max(1.,np.linalg.norm(root.real)))
                    if rel_im>1e-5:
                        raise FloatingPointError(f'FD: za duza czesc urojona sqrtm ({rel_im:g})')
                    root=root.real
                raw=float(np.sum((mu1-mu2)**2)+np.trace(c1+c2-2*root))
                scale=max(1.,float(np.trace(c1+c2)))
                if not np.isfinite(raw) or raw < -1e-6*scale:
                    raise FloatingPointError(f'FD: wynik {raw}')
                return max(0.,raw)
            probe=np.random.default_rng(21).normal(size=(100,6))
            assert frechet(probe,probe)<1e-6
            RAPORT['testy']['frechet_tozsamosc']=True
            ekstraktor=keras.Model(klasyfikator.input,klasyfikator.get_layer('cechy').output)
            # Ta sama liczba obrazow w kazdym porownaniu, real A i B sa rozlaczne.
            n_fd=min(N_GEN,len(x_fash_te)//2)
            idx_fd=np.random.default_rng(SEED+200).permutation(len(x_fash_te))
            f_real_a=ekstraktor.predict(x_fash_te[idx_fd[:n_fd]],batch_size=128,verbose=0)
            f_real_b=ekstraktor.predict(x_fash_te[idx_fd[n_fd:2*n_fd]],batch_size=128,verbose=0)
            f_gen=ekstraktor.predict(x_gen[:n_fd],batch_size=128,verbose=0)
            keras.utils.set_random_seed(SEED+201)
            G_losowy=zbuduj_generator()
            x_random=G_losowy.predict(z_gen[:n_fd],batch_size=128,verbose=0)
            f_random=ekstraktor.predict(x_random,batch_size=128,verbose=0)
            fd={'real_A_vs_real_B':frechet(f_real_a,f_real_b),
                'GAN_Keras':frechet(f_real_a,f_gen),'G_nietrenowany':frechet(f_real_a,f_random)}
            if PYTORCH:
                G_t.eval()
                out=[]; rng_t=torch.Generator().manual_seed(SEED+202)
                with torch.no_grad():
                    for start in range(0,n_fd,128):
                        zt=torch.randn(min(128,n_fd-start),LAT_T,generator=rng_t)
                        out.append(G_t(zt.to(device)).cpu().numpy().transpose(0,2,3,1))
                f_t=ekstraktor.predict(np.concatenate(out),batch_size=128,verbose=0)
                fd['GAN_PyTorch']=frechet(f_real_a,f_t)
            RAPORT['bonus_fid_lite']={'odleglosci':fd,'liczba_obrazow_na_zbior':n_fd,
                                     'cechy':64,'uwaga':'FID-lite, nie oryginalny Inception FID. Brak porownywalnosci z literatura.'}
            RAPORT['opcjonalne']['fid_lite']='wykonano'
            rt.table(pd.DataFrame({'porownanie':list(fd),'FD':list(fd.values()),'n_na_zbior':n_fd}),'fid_lite.csv')
            if not fd['real_A_vs_real_B'] < fd['GAN_Keras'] < fd['G_nietrenowany']:
                rt.warn('Ranking FID-lite nie spelnia oczekiwanego real-real < GAN < losowy. Zachowujemy wynik; sprawdz obrazy.')
            fig,ax=plt.subplots(figsize=(8,4.1))
            ax.bar(list(fd),list(fd.values()))
            ax.set(ylabel='FD w 64 cechach',title='FID-lite - mniejsza odleglosc jest lepsza')
            ax.tick_params(axis='x',rotation=15)
            fig.tight_layout();rt.figure(fig,'fid_lite')
            rt.done('bonus_fid_lite')
        else:
            print('FID-lite pominiete. Do wlaczenia: --bonus.')

        # %% Komorka 60
        rt.mark(60)
        wnioski=[
            'LAB07 - wyniki z lokalnego przebiegu', f'Profil: {PROFIL}; seed: {SEED}',
            f"VAE: test rec_loss={RAPORT['zad_2_2']['ocena_test']['rec_loss']:.3f}, KL={RAPORT['zad_2_2']['ocena_test']['kl_loss']:.3f}.",
            f'Eksperyment beta: {werdykt_beta}; pelna tabela w porownanie_beta.csv.',
            f'GAN: {EPOKI_GAN} epok; straty nie sa samodzielna miara jakosci obrazow.',
            f'Klasyfikator Fashion test accuracy={clf_eval["accuracy"]:.4f}.',
            f'Entropia klas GAN={entropia_gen:.4f}, real={entropia_real:.4f}.',
            f'Srednia pewnosc GAN={pewnosc_gen:.4f}, real={pewnosc_real:.4f}.',
            'Mode collapse: wymaga oceny plansz wraz z histogramem; brak automatycznego potwierdzenia/wykluczenia.',
            'Najblizsi sasiedzi: podobienstwo 8 par nie rozstrzyga, czy model zapamietal trening.',
            'VAE walidowano na 10% puli treningowej; oficjalny test wykorzystano do oceny i ilustracji.',
            'Usprawnienie do przyszlego mini-projektu: zmniejszenie lr D przy tym samym budzecie GAN, min. 3 seedy; NIE wykonano tutaj.',
        ]
        if BONUS:
            wnioski.append('FID-lite: '+str(RAPORT['bonus_fid_lite']['odleglosci']))
        wnioski.append('Ostrzezenia: '+('; '.join(RAPORT['ostrzezenia']) or 'brak technicznych; obrazy nadal wymagaja oceny'))
        (rt.root/'wnioski.txt').write_text('\n'.join(wnioski)+'\n',encoding='utf-8')
        RAPORT['status']='UKONCZONO_TEST' if PROFIL!='full' else 'UKONCZONO_DO_OCENY_WYNIKOW'
        RAPORT['bonus_wykonany']=BONUS
        RAPORT['pytorch_wykonany']=PYTORCH
        RAPORT['liczba_wykresow']=len(RAPORT['wykresy'])
        rt.done('podsumowanie')
        print('\nKONIEC:',RAPORT['status'])
        print('Wyniki:',rt.root)
        print('Wykresy:',RAPORT['liczba_wykresow'])
        if TRYB_SKRYPTU and rt.template is not None:
            print('Notebook z wynikami:',rt.root/'Lab07_z_wynikami.ipynb')
        if not TRYB_SKRYPTU:
            rt.close()
    except BaseException as exc:
        if rt is not None:
            rt.close(error=exc)
        raise
    else:
        if rt is not None:
            rt.close()


if __name__ == '__main__':
    # Agg i backend ustawiamy PRZED importem TF/Keras/pyplot.
    os.environ['KERAS_BACKEND']='tensorflow'
    os.environ.setdefault('MPLBACKEND','Agg')
    try:
        if hasattr(sys.stdout,'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8',errors='replace')
        wykonaj(pobierz_argumenty())
    except KeyboardInterrupt:
        print('Przerwano. Zachowano zapisane wyniki czastkowe.',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
