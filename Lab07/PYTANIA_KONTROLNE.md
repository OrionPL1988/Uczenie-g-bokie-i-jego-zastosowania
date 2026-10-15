# Lab07 - pytania kontrolne

Odpowiedzi do pytan ze zrodlowego notebooka; skroty do nauki, nie gotowe oswiadczenie o autorstwie.

## 1. Dwa skladniki VAE
Rekonstrukcja (suma BCE po pikselach) wymusza zachowanie informacji o obrazie, a KL ogranicza odchylenie q(z|x) od N(0,I). Bez KL latent nie musi nadawac sie do losowania z priora. Bez rekonstrukcji model moze ignorowac obraz i minimalizowac sam KL. Nie oceniamy skali strat po ich nazwach: BCE to tutaj suma 784 skladnikow na obraz.

## 2. Reparametryzacja
z = mu + exp(0.5 * log_var) * eps, eps ~ N(0,I). Losujemy eps, natomiast mu i sigma pozostaja w rozniczkowalnym wyrazeniu. Gradient moze wiec trenowac obie glowy enkodera. W Keras uzywamy SeedGenerator, a nie tego samego niezaleznego seedu jako stalej przy kazdym wywolaniu.

## 3. Po co log wariancji?
Glowica moze zwracac dowolna liczbe rzeczywista. Po exp otrzymujemy dodatnia wariancje, a po exp(0.5 * log_var) odchylenie standardowe. Pomylenie tych dwoch wielkosci zmienia rozrzut probek. Ekspozycja bardzo duzych log_var nadal moze prowadzic do przepelnienia - dlatego sa testy skonczonosci.

## 4. Rozmycie VAE vs GAN
W ukladzie z notebooka VAE placi jednoczesnie za rekonstrukcje pikseli i regularyzacje ograniczonego latentu; niejednoznaczne detale moga zostac usrednione. GAN korzysta z sygnalu dyskryminatora i moze tworzyc ostrzejsze wzory, ale moze tez tracic roznorodnosc. Nie jest to gwarancja, ze kazdy GAN wygra z kazdym VAE.

## 5. Cel GAN i non-saturating loss
Min_G max_D: E_real[log D(x)] + E_z[log(1-D(G(z)))]. W implementacji D zwraca logity. D minimalizuje sume BCE na prawdziwych (etykieta 0.9) i falszywych obrazach (0). G minimalizuje BCE D(G(z)) z etykieta 1, czyli odpowiednik -log D(G(z)); nie minimalizuje tej samej straty co D. Non-saturating loss pomaga, gdy D na poczatku latwo rozpoznaje probki generatora.

## 6. detach i osobne listy wag
fake.detach() odcina graf G podczas liczenia straty D. Potem loss_G uzywa nieodcietego fake i nowego przebiegu D. W Keras gradient i krok optimizera wyliczamy dla wag tylko odpowiedniej sieci, a w kroku D dodatkowo odcinamy fake. BatchNorm generatora moze aktualizowac statystyki przy training=True: to nie to samo co aktualizacja jego wag gradientem D.

## 7. Mode collapse
Generator moze odtwarzac tylko kilka typow probek mimo roznych z. Porownujemy plansze stalego szumu, histogram klas, znormalizowana entropie i pewnosc klasyfikatora z prawdziwymi danymi. Wysoka entropia klas nie wyklucza kopiowania jednego wzorca na klase ani braku realizmu. Slaby klasyfikator moze zafalszowac te wskazniki.

## 8. Dlaczego straty GAN nie wystarczaja?
G i D zmieniaja sobie zadanie w trakcie treningu, wiec loss nie jest pojedyncza skala realizmu. Ogladamy probki, ich ewolucje i roznorodnosc, wspierajac ocene metrykami cech. FID-lite z tego laba korzysta z 64 cech klasyfikatora Fashion-MNIST; nie jest Inception FID i jego liczb nie porownujemy z wynikami z literatury. Na malych probkach nawet real-real nie musi dac zera.

## Dyskusja: prywatnosc i dyfuzja
Test najblizszego sasiada jest lokalna kontrola podobienstwa, nie dowodem braku zapamietywania calego treningu. Przy publikowaniu modelu trzeba uwzglednic pochodzenie danych, zgody i mozliwe naduzycia. W szkicu dyfuzji z notebooka proces w przod zaszumia obraz, a siec uczy sie odszumiania kolejnych poziomow; jest to inny cel uczenia niz gra generatora i dyskryminatora.

## Materialy techniczne wykorzystane przy opracowaniu
- Zalaczony notebook Lab07_VAE_GAN_modele_generatywne.ipynb (polecenia i zakres).
- Keras, Variational AutoEncoder: https://keras.io/examples/generative/vae/
- Keras, DCGAN overriding train_step: https://keras.io/examples/generative/dcgan_overriding_train_step/
- SciPy, sqrtm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html

W finalnej bibliografii uwzglednij rowniez literature przedmiotu wskazana przez prowadzaca.
