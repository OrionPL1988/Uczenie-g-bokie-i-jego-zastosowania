# Lab02 - Odpowiedzi na pytania kontrolne

**1. Reguła perceptronu.** Dla błędu `y - y_hat` aktualizuję `w <- w + eta*(y-y_hat)*x` oraz `b <- b + eta*(y-y_hat)`. To nie jest klasyczny gradient descent: funkcja progowa nie dostarcza użytecznej pochodnej. Perceptron gwarantuje zbieżność, gdy zbiór jest liniowo separowalny z dodatnim marginesem.

**2. Dlaczego XOR?** Dwie klasy XOR zajmują przeciwległe wierzchołki kwadratu. Żadna prosta nie rozdzieli tych klas. MLP tworzy nieliniowe przekształcenie w warstwie ukrytej i może opisać granicę, której prosta nie potrafi.

**3. Po co aktywacje nieliniowe?** Złożenie warstw afinicznych jest pojedynczym przekształceniem afinicznym: `W2(W1 x + b1) + b2 = (W2W1)x + (W2b1 + b2)`. Wiele warstw bez nieliniowości nie rozwiązuje problemu nieliniowej separacji.

**4. Znikający gradient.** Reguła łańcuchowa mnoży pochodne z wielu warstw. Sigmoid ma pochodną nie większą niż 0.25, a przy nasyceniu bliską zera. ReLU ma dla dodatnich argumentów pochodną 1, choć może mieć martwe neurony dla argumentów niedodatnich.

**5. Parametry sieci 10 -> 32 -> 16 -> 3.** `(10+1)*32 = 352`, `(32+1)*16 = 528`, `(16+1)*3 = 51`; razem **931 parametrów** (z biasami).

**6. Sigmoid + BCE.** Reguła łańcuchowa upraszcza się: pochodna straty BCE po logicie to `y_hat - y` (dla średniej po batchu: podzielone przez `n`). `BCEWithLogitsLoss` oblicza operację stabilniej numerycznie niż osobne `sigmoid`, `log` i BCE, szczególnie przy ekstremalnych logitach.

**7. Sequential vs Functional.** `Sequential` to liniowy ciąg warstw. `Functional API` pozwala budować graf z wieloma wejściami, wyjściami i odgałęzieniami. Przykład: połączenie skip/concatenate pomiędzy wejściem a wyjściem warstwy ukrytej.

**8. Pętla PyTorch.** `model.train()` -> batch z `DataLoader` -> `opt.zero_grad()` -> forward -> loss -> `loss.backward()` -> `opt.step()`. Do walidacji `model.eval()` i `with torch.no_grad()` oraz obliczanie metryk bez zmiany wag. `train` i `eval` są szczególnie ważne przy Dropout i BatchNorm.
