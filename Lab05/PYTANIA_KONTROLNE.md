# Lab05 - odpowiedzi do pytań kontrolnych

1. **BCEWithLogitsLoss:** przyjmuje logity, oblicza sigmoidę i BCE stabilnie numerycznie; ostatnia warstwa MLP zwraca logit, nie prawdopodobieństwo.
2. **state_dict a deepcopy:** zwykłe przypisanie słownika wag nie tworzy niezależnego snapshotu; po kolejnych epokach "najlepszy model" byłby tym ostatnim. Stąd `copy.deepcopy`.
3. **ROC-AUC vs PR-AUC:** ROC-AUC mierzy ranking globalnie, PR-AUC mocniej zależy od prewalencji i skupia się na klasie pozytywnej. Dla rzadkich zdarzeń PR-AUC często lepiej pokazuje jakość wykrywania.
4. **Wybór progu:** na walidacji, uwzględniając koszt FN/FP oraz wymaganie recall. Test wykorzystujemy dopiero do końcowej ewaluacji wybranego progu.
5. **Kalibracja vs dyskryminacja:** kalibracja dotyczy zgodności prawdopodobieństw z częstościami zdarzenia; ROC-AUC mierzy ranking. Monotoniczny Platt może poprawić Brier bez zmiany AUC.
6. **Data leakage:** np. selekcja cech na całym zbiorze przed CV, oversampling przed CV, dopasowanie skalera przed podziałem czy ten sam pacjent w train i test. Wszystkie operacje `fit` trzeba wykonywać wewnątrz folda.
7. **KFold vs GroupKFold:** KFold dzieli wiersze; GroupKFold trzyma całą grupę (np. pacjenta) po jednej stronie podziału.
8. **Nadeau–Bengio:** foldy współdzielą dane treningowe, więc ich wyniki nie są niezależne. Korekta poszerza CI względem naiwnie liczonych przedziałów. To przybliżenie, zwłaszcza dla AUC blisko 1.
9. **Permutation importance:** mierzy spadek metryki po permutacji jednej cechy. Gdy cechy są skorelowane, inne mogą zastępować permutowaną, więc ważność pojedynczej cechy bywa zaniżona.
10. **Random search / test:** hiperparametry wybieramy na CV części dev; test jest tylko do jednorazowej oceny gotowej procedury.
11. **Zagnieżdżona CV:** pętla wewnętrzna stroi, zewnętrzna ocenia całe strojenie; ogranicza optymizm wyniku wybranej najlepszej konfiguracji.

Wszystkie wartości liczbowe do sprawozdania pobierz z `wyniki_full/podsumowanie.json`, nie z trybu FAST.
