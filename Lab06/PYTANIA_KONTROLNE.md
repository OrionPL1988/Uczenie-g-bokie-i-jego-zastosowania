# Lab06 - odpowiedzi do pytań kontrolnych

1. **LDA i limit C-1**: LDA maksymalizuje separację klas na podstawie etykiet.
   Macierz rozrzutu międzyklasowego ma rząd najwyżej `C-1`; PCA maksymalizuje
   wariancję danych bez etykiet. Dla 10 cyfr LDA daje najwyżej 9 osi.
2. **Explained variance ratio**: w PCA udział całkowitej wariancji danych
   wyjaśniany przez kolejną składową. W LDA jest to udział w zachowanej
   separacji klas (wariancji dyskryminacyjnej), nie w ogólnej wariancji obrazów.
3. **TruncatedSVD zamiast PCA**: przy macierzach rzadkich (np. TF-IDF),
   ponieważ nie centruje danych i nie niszczy rzadkości.
4. **Liniowy AE a PCA**: przy MSE, liniowych warstwach, odpowiednim wąskim gardle
   i osiągnięciu optimum ich podprzestrzenie się pokrywają. Nie muszą być równe
   poszczególne wektory/wagi, bo podprzestrzeń ma wiele baz.
5. **DAE**: odtwarza czysty sygnał z zaszumionego wejścia, więc nawet bez
   małego latentu uczy się struktury i zależności pomiędzy pikselami.
6. **Silhouette a ARI/NMI**: silhouette mierzy zwartosc/rozdzielenie grup
   w konkretnej metryce i przestrzeni; zmianie wymiaru towarzyszy zmiana
   geometrii/odległości, więc wyników nie zestawiamy wprost. ARI/NMI porównują
   podział z prawdziwymi etykietami i są niewrażliwe na numery skupień.
7. **Conv2DTranspose (k=4,s=2,valid)**: `(7-1)*2+4=16`, czyli
   wyjście ma kształt `16 x 16 x 16` (dla pojedynczego obrazu).
8. **Próg anomalii bez wycieku**: np. 95. percentyl błędu na osobnym
   walidacyjnym zbiorze próbek normalnych; nie dobieramy progu do testu.
   Metoda może zawieść, gdy anomalia jest podobna do normalnych (np. 9 do 4/7)
   lub gdy AE zbyt dobrze rekonstruuje także nieznane klasy.

Przykład usprawnienia do omówienia po FULL: dla detekcji anomalii można
wytrenować CAE *wyłącznie* na obrazach normalnych i porównać ROC-AUC z AE
gęstym przy identycznym podziale, progu walidacyjnym i metryce MSE.
