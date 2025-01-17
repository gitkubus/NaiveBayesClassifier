# NaiveBayesClassifier
## Autorzy
- Jakub Janus
- Jakub Kaliński

Projekt został zrealizowany w ramach kursu Rachunek Prawdopodobieństwa i Statystyka

## Opis projektu
Repozytorium zawiera implementację Naiwnego Klasyfikatora Bayesowskiego dla zbiorów danych o cechach kategorycznych lub ilościowych.

## MultinomialNaiveBayesClassifier
Klasa zaimplementowana w pliku `MultinomialNaiveBayesClassifier.py`, implementuje ona Naiwny Klasyfikator Bayesowski dla danych kategorycznych
Metody klasy:
- fit - do trenowania modelu na zbiorze treningowym.
- predict - do przewidywania klasy dla nowych danych.
- predict_proba - do zwracania prawdopodobieństw przynależności do każdej klasy

## GaussianNaiveBayesClassifier
Klasa zaimplementowana w pliku `MultinomialNaiveBayesClassifier.py`, implementuje ona Naiwny Klasyfikator Bayesowski dla danych ilościowych
Metody klasy:
- fit - do trenowania modelu na zbiorze treningowym.
- predict - do przewidywania klasy dla nowych danych.
- predict_proba - do zwracania prawdopodobieństw przynależności do każdej klasy

## main.ipynb
W pliku `main.ipynb` znajduje się wstępna analiza danych oraz sprawdzenie poprawności działania implementacji obu klasyfikatorów na zbiorach `iris` z biblioteki scikit-learn oraz `mushroom` z platformy Kaggle (mushrooms.csv). W pliku `main.ipynb` znajdują się również procentowe wyniki klasyfiacji modeli oraz confusion matrix tych wyników.
