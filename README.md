# Uczenie Maszyn - projekt
Predykcja opóźnienia lotu z wykorzystaniem LLM i klasycznych metod uczenia maszynowego.

## Opis problemu
W tym projekcie zajmiemy się badaniem skuteczności odtwarzania informacji poprzez LLM. Chcemy sprawdzic jak będą zachowywac się wyniki i jakie ewentualne uprzedzenia modelu znajdują się w jego wenwętrznej pamięci. 

## Proponowana metoda
### Generowanie opisów
Z pomocą modelów opensource `llama3.2` oraz `gemma2` predykujemy opóźnienie lotu w postaci wartości oraz opisu (na potrzeby dalszego przetwarzania). 

### Predykcja
Następnie na podstawie opisów próbowac będziemy wyznaczyc opóźnienie.

### Predykcja za pomocą innego algorytmu
Równolegle dokonamy predykcji opóźnienia za pomocą innego algorytmu, np. Bayesa

### Porównanie
Porównamy te trzy metody i zobaczymy, która najlepiej sobie radzi.

## Zestaw danych
Pracowac będziemy na danych lotów krajowych w USA z 2018 roku.

[Link do folderu z przygotowanymi zestawami danych](./llm_prepared_datasets)

## Plan badań
    1. Losowe wybranie 3000 rekordów do badań
    2. Wygenerowanie opisów przez LLM
    3. Predykcja na podstawie opisów
    4. Predykcja algorytmem Bayesa
    5. Porównanie i opracowanie wyników

## Schemat
![Schemat projektu](./images/schemat_projektu.jpeg)


## Tokenizacja
Do tokenizacji używamy modelu `TFIDF`.

### Proces tokenizacji
1. Wczytanie danych z pliku CSV do DataFrame.
2. Usunięcie wierszy z brakującymi wartościami.
3. Inicjalizacja wektoryzatora TF-IDF.
4. Dopasowanie i przekształcenie danych tekstowych za pomocą TF-IDF.
5. Utworzenie DataFrame z wartościami TF-IDF dla pierwszego dokumentu.
6. Sortowanie wartości TF-IDF w porządku malejącym.
7. Generowanie tokenów dla każdego dokumentu:
    - Dla każdego dokumentu tworzony jest słownik tokenów z ich wartościami TF-IDF.
8. Zapisanie tokenów do pliku JSON.



