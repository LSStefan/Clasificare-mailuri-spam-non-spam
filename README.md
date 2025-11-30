# Rationamentul din spatele pasilor implementati

Proiectul de clasificare a mail-urilor in categorii (spam/non-spam) a urmat o serie de pasi standard de Machine Learning (ML) si Procesare a Limbajului Natural (NLP). Fiecare pas a fost executat cu un scop precis pentru a transforma datele brute intr-un format optim pentru antrenarea modelului.

---

## 1. Procesarea datelor

### 1.1 Incarcarea datelor si curatarea initiala

* **Eliminarea Coloanelor Inutile (Ex: `Unnamed: 2`, `Unnamed: 3`):**
    * **Rationament:** Aceste coloane apar in urma importarii datelor si sunt pline de valori lipsa (`NaN`), neaducand nicio informatie utila pentru clasificare.
    * **Scop:** Reducerea dimensionalitatii, eliminarea zgomotului si simplificarea structurii DataFrame-ului.

* **Eliminarea duplicatelor:**
    * **Rationament:** Prezenta mesajelor identice in setul de date ar supraestima performanta modelului in timpul evaluarii si ar introduce o polarizare nejustificata in timpul antrenarii.
    * **Scop:** Asigurarea faptului ca fiecare mesaj este unic si obtinerea unei baze de antrenare si testare corecte.

### 1.2 Encodarea variabilei tinta

* **Aplicarea `LabelEncoder` pe `v1` (Eticheta):**
    * **Rationament:** Modelele de ML lucreaza exclusiv cu date numerice. Etichetele textuale (`ham`/`spam`) trebuie convertite.
    * **Scop:** Transformarea etichetelor in valori numerice binare (**0** si **1**), formatul necesar pentru clasificarea binara.

---

## 2. Curatarea si vectorizarea textului (NLP)

### 2.1 Functia de curatare a textului

* **Normalizare (Lowercase), Eliminarea punctuației, Numerelor si Stop Words:**
    * **Rationament:** Aceste operatiuni standard reduc varianta inutila a datelor (Ex: `Buy` vs `buy`), elimina zgomotul sintactic (punctuatia) si se concentreaza pe cuvintele purtatoare de sens.
    * **Scop:** Optimizarea vocabularului pentru a reduce dimensiunea matricei de caracteristici.

* **Lematizarea (`WordNetLemmatizer`):**
    * **Rationament:** Reducerea cuvintelor la forma lor de baza, asigurand ca forme diferite ale aceluiasi cuvant (Ex: `running`, `runs` $\rightarrow$ `run`) sunt tratate ca un singur *token*.
    * **Scop:** Agregarea variantelor lexicale si imbunatatirea consistentei datelor.

### 2.2 Vectorizarea TF-IDF

* **De ce Vectorizare?**
    * **Rationament:** Textul curatat, fiind un set de *token*-uri, trebuie convertit intr-o **matrice de numere** pentru a fi inteles de algoritmii ML.
    * **Scop:** Conversia textului intr-un format numeric procesabil.

* **De ce TF-IDF (Term Frequency-Inverse Document Frequency)?**
    * **Rationament:** TF-IDF masoara importanta relativa a fiecarui cuvant. Cuvintele comune (putere de discriminare scazuta) primesc scoruri mici, iar cuvintele rare si specifice (putere de discriminare mare, Ex: "urgent", "castiga") primesc scoruri mari.
    * **Scop:** Ponderarea caracteristicilor pentru a ajuta modelul sa identifice cuvintele cel mai relevante pentru clasa **Spam**.

---

## 3. Antrenare si evaluare

### 3.1 Impartirea datelor (Train/Test Split)

* **Rationament:** Impartirea datelor (80% antrenare, 20% testare) este vitala pentru a preveni **supra-antrenarea (overfitting)**. Prin utilizarea parametrului `stratify=y`, ne asiguram ca seturile de antrenare si testare mentin aceeasi proportie de `ham`/`spam`.
* **Scop:** Evaluarea capacitatii reale a modelului de a generaliza pe date noi, nevazute.

### 3.2 Antrenarea cu Grid Search

* **De ce mai multe modele (MNB, LR, SVC)?**
    * **Rationament:** Nu exista un algoritm universal cel mai bun. MNB este eficient pentru clasificarea textului, LR ofera o interpretare buna si SVC este robust in spatii de inalta dimensionalitate.
    * **Scop:** Identificarea algoritmului care ofera cea mai buna performanta pe setul nostru de date TF-IDF.

* **De ce Grid Search?**
    * **Rationament:** Grid Search testeaza automat combinatii predefinite de **hiperparametri** (Ex: `alpha` pentru MNB, `C` pentru LR/SVC).
    * **Scop:** Gasirea configuratiei optime care maximizeaza metrica de performanta (`f1-score`) pentru fiecare model.

### 3.3 Evaluarea cu matricea de confuzie

* **De ce F1-Score si Matricea de Confuzie?**
    * **Rationament:** Setul de date este dezechilibrat (mai mult `ham` decat `spam`). Acuratetea simpla poate fi inselatoare. **F1-Score** si **Matricea de Confuzie** se concentreaza pe clasa minoritara.
    * **Scop:** Evidentierea erorilor critice, in special **False Negatives** (spam marcat gresit ca ham), care sunt cele mai costisitoare in contextul unui filtru anti-spam. Vizualizarea sub forma de grafic simplifica interpretarea acestor erori.