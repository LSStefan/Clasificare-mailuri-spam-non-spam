import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import seaborn as sns # type: ignore
import re
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import nltk # type: ignore
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

nltk.download('stopwords')  # type: ignore
nltk.download('wordnet') # type: ignore
nltk.download('punkt') # type: ignore

df = pd.read_excel('email_spam.xlsx') 



#Vericați structura setului de date.
def view_structure():
    print(df.columns.tolist())
    print(df.head())




#Vericați și eliminați duplicatele din dataset
def eliminate_dups():
    initial_rows = len(df)
    print(f"Numar de randuri inainte de eliminarea duplicatelor: {initial_rows}")
    df.drop_duplicates(subset=['v2'], inplace=True)
    final_rows = len(df)
    duplicates_removed = initial_rows - final_rows

    print(f"Numar de randuri dupa eliminarea duplicatelor: {final_rows}")
    print(f"Total duplicate eliminate: {duplicates_removed}")




#Eliminați coloanele inutile.
def eliminate_stupid_columns():
    print("Structura DatFrame-ului inainte de eliminare")
    print(df.columns.tolist())

    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
    print("Structura DataFrame-ului dupa curatare")
    print(f"Coloane ramase: {df.columns.tolist()}")




#Encodați variabila țintă.
#trece de la spam/ham la ceva numeric
def encode_target():
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['v1'])

    print("Maparea etichetelor")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

    print("\nVerificare Encodare (Primele randuri)")

    print(df[['v1', 'Label_Encoded']].head())

    y_labels = df['Label_Encoded']
    return y_labels





#Vizualizați distribuția variabilei țintă.
def view_distribution():
    print("Distributia valprilor(%)")
    # procent ham si spam
    distribution = df['v1'].value_counts(normalize=True) * 100
    print(distribution)

    # Crearea graficului
    plt.figure(figsize=(7, 5))
    sns.countplot(x='v1', data=df, palette='viridis')

    plt.title('Distributia mesajelor: Ham vs. Spam')
    plt.xlabel('Clasa (v1)')
    plt.ylabel('Numar de Mesaje')

    

    plt.show()





# Inițializare instrumente
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



#Implementați și aplicați metode de curățare a textului peste caracteristicile rămase.
def clean_text(text):
    # 1. Normalizare: Convertirea la litere mici
    text = text.lower()
    
    # 2. Eliminarea Elementelor Zgomotoase (Noise Removal)
    # Elimina URL-urile
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    # Elimina numerele
    text = re.sub(r'\d+', '', text)
    # Elimina semnele de punctuatie și caracterele non-alfanumerice (pastrand doar spatiile)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenizare (Spargerea textului în cuvinte)
    words = text.split()
    
    # 4. Eliminarea Stop Words și 5. Lematizare
    processed_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words and len(word) > 1 # Filtreaza stop words și cuvintele cu o singură litera
    ]
    
    # 6. Reconstruirea textului curatat
    return " ".join(processed_words)


#Transformați textul curățat în reprezentare numerică.
def text_to_numbers():
    # X_full este setul complet de texte curatate (caracteristicile)
    X_full = df['Clean_Text']
    # y_full este setul complet de etichete encodate (variabila tinta)
    y_full = df['Label_Encoded'] 

    # 1. Initializare TF-IDF
    # max_features=5000 limiteaza vocabularul la cele mai frecvente 5000 de cuvinte
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

    # 2. Aplicarea TF-IDF pe setul COMPLET (X_full)
    # fit_transform invata vocabularul (fit) si il vectorizeaza imediat (transform)
    X_vectorized_full = tfidf_vectorizer.fit_transform(X_full)

    print("--- Reprezentare Numerica (Set Complet) ---")
    print(f"Set Vectorizat Complet (Linii, Cuvinte): {X_vectorized_full.shape}")

    return X_vectorized_full




def train_and_evaluate_models(X_vectorized, y_labels):
    """
    Antreneaza, optimizeaza si evalueaza modele de clasificare pe datele vectorizate.

    Args:
        X_vectorized (scipy.sparse.matrix): Caracteristicile (textul vectorizat TF-IDF).
        y_labels (pandas.Series): Variabila tinta (etichetele 0 sau 1).
    """

    # --- 1. Impartirea Datelor ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, 
        y_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_labels
    ) 
    print("--- 1. Impartirea Datelor (Train/Test Split) realizata. ---")
    
    best_models = {}
    confusion_matrices = {} 

    # --- 2. Definirea Modelelor si a Hiperparametrilor ---
    models = {
        'MNB': (MultinomialNB(), {'alpha': [0.1, 0.5, 1.0, 2.0]}), # type: ignore
        'LR': (LogisticRegression(max_iter=1000, random_state=42),  # type: ignore
               {'C': [0.1, 1, 10], 'solver': ['liblinear']}),
        'SVC': (SVC(random_state=42), {'C': [1, 10], 'kernel': ['linear']}) # type: ignore
    }
    
    # --- 3. Grid Search si Antrenare ---
    print("\n--- 3. Grid Search pentru optimizarea hiperparametrilor ---")
    
    for name, (model, params) in models.items():
        print(f"Antrenare {name}...")
        
        # Grid Search cu 5-fold Cross-Validation si metrica F1
        grid_search = GridSearchCV(model, params, cv=5, scoring='f1', verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_models[name] = grid_search.best_estimator_
        print(f"{name} - Hiperparametri optimi: {grid_search.best_params_}")
        
    # --- 4. Evaluare pe Setul de Testare ---
    print("\n--- 4. Evaluarea Modelului cu Hiperparametri Optimi ---")
    labels = ['Ham (0)', 'Spam (1)']
    
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        
        # Raport de Clasificare
        print(f"\n==================== {name} ====================")
        print(f"Acuratete (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=labels))
        
        # Stocare Matrice de Confuzie
        cm = confusion_matrix(y_test, y_pred) # type: ignore
        confusion_matrices[name] = cm
        
    # --- 5. Vizualizarea Matricilor de Confuzie ---
    print("\n--- 5. Afisare Matrice de Confuzie sub forma de grafic ---")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Matricile de Confuzie (Cele Mai Bune Modele)', fontsize=16)
    

    for i, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[i]
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    ax=ax,
                    xticklabels=labels,
                    yticklabels=labels)
        
        ax.set_title(f'Model: {name}')
        ax.set_xlabel('Prezis')
        ax.set_ylabel('Real')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return best_models

# --- EXEMPLU DE APELARE A FUNCTIEI ---
# Presupunand ca X_vectorized_full si y_full sunt variabilele tale finale:
# final_models = train_and_evaluate_models(X_vectorized_full, y_full)


def main():
    eliminate_dups()
    eliminate_stupid_columns()
    y_labels = encode_target()
    df['Clean_Text'] = df['v2'].apply(clean_text)
    x_vectorized = text_to_numbers()
    train_and_evaluate_models(x_vectorized,y_labels)



main()



