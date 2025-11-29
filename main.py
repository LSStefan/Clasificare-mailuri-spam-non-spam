import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import seaborn as sns # type: ignore

df = pd.read_excel('email_spam.xlsx') 


def eliminate_dups():
    initial_rows = len(df)
    print(f"Numar de randuri inainte de eliminarea duplicatelor: {initial_rows}")
    df.drop_duplicates(subset=['v2'], inplace=True)
    final_rows = len(df)
    duplicates_removed = initial_rows - final_rows

    print(f"Numar de randuri dupa eliminarea duplicatelor: {final_rows}")
    print(f"Total duplicate eliminate: {duplicates_removed}")



def eliminate_stupid_columns():
    print("Structura DatFrame-ului inainte de eliminare")
    print(df.columns.tolist())

    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
    print("Structura DataFrame-ului dupa curatare")
    print(f"Coloane ramase: {df.columns.tolist()}")



#trece de la spam/ham la ceva numeric
def encode_target():
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['v1'])

    print("Maparea etichetelor")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

    print("\n--- Verificare Encodare (Primele rânduri) ---")

    print(df[['v1', 'Label_Encoded']].head())


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

    # afisarea valorilor pe grafic
    for i, count in enumerate(df['v1'].value_counts()):
        plt.text(i, count + 10, f'{count}', ha='center', fontsize=12)

    plt.show()

view_distribution()