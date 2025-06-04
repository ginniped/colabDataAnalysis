
from google.colab import files
uploaded = files.upload()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# Caricamento file Excel
file_path = 'c.xls'
df = pd.read_excel(file_path, sheet_name="Foglio1")

# Rinomina colonna iniziale
df.rename(columns={df.columns[0]: "Pressione"}, inplace=True)

# Trasformazione da wide a long
df_long = df.melt(id_vars=["Pressione"], var_name="Umidità", value_name="Tensione")
df_long.dropna(inplace=True)

# Estrai il valore numerico dell'umidità (es. da '18.86.1' prendi '18.86')
df_long["Umidità"] = df_long["Umidità"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

df_long.head()



# Analisi varianza per (pressione, umidità)
varianza = df_long.groupby(["Pressione", "Umidità"])["Tensione"].agg(["mean", "std", "var", "count"]).reset_index()
varianza.head(500)



# Grafico della risposta del sensore per diverse pressioni
pressioni_selezionate = [1.61]
# , 2.42, 3.23, 4.03, 4.84, 5.65, 6.45, 8.07, 9.68,  10.97, 12.1, 13.71,  15.33, 16.94

plt.figure(figsize=(10, 6))
for p in pressioni_selezionate:
    subset = df_long[df_long["Pressione"] == p]
    media = subset.groupby("Umidità")["Tensione"].mean().reset_index()
    plt.plot(media["Umidità"], media["Tensione"], marker='o', label=f"{p} MPa")

plt.xlabel("Umidità (%)")
plt.ylabel("Tensione media")
plt.title("Risposta del sensore a diverse pressioni")
plt.legend()
plt.grid(True)
plt.show()


# Grafico della risposta del sensore per diverse pressioni
pressioni_selezionate = [1.61, 2.42, 3.23, 4.03, 4.84, 5.65, 6.45]
# , 2.42, 3.23, 4.03, 4.84, 5.65, 6.45, 8.07, 9.68,  10.97, 12.1, 13.71,  15.33, 16.94

plt.figure(figsize=(10, 6))
for p in pressioni_selezionate:
    subset = df_long[df_long["Pressione"] == p]
    media = subset.groupby("Umidità")["Tensione"].mean().reset_index()
    plt.plot(media["Umidità"], media["Tensione"], marker='o', label=f"{p} MPa")

plt.xlabel("Umidità (%)")
plt.ylabel("Tensione media")
plt.title("Risposta del sensore a diverse pressioni")
plt.legend()
plt.grid(True)
plt.show()



# Grafico della risposta del sensore per diverse pressioni
pressioni_selezionate = [1.61,  2.42, 3.23, 4.03, 4.84, 5.65, 6.45, 8.07, 9.68,  10.97, 12.1, 13.71,  15.33, 16.94]
# 1.61,  2.42, 3.23, 4.03, 4.84, 5.65, 6.45, 8.07, 9.68,  10.97, 12.1, 13.71,  15.33, 16.94

plt.figure(figsize=(10, 6))
for p in pressioni_selezionate:
    subset = df_long[df_long["Pressione"] == p]
    media = subset.groupby("Umidità")["Tensione"].mean().reset_index()
    plt.plot(media["Umidità"], media["Tensione"], marker='o', label=f"{p} MPa")

plt.xlabel("Umidità (%)")
plt.ylabel("Tensione media")
plt.title("Risposta del sensore a diverse pressioni")
plt.legend()
plt.grid(True)
plt.show()



# Esporta l'analisi in un file Excel
varianza.to_excel("risultati_analisi_varianza.xlsx", index=False)
print("File esportato: risultati_analisi_varianza.xlsx")



# Regressione lineare per una pressione fissa
pressione_target = 1.61
subset = df_long[df_long["Pressione"] == pressione_target]
X = subset["Umidità"].values.reshape(-1, 1)
y = subset["Tensione"].values

model = LinearRegression()
model.fit(X, y)

print("Coefficiente (sensibilità):", model.coef_[0])
print("Intercetta:", model.intercept_)
print("R²:", model.score(X, y))

# Grafico della regressione
plt.scatter(X, y, label="Dati")
plt.plot(X, model.predict(X), color='red', label="Regressione")
plt.xlabel("Umidità (%)")
plt.ylabel("Tensione")
plt.title(f"Regressione lineare a {pressione_target} MPa")
plt.legend()
plt.grid(True)
plt.show()


'''
Per ogni pressione:
Prova 4 modelli:
  Regressione Lineare
  Polinomiale (grado 3)
  Random Forest
  SVR (Support Vector Regression)
Calcola l'R² per ognuno
Sceglie automaticamente il migliore
Disegna il grafico del modello migliore per ogni pressione
'''

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Tutte le pressioni uniche nel dataset
pressioni_uniche = sorted(df_long["Pressione"].dropna().unique())

# Dizionario dei modelli disponibili
def get_models():
    return {
        "Lineare": LinearRegression(),
        "Polinomiale (grado 3)": make_pipeline(PolynomialFeatures(3), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf')
    }

# Analisi per ogni pressione
for p in pressioni_uniche:
    subset = df_long[df_long["Pressione"] == p]

    if subset.shape[0] < 4:
        print(f"⚠️  Pressione {p} MPa: troppi pochi punti, salto.")
        continue

    X = subset[["Umidità"]].values
    y = subset["Tensione"].values

    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    y_best_pred = None

    models = get_models()

    for name, model in models.items():
        try:
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = name
                y_best_pred = y_pred
        except Exception as e:
            print(f"Errore con modello {name} a pressione {p}: {e}")

    # Output e grafico
    print(f"✅ Pressione {p} MPa - Miglior modello: {best_model_name} (R² = {best_r2:.4f})")

    # Ordina per visualizzare meglio il grafico
    ordine = np.argsort(X.flatten())
    X_ord = X.flatten()[ordine]
    y_ord = y_best_pred[ordine]

    # Grafico
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Dati reali")
    plt.plot(X_ord, y_ord, color="red", label=f"{best_model_name} (R² = {best_r2:.2f})")
    plt.title(f"Pressione {p} MPa - Miglior modello: {best_model_name}")
    plt.xlabel("Umidità (%)")
    plt.ylabel("Tensione")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
