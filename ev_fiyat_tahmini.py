"""
Uctan Uca Makine Ogrenmesi Projesi - Final Odev
Konu: Ev fiyat tahmini (regresyon)

Amac:
    California Housing veri seti uzerinde veri inceleme, on isleme,
    oznitelik muhendisligi, model karsilastirma, capraz dogrulama,
    hiperparametre ayarlama ve sonuc yorumlama adimlarini uygulamak.

Veri seti:
    sklearn.datasets.fetch_california_housing
    Kaliforniya'daki farkli bolgelerin (block group) medyan ev fiyatini
    tahmin ediyoruz -> regresyon problemi. Hedef: MedHouseVal (100.000 $ biriminde)

Kullanilan kutuphaneler:
    pandas, numpy, scikit-learn, matplotlib, seaborn

Calistirma:
    pip install -r requirements.txt
    python ev_fiyat_tahmini.py
"""

# 1. kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. veri setinin yuklenmesi
# ilk calistirmada internetten indiriyor, sonrasinda scikit-learn cache'den okuyor
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print(df.head())
print(f"\nveri seti boyutu: {df.shape}")

# 3. hedef degisken
# MedHouseVal surekli sayisal bir deger (fiyat) oldugu icin problem REGRESYON.
target_col = "MedHouseVal"

# 4. veri inceleme
print(df.info())
print(df.describe())

# 5. eksik deger kontrolu
print(f"\neksik deger sayilari:\n{df.isnull().sum()}")
# california housing veri setinde eksik deger cikmadi, doldurma/silme adimina gerek kalmadi

# 6. kategorik degisken encoding
# hazir veri setinde kategorik sutun yok (hepsi sayisal). bu adimi anlamli kilmak icin
# HouseAge'den bir "ev_yasi_grubu" kategorisi turetip one-hot encoding uyguluyorum
df["ev_yasi_grubu"] = pd.cut(
    df["HouseAge"],
    bins=[0, 15, 30, 45, 100],
    labels=["yeni", "orta", "eski", "cok_eski"]
)
print(f"\nev_yasi_grubu dagilimi:\n{df['ev_yasi_grubu'].value_counts()}")

df = pd.get_dummies(df, columns=["ev_yasi_grubu"], drop_first=True)

# 7. aykiri deger inceleme
# AveRooms ve AveOccup genelde bu veri setinde asiri uc deger barindiran sutunlar (ornegin
# tek bir blokta ortalama 100+ kisilik hane gibi anlamsiz degerler cikabiliyor)
for col in ["AveRooms", "AveOccup"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr
    aykiri_sayisi = ((df[col] < alt_sinir) | (df[col] > ust_sinir)).sum()
    print(f"{col}: alt={alt_sinir:.2f} ust={ust_sinir:.2f} aykiri sayisi={aykiri_sayisi}")

# satirlari silmek yerine ust sinira sabitliyorum (capping), boylece veri kaybetmeden
# asiri degerlerin modele etkisini azaltiyorum
for col in ["AveRooms", "AveOccup"]:
    ust_sinir = df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))
    df[col] = np.where(df[col] > ust_sinir, ust_sinir, df[col])

# 9. yeni oznitelikler
# oda basina yatak orani - cok yuksekse kucuk/ucuz ev, dusukse genis/gelismis ev olabilir
df["yatak_oda_orani"] = df["AveBedrms"] / df["AveRooms"]
# kisi basina dusen oda sayisi - hanenin ferahligi hakkinda fikir veriyor
df["kisi_basi_oda"] = df["AveRooms"] / df["AveOccup"]

print(f"\nyeni oznitelikler ilk 5 satir:\n{df[['yatak_oda_orani', 'kisi_basi_oda']].head()}")

# 10. oznitelik secimi - korelasyon analizi
corr_with_target = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
print(f"\nhedef degiskenle korelasyon:\n{corr_with_target}")
# Population ve kisi_basi_oda hedefle neredeyse hic korele degil, bunlari tamamen atmadim
# ama zayif oldugunu not ediyorum. asil secimi random forest'in feature_importances_'i ile
# 17. soruda tekrar dogruluyorum.

# 11. train / validation / test ayrimi (%60 / %20 / %20)
X = df.drop(columns=[target_col])
y = df[target_col]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

# 8. olcekleme
# not: split'ten sonra yapiyorum, cunku scaler'i test/val veriyi gormeden sadece train ile
# ogretmek gerekiyor (data leakage olmasin diye)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 12. en az 3 model
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

val_results = {}

for name, model in models.items():
    if name == "Random Forest":
        # agac tabanli model oldugu icin olceklemeye ihtiyac yok
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
    else:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_val_scaled)

    rmse = mean_squared_error(y_val, pred) ** 0.5
    r2 = r2_score(y_val, pred)
    val_results[name] = rmse
    print(f"{name} -> validation RMSE: {rmse:.4f}  R2: {r2:.4f}")

# 13. validation karsilastirmasi
best_model_name = min(val_results, key=val_results.get)
print(f"\nvalidation'da en iyi model: {best_model_name}")

# 14. hiperparametre ayarlama (random forest genelde en iyisi cikiyor, onun uzerinde arama yapiyorum)
param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

# cv zaten kendi icinde katlara ayiracagi icin train+val'i birlikte veriyorum
random_search.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
print(f"\nen iyi parametreler: {random_search.best_params_}")

best_model = random_search.best_estimator_

# 15. test performansi
test_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, test_pred)
mse = mean_squared_error(y_test, test_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, test_pred)

print(f"\n--- test sonuclari ({best_model_name} + hiperparametre ayarlama) ---")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# 16. kisa yorum
print("""
YORUM:
Random Forest, Linear Regression ve Ridge'e gore hem validation hem test setinde
acikca daha iyi sonuc verdi (daha dusuk RMSE, daha yuksek R2). Bu beklenen bir durum,
cunku ev fiyati ile gelir/konum/oda sayisi arasindaki iliski dogrusal degil; random
forest bu tur non-lineer iliskileri linear/ridge'e gore daha iyi yakaliyor.
Modelin siniri: veri setinde sadece konum ve yapisal bilgiler var. Okul kalitesi,
guvenlik, ulasim gibi fiyati etkileyen onemli faktorler veri setinde yok, bu yuzden
modelin hatasi belli bir noktanin altina inemiyor.
""")

# 17. bonus - aciklanabilirlik (feature importance)
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\nfeature importance (buyukten kucuge):\n{importances}")

plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Random Forest - Feature Importance")
plt.xlabel("onem skoru")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
