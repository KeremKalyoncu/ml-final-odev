"""
Uctan Uca Makine Ogrenmesi Projesi - Final Odev
Konu: Ev fiyat tahmini (regresyon)

Amac:
    California Housing veri seti uzerinde veri inceleme, on isleme,
    oznitelik muhendisligi, model karsilastirma, capraz dogrulama,
    hiperparametre ayarlama ve sonuc yorumlama adimlarini uygulamak.

Veri seti:
    sklearn.datasets.fetch_california_housing (regresyon problemi, hedef: ev fiyati)

Kullanilan kutuphaneler:
    pandas, numpy, scikit-learn, matplotlib, seaborn

Nasil calisir:
    python ev_fiyat_tahmini.py
"""

# 1. kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: veri setini yukle (fetch_california_housing), df'e cevir, problemi aciklayan yorum ekle
# TODO: hedef degiskeni belirle (regresyon)
# TODO: veri inceleme (head, shape, dtypes, describe)
# TODO: eksik deger kontrolu
# TODO: kategorik degisken varsa encode et
# TODO: aykiri deger inceleme
# TODO: olcekleme
# TODO: en az 2 yeni oznitelik uret
# TODO: en az 1 oznitelik secimi (korelasyon / feature importance / lasso)
# TODO: train / validation / test ayir
# TODO: en az 3 model egit (Linear, Ridge/Lasso, RandomForestRegressor ...)
# TODO: validation karsilastirma
# TODO: en iyi model icin grid/random search
# TODO: test performansi (MAE, MSE, RMSE, R2)
# TODO: model yorumu
# TODO: bonus - aciklanabilirlik (feature importance / shap)
