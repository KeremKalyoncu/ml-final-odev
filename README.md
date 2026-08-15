# Ev Fiyat Tahmini - ML Final Odev

Turkiye Yapay Zeka Akademisi Makine Ogrenmesi dersi final odevi.
Konu: uctan uca bir makine ogrenmesi projesi (regresyon).

## Amac
California Housing veri seti uzerinde veri inceleme, on isleme, oznitelik
muhendisligi, birden fazla model egitip karsilastirma, capraz dogrulama,
hiperparametre ayarlama ve sonuc yorumlama.

## Veri seti
`sklearn.datasets.fetch_california_housing` - Kaliforniya'daki bolgelere ait
ev fiyati verisi (regresyon problemi, hedef: ev fiyati).

## Nasil calistirilir
```bash
pip install -r requirements.txt
python ev_fiyat_tahmini.py
```

## Sonuc

Validation karsilastirmasi:

| Model             | RMSE   | R2     |
|-------------------|--------|--------|
| Linear Regression | 0.6606 | 0.6822 |
| Ridge              | 0.6606 | 0.6822 |
| Random Forest      | 0.5103 | 0.8104 |

Random Forest acik ara en iyi sonucu verdi, hiperparametre aramasi (RandomizedSearchCV)
sonrasi test setinde:

- MAE: 0.3260
- MSE: 0.2525
- RMSE: 0.5025
- R2: 0.8073

En onemli degisken `MedInc` (bolgenin medyan geliri) - feature importance'in yarisindan
fazlasini tek basina o aciklıyor. Ev fiyati ile gelir/konum arasindaki iliski dogrusal
olmadigi icin Random Forest, Linear Regression'a gore belirgin fark atiyor. Modelin
siniri: veri setinde okul kalitesi, guvenlik gibi fiyati etkileyen faktorler yok, bu
yuzden hata belli bir noktanin altina inemiyor.

`feature_importance.png` dosyasinda gorsel de mevcut.
