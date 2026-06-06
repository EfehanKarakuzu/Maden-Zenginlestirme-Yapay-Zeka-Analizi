import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. MADEN TESİSİ SENSÖR VERİLERİNİ OLUŞTURMA
np.random.seed(42)
n_samples = 1000

data = {
    'Demir_Besleme_Yuzdesi': np.random.uniform(50, 60, n_samples),   
    'Silika_Besleme_Yuzdesi': np.random.uniform(10, 15, n_samples),  
    'Flotasyon_Hava_Debisi': np.random.uniform(200, 350, n_samples), 
    'Flotasyon_Kopuk_Seviyesi': np.random.uniform(1.5, 4.0, n_samples), 
    'Flotasyon_PH': np.random.uniform(8.5, 10.5, n_samples)          
}

df = pd.DataFrame(data)

df['Hedef_Demir_Orani'] = (df['Demir_Besleme_Yuzdesi'] * 1.1) + (df['Flotasyon_Hava_Debisi'] * 0.02) - (df['Silika_Besleme_Yuzdesi'] * 0.3) + np.random.normal(0, 0.5, n_samples)
df['Hedef_Silika_Orani'] = (df['Silika_Besleme_Yuzdesi'] * 0.8) - (df['Flotasyon_Kopuk_Seviyesi'] * 0.5) + (df['Flotasyon_PH'] * 0.2) + np.random.normal(0, 0.2, n_samples)

print("--- MADEN TESİSİ VERİ SETİ (İLK 5 SATIR) ---")
print(df.head())

# 2. YAPAY ZEKA MODELİNİN EĞİTİLMESİ
X = df[['Demir_Besleme_Yuzdesi', 'Silika_Besleme_Yuzdesi', 'Flotasyon_Hava_Debisi', 'Flotasyon_Kopuk_Seviyesi', 'Flotasyon_PH']]
y_demir = df['Hedef_Demir_Orani']
y_silika = df['Hedef_Silika_Orani']

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X, y_demir, test_size=0.2, random_state=42)
X_train_si, X_test_si, y_train_si, y_test_si = train_test_split(X, y_silika, test_size=0.2, random_state=42)

model_demir = RandomForestRegressor(n_estimators=100, random_state=42)
model_silika = RandomForestRegressor(n_estimators=100, random_state=42)

print("\nYapılan işlemler analiz ediliyor ve yapay zeka eğitiliyor...")
model_demir.fit(X_train_fe, y_train_fe)
model_silika.fit(X_train_si, y_train_si)

# 3. BAŞARI ORANI VE TAHMİNLER
tahmin_demir = model_demir.predict(X_test_fe)
tahmin_silika = model_silika.predict(X_test_si)

basari_demir = r2_score(y_test_fe, tahmin_demir)
basari_silika = r2_score(y_test_si, tahmin_silika)

print("\n--- YAPAY ZEKA TAHMİN BAŞARI SONUÇLARI ---")
print(f"Demir (Fe) Oranı Tahmin Başarısı: %{basari_demir * 100:.2f}")
print(f"Silika (SiO2) Oranı Tahmin Başarısı: %{basari_silika * 100:.2f}")

# 4. GRAFİKLE GÖSTERME
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test_fe, tahmin_demir, color='blue', alpha=0.5)
plt.title('Demir: Gerçek vs Tahmin')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Yapay Zeka Tahmini')

plt.subplot(1, 2, 2)
plt.scatter(y_test_si, tahmin_silika, color='green', alpha=0.5)
plt.title('Silika: Gerçek vs Tahmin')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Yapay Zeka Tahmini')

plt.tight_layout()
print("\nGrafik penceresi açılıyor...")
plt.show()