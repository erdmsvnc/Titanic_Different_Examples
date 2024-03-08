# titanic_different_examples

# 1-) ÖN İŞLEME KODLARI 🥇


Gerekli kütüphaneleri import ediyoruz
---
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
Gerekli datasetleri ekliyoruz
---
```
df = pd.read_csv("gender_submission.csv")
df1 = pd.read_csv("test.csv")
df2 = pd.read_csv("train.csv")
```

Diğer örneklerden farklı olarak gender_submission.csv ve test.csv dosyamızı concat ediyoruz ve onun üzerinden örneklerimizi oluşturacağız
---

```
df_df1_concat = pd.concat([df1,df], axis=1)
```

Df deki bütün boş yerleri ortalama ile doldurduğu için önce agenin ortalamasını bulduk sonra doldurulmuş halini yalnız olarak ayırdık 
sonra asıl df deki age sütununu attık ve ayırdığımız ortalama sütunu ile birleştirdik
---
```
ortalama_sutun1 = df_df1_concat['Age'].mean()
df_df1_concat_fill = df_df1_concat['Age'].fillna(ortalama_sutun1)
df_df1_concat.pop('Age')
df_df1_concat_fill_son = pd.concat([df_df1_concat, df_df1_concat_fill], axis=1)
```

# 2-) Gösterimler 🥈


Belirlediğimiz yaş aralıkalarının hangi oranda hayatta kaldığı gösterimi
---


```
full_data = [ df_df1_concat_fill_son ,df2 ]

for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
df_df1_concat_fill_son['CategoricalAge'] = pd.cut(df_df1_concat_fill_son['Age'], 5)

print (df_df1_concat_fill_son[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
```

<img width="982" alt="Ekran Resmi 2024-03-08 11 57 11" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/f3a3bcd2-d2be-42e6-9216-86f675689847">


Embarked sütunumuzda ki verilere dayanarak belirtilen konumdan kaç yolcu bindiği gösterimi
---


```
plt.hist(df_df1_concat_fill_son.Embarked, bins=20)
plt.title("Konumlardan Binen Yolcu Sayisi")
plt.show()
```

<img width="1420" alt="Ekran Resmi 2024-03-08 11 58 29" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/7704f25d-d148-4fe4-9acf-465aa0de5aa4">


Erkeklerin ve kadınların ayrı ayrı toplam ne kadar bilet ücreti ödedikleri gösterimi
---


```
plt.bar(df_df1_concat_fill_son.Sex, df_df1_concat.Fare, color = "blue")
plt.title("Cinsiyetlerin Toplam Ödedikleri Bilet Fiyatları")
plt.show()
```

<img width="1407" alt="Ekran Resmi 2024-03-08 12 00 18" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/1f29d929-05fb-4299-9417-6f744bc0b14d">

Belirtilen bilet türünde ki toplam alım sayısı
---


```
plt.hist(df_df1_concat_fill_son.Pclass, bins = 20)
plt.title("Belirtilen Bilet Türünde ki Toplam Alım Sayısı")
plt.show()
```

<img width="1405" alt="Ekran Resmi 2024-03-08 12 02 15" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/4a0cd78c-46ea-4859-98f8-1b7d029c9f06">

Toplam erkek ve kadın sayısının ayrı ayrı gösterimi
---


```
df_df1_concat_fill_son['Sex'].value_counts().plot(kind='bar', color=['blue', 'pink'])
plt.title('Cinsiyet Histogramı')
plt.xlabel('Cinsiyet')
plt.ylabel('Frekans')
plt.show()
```

<img width="1461" alt="Ekran Resmi 2024-03-08 12 03 59" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/18e19308-99a3-401f-808b-cc9cb2c2e579">


Embarked sütunumuzda ki verilere dayanarak belirtilen konumdan kaç yolcu bindiği gösteriminin daha modern gösterimi
---

```
df_df1_concat_fill_son['Embarked'].value_counts().plot(kind='bar', color=['blue', 'pink','green'])
plt.title('Binis Histogrami')
plt.xlabel('Konumlar')
plt.ylabel('Frekans')
plt.show()
```

<img width="1441" alt="Ekran Resmi 2024-03-08 12 05 44" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/3cf1ebd2-ed59-488a-a02e-4db4f411107e">


Bilet türlerinin ne kadar satıldığı bilgisinin 'pie' türünde gösterimi
---


```
df_df1_concat_fill_son['Pclass'].value_counts().plot(kind = 'pie', figsize = (8,6), shadow = True, autopct = '%1.2f%%')
plt.title("Bilet Türlerinin Karşılaştırılması")
plt.axis('equal')
plt.show()
```

<img width="619" alt="Ekran Resmi 2024-03-08 12 07 34" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/c442c748-4281-47fb-9773-94109ed5a6c6">

Embarked sütunumuzda ki verilere dayanarak belirtilen konumdan kaç yolcu bindiği gösteriminin daha modern ve 'pie' türünde gösterimi
---


```
df_df1_concat_fill_son['Embarked'].value_counts().plot(kind = 'pie', figsize = (8,6), shadow = True, autopct = '%1.2f%%')
plt.title("Yolcuların Binis Yerlerinin Dagilimi")
plt.axis('equal')
plt.show()
```

<img width="633" alt="Ekran Resmi 2024-03-08 12 09 16" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/b1f9a5dc-27ce-4e2d-a003-5139ba7b86b0">

Cinsiyetlere ayrılmış bir şekilde hangi bilet türünden kaç kişi aldığının gösterimi
---


```
df_df1_concat_fill_son.groupby(['Pclass', 'Sex']).size().unstack().plot(kind='bar', stacked=False)
plt.title('Bilet Türü Ve Cinsiyet Histogramı')
plt.xlabel('pclass')
plt.ylabel('Frekans')
plt.show()
```

<img width="1434" alt="Ekran Resmi 2024-03-08 12 10 52" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/03f0f277-d4aa-4fb4-a5e1-dea2f790e29a">


Cinsiyetlere ayrılmış bir şekilde hangi konumdan kaç yolcu bindiğinin gösterimi
---

```
df_df1_concat_fill_son.groupby(['Embarked', 'Sex']).size().unstack().plot(kind='bar', stacked=False)
plt.title('Binen Yolcu Konumları Ve Cinsiyet Histogramı')
plt.xlabel('Embarked')
plt.ylabel('Frekans')
plt.show()
```

<img width="1444" alt="Ekran Resmi 2024-03-08 12 12 33" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/76771b05-3393-452c-be58-91758ac8ec12">

Belirtilen Konumlardan Binen Yolcuların Hayatta Kalma Sayılarının Karşılaştırılmasının gösterimi
---


```
df_df1_concat_fill_son.groupby(['Embarked', 'Survived']).size().unstack().plot(kind = 'bar', stacked = True)
plt.title("Belirtilen Konumlardan Binen Yolcuların Hayatta Kalma Sayılarının Karşılaştırılması")
plt.show()
```

<img width="1448" alt="Ekran Resmi 2024-03-08 12 14 42" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/527fcdc7-ffb0-4763-81e3-9c65a162ee62">


Alınan bilet türlerinin hayatta kalma oranlarının gösterimi
---


```
df_df1_concat_fill_son.groupby(['Pclass', 'Survived']).size().unstack().plot(kind='bar', stacked=True )
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()
```

<img width="1448" alt="Ekran Resmi 2024-03-08 12 16 30" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/781fda4f-4b0a-448c-8b9c-db5baf77c8ed">

Belirtilen konumlardan binen yolcuların aldıkları bilet türlerinin karşılaştırılmasının gösterimi
---


```
df_df1_concat_fill_son.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar', stacked=False)
plt.title('Bilet Türü Ve Binis Yerleri')
plt.xlabel('Embarked')
plt.ylabel('Frekans')
plt.show()
```

<img width="1422" alt="Ekran Resmi 2024-03-08 12 18 19" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/c8741194-889b-4c46-ad07-11949430ba4d">


!!! Bilet türünün kalitesi arttıkça ölüm oranının azaldığı görülmüştür yani bilet türü iyi olanlar önce kurtarılmıştır !!!
---


# 3-) Seaborn kütüphanesi denemeler 🥉

Yaş yoğunluklarının dağılımı gösterimi
---


```
sns.kdeplot(df_df1_concat_fill_son.Age, shade = True)
plt.show()
```

<img width="1439" alt="Ekran Resmi 2024-03-08 12 22 23" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/e6735d48-f68f-4864-91b6-1587bcaecdf4">



Binen yolcuların bilet türlerinin dağılımının gösterimi
---


```
sns.countplot(y='Pclass', data=df_df1_concat_fill_son, palette='rocket')
plt.show()
```

<img width="1420" alt="Ekran Resmi 2024-03-08 12 25 02" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/d952815d-431e-4712-a861-d43cfab6449d">



Belirtilen konumlardan binen yolcuların yaş ortalaması grafiğinin gösterimi
---
```
mean_age_by_embarked = df_df1_concat_fill_son.groupby('Embarked')['Age'].mean().reset_index()

sns.barplot(x='Embarked', y='Age', data=mean_age_by_embarked)
plt.show()
```

<img width="1462" alt="Ekran Resmi 2024-03-08 12 26 24" src="https://github.com/buzzi0/titanic_different_examples/assets/103946477/d2dfac43-2d29-4292-aeda-b414943f794c">















