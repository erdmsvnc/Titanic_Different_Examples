import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("gender_submission.csv")
df1 = pd.read_csv("test.csv")
df2 = pd.read_csv("train.csv")


df_df1_concat = pd.concat([df1,df], axis=1)
"""
df deki bütün boş yerleri ortalama ile doldurduğu için önce agenin ortalamasını bulduk sonra doldurulmuş halini yalnız olarak ayırdık 
sonra asıl df deki age sütununu attık ve ayırdığımız ortalama sütunu ile birleştirdik
"""
ortalama_sutun1 = df_df1_concat['Age'].mean()
df_df1_concat_fill = df_df1_concat['Age'].fillna(ortalama_sutun1)
df_df1_concat.pop('Age')
df_df1_concat_fill_son = pd.concat([df_df1_concat, df_df1_concat_fill], axis=1)

"""
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
"""
plt.hist(df_df1_concat_fill_son.Embarked, bins=20)
plt.title("Konumlardan Binen Yolcu Sayisi")
plt.show()
"""
plt.bar(df_df1_concat_fill_son.Sex, df_df1_concat.Fare, color = "blue")
plt.title("Cinsiyetlerin Toplam Ödedikleri Bilet Fiyatları")
plt.show()

plt.hist(df_df1_concat_fill_son.Pclass, bins = 20)
plt.title("Belirtilen Bilet Türünde ki Toplam Alım Sayısı")
plt.show()

df_df1_concat_fill_son['Sex'].value_counts().plot(kind='bar', color=['blue', 'pink'])
plt.title('Cinsiyet Histogramı')
plt.xlabel('Cinsiyet')
plt.ylabel('Frekans')
plt.show()

df_df1_concat_fill_son['Embarked'].value_counts().plot(kind='bar', color=['blue', 'pink','green'])
plt.title('Binis Histogrami')
plt.xlabel('Konumlar')
plt.ylabel('Frekans')
plt.show()



df_df1_concat_fill_son['Pclass'].value_counts().plot(kind = 'pie', figsize = (8,6), shadow = True, autopct = '%1.2f%%')
plt.title("Bilet Türlerinin Karşılaştırılması")
plt.axis('equal')
plt.show()




plt.plot(df_df1_concat_fill_son.PassengerId,df_df1_concat.Fare,color = "blue" )
plt.xlabel("Yolcu Id")
plt.ylabel("Bilet Fiyatlari")
plt.title("Bilet Fiyatlarinin Degisimi")
plt.show()


df_df1_concat_fill_son['Embarked'].value_counts().plot(kind = 'pie', figsize = (8,6), shadow = True, autopct = '%1.2f%%')
plt.title("Yolcuların Binis Yerlerinin Dagilimi")
plt.axis('equal')
plt.show()

df_df1_concat_fill_son.groupby(['Pclass', 'Sex']).size().unstack().plot(kind='bar', stacked=False)
plt.title('Bilet Türü Ve Cinsiyet Histogramı')
plt.xlabel('pclass')
plt.ylabel('Frekans')
plt.show()


df_df1_concat_fill_son.groupby(['Embarked', 'Sex']).size().unstack().plot(kind='bar', stacked=False)
plt.title('Binen Yolcu Konumları Ve Cinsiyet Histogramı')
plt.xlabel('Embarked')
plt.ylabel('Frekans')
plt.show()


df_df1_concat_fill_son.groupby(['Embarked', 'Survived']).size().unstack().plot(kind = 'bar', stacked = True)
plt.title("Belirtilen Konumlardan Binen Yolcuların Hayatta Kalma Sayılarının Karşılaştırılması")
plt.show()

df_df1_concat_fill_son.groupby(['Pclass', 'Survived']).size().unstack().plot(kind='bar', stacked=True )
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()


df_df1_concat_fill_son.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar', stacked=False)
plt.title('Bilet Türü Ve Binis Yerleri')
plt.xlabel('Embarked')
plt.ylabel('Frekans')
plt.show()

#Bilet türünün kalitesi arttıkça ölüm oranının azaldığı görülmüştür yani bilet türü iyi olanlar önce kurtarılmıştır

sns.kdeplot(df_df1_concat_fill_son.Age, shade = True)
plt.show()

sns.catplot(x = "Pclass", y = "Survived", hue='Fare', kind='bar', data=df_df1_concat_fill_son)
plt.show()


sns.countplot(y='Pclass', data=df_df1_concat_fill_son, palette='rocket')
plt.show()
"""
mean_age_by_embarked = df_df1_concat_fill_son.groupby('Embarked')['Age'].mean().reset_index()

sns.barplot(x='Embarked', y='Age', data=mean_age_by_embarked)
plt.show()
"""
df_df1_concat_fill_son.info()
df_df1_concat_fill_son.describe()
df_df1_concat_fill_son.tail()


"""

