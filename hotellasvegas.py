#%%
import pandas as pd #pentru prelucrarea si manipularea datelor
import matplotlib.pyplot as plt #pentru grafice simple
import numpy as np
import seaborn as sns #pentru grafice statistice complexe
import statsmodels.api as sm 


df = pd.read_csv("https://raw.githubusercontent.com/andreipatrascu51-ctrl/g54ff/refs/heads/main/LasVegasTripAdvisorReviews-Dataset.csv", sep=';')


print("Primele 5 inregistrari:")
print(df.head(5))
#%%

print("Ultimele 5 inregistrari:")
print(df.tail(5))
#%%

print("Informatii generale")
print(df.info())
#%%

print("Valori lipsa:")
print(df.isnull().sum())
#%%

print("Tipurile de date:")
print(df.dtypes)


randuri, coloane=df.shape
print(f"Numarul de inregistrari: {randuri} randuri, {coloane} coloane")
#%%


variabile_numerice=df.select_dtypes(include=["int64", "float64"])
variabile_categoriale=df.select_dtypes(include=["object"])


df[variabile_numerice.columns] = df[variabile_numerice.columns].fillna(df[variabile_numerice.columns].median())
df[variabile_categoriale.columns] = df[variabile_categoriale.columns].fillna(df[variabile_categoriale.columns].mode().iloc[0])

print(f"Variabile numerice:{variabile_numerice.columns}")
print(f"Variabile categoriale: {variabile_categoriale.columns}")


print("Statistici descriptive:")
print(df[variabile_numerice.columns].describe())


media = variabile_numerice.mean()
mediana = variabile_numerice.median()
deviatia_standard = variabile_numerice.std()
coeficientvariatie = deviatia_standard / media      
q1 = variabile_numerice.quantile(0.25)
q3 = variabile_numerice.quantile(0.75)
iqr = q3-q1


 #%%
print("Media:")
print(media)
print("Mediana:")
print(mediana)
print("Deviatia standard:")
print(deviatia_standard)
print("Coeficient de variatie:")
print(coeficientvariatie)
print("Q1:")
print(q1)
print("Q3:")
print(q3)
print("IQR:")
print(iqr)

plt.figure(figsize=(10,6))
plt.hist(df["Score"],bins=5,color='orange',edgecolor='black') 
plt.xlabel("Score") 
plt.ylabel("Numar recenzii") 
plt.title("Distributia scorurilor") 
plt.show()
#%%

mediescortipturist = df.groupby("Traveler type")["Score"].mean() 
print("Scor mediu in functie de tipul turistului:")
print(mediescortipturist)


mediescortipturist.plot(kind='bar',color='skyblue')
plt.title("Scor mediu in functie de tipul turistului")
plt.ylabel("Scor mediu")
plt.show()


mediivariabilepescor = df.groupby("Score")[variabile_numerice.columns].mean()
print("Medii ale variabilelor numerice in functie de score:")
print(mediivariabilepescor)

scormediu = df["Score"].mean() 
scormax = df["Score"].max()
scormin = df["Score"].min()
topturisti = df["Traveler type"].value_counts()


print("Scor mediu:", scormediu)
print("Scor maxim:", scormax)
print("Scor minim", scormin)
print("Top tipuri de turisti:", topturisti)

#%%


df_dummy = pd.get_dummies(df,drop_first=True)
print("Dummy")  
print(df_dummy.head(5))
#%%

corelatiescore = df_dummy.corr(numeric_only=True)["Score"].drop("Score")
print("Corelatii intre variabilele independente si cea dependenta(Score)")
print(corelatiescore)

corelatieindependente = df_dummy.drop(columns="Score").corr(numeric_only=True)
print("Corelatii intre variabilele independente:")
print(corelatieindependente)

plt.figure(figsize=(14,10)) 
sns.heatmap(corelatieindependente, annot=False) 
plt.title("Corelatii intre variabilele independente")
plt.show()

#%%
X = df_dummy.drop(columns="Score") 
Y = df_dummy["Score"] 

X=X.astype(float) 
Y=Y.astype(float)

X = sm.add_constant(X) 

model = sm.OLS(Y,X).fit() 
print(model.summary()) 

#%%
p_values = model.pvalues 

variabilesemnificative = p_values[p_values < 0.05].index 
variabilesemnificative = variabilesemnificative.drop("const") 

X2=df_dummy[variabilesemnificative] 
X2=X2.astype(float)

X2 = sm.add_constant(X2) 

model2 = sm.OLS(Y,X2).fit() 
print(model2.summary())
#%%

p_values = model2.pvalues 

variabilesemnificative = p_values[p_values < 0.05].index
variabilesemnificative = variabilesemnificative.drop("const")

X3=df_dummy[variabilesemnificative]

X3=X3.astype(float)

X3=sm.add_constant(X3)

model3=sm.OLS(Y,X3).fit()
print(model3.summary())

#%%
df["Score_Predict"] = model3.predict(X3) 
df["Reziduuri"] = df["Score"] - df["Score_Predict"] 

plt.hist(df["Reziduuri"], bins= 20) 
plt.title("Distributia reziduurilor") 
plt.xlabel("Valorile reziduurilor")
plt.ylabel("Numarul de observatii ")
plt.show()

sm.qqplot(df["Reziduuri"], line = "s") 
plt.title("Q-Q plot al reziduurilor") 
plt.show()

sns.scatterplot(x=df["Score_Predict"], y=df["Reziduuri"]) 
plt.axhline(0) 
plt.title("Reziduuri vs Valori estimate")
plt.show()


from scipy.stats import shapiro 
stat,p = shapiro(df["Reziduuri"])
print("p-value:",p)
if p > 0.05:
    print("Reziduurile sunt normale deoarece p > 0.05")
else:
    print("Reziduurile nu sunt normale deoarece p < 0.05")

#%%


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame() 
vif["Variabila"] = X3.columns 
vif["VIF"] = [variance_inflation_factor(X3.values,i) for i in range(X3.shape[1])]
print("VIF pentru variabile independente:")
print(vif)

# %%

#TEST Breusch-Pagen pentru heteroscedasticitate
from statsmodels.stats.diagnostic import het_breuschpagan

bp = het_breuschpagan(model3.resid, model3.model.exog)
b = bp[1]
print("Breusch-Pagan p-value:", bp[1])
if b > 0.05:
    print("Nu exista heteroscedasticitate, ipoteza OLS este indeplinita")
else:
    print("Exista heteroscedasticitate, rezultatele OLS pot fi distorsionate")
