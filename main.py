import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Lee el archivo CSV en un DataFrame de pandas
df = pd.read_csv('clientes.csv')
print(df.head())

escalador=MinMaxScaler().fit(df.values)
df=pd.DataFrame(escalador.transform(df.values),columns=["Saldo","transacciones"])
print(df)

kmeans=KMeans(n_clusters=3).fit(df.values)
print(kmeans.labels_)

#agregamos un titulo que identifica los k grupos
df["cluster"]=kmeans.labels_
print(kmeans.cluster_centers_,kmeans.inertia_) 

#instrucciones para graficar los clusters
plt.figure(figsize=(6,5),dpi=100)
colores=["red","blue","orange","black","purple","pink","brown"]
for cluster in range(kmeans.n_clusters):
	plt.scatter(df[df["cluster"] == cluster] ["Saldo"],
               df[df["cluster"]==cluster]["transacciones"],
               marker="o",s=180,color=colores[cluster],alpha=0.5)	
	plt.scatter(kmeans.cluster_centers_[cluster][0],
		        kmeans.cluster_centers_[cluster][1],
		        marker="P",s=280,color=colores[cluster])
plt.title("clientes", fontsize=20)
plt.xlabel("saldo en cuenta de ahorros",fontsize=15)
plt.ylabel("veces que uso tarjeta de credito",fontsize=15)
plt.text(1.15, 0.2,"k=%i" %kmeans.n_clusters,fontsize=15)
plt.text(1.15, 0,"Inercia = %0.2f" %kmeans.inertia_,fontsize=15)
plt.xlim(-0.1, 1.15)
plt.ylim(-0.1, 1.15)
plt.show()

kmeans=KMeans(n_clusters=4).fit(df.values)
# kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df.values)

#calculo del número de cluster(k) "metodo del codo"
inercias=[]
for k in range(2,10):
	kmeans=KMeans(n_clusters=k).fit(df.values)
	inercias.append(kmeans.inertia_)

plt.figure(figsize=(6,5), dpi=100)
plt.scatter(range(2,10),inercias,marker="o",s=180,color="purple")
plt.xlabel("Numero de cluster",fontsize=25)
plt.ylabel("Inercia",fontsize=25)
plt.show()

df.to_csv('C:/Users/eber4/workspace/uerre/algoritmos/actividad_2.4/clientes_con_kmeans.csv')
