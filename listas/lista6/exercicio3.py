#train_split é o vetor de entrada, com 1000 linhas e 4 colunas (idade, peso, carga final e vo2max)
#o que queremos descobrir aqui é a idade, em qual faixa de idade está incluída a entrada xi- logo, ela será o y
from sklearn.preprocessing import normalize

def multivariate_gaussiana(matrix_pontos, mean, covariance_matrix, dimension):
	res = []
	inverse_covariance = covariance_matrix.I
	det_covariance = np.sqrt(np.linalg.det(covariance_matrix))
	for point in matrix_pontos.H:
		value = (point - mean.H)*covariance_matrix.I*(point - mean.H).H
		value_exp = np.exp((-1/2)*value)
		dem = (1/(np.power((2*math.pi), dimension/2)))*(1/det_covariance)
		res.append(value_exp/dem)
	return res


#QUESTAO 3.1
id_1 = []
id_2 = []
id_3 = []

for point in train_split:
	if int(point[0]) >= 18 and int(point[0]) < 40: id_1.append(point)
	elif int(point[0]) >= 40 and int(point[0]) < 60: id_2.append(point)
	elif int(point[0]) >= 60: id_3.append(point)


all_res = []
all_ids = [id_1, id_2, id_3]
all_pred = []

for dataset in all_ids:
	f_x_1 = []
	for point in dataset:
		f_x_1.append([float(point[0]), float(point[1]), float(point[2])])

	f_x_1 = np.asmatrix(normalize(f_x_1))
	f_x_1 = f_x_1.H

	mean_idade = np.mean(f_x_1[0])
	mean_peso = np.mean(f_x_1[1])
	mean_carga= np.mean(f_x_1[2])


	mean = [[mean_idade], [mean_peso], [mean_carga]]
	mean = np.asmatrix(mean)

	covariance_matrix = np.cov(f_x_1)
	covariance_matrix = np.asmatrix(covariance_matrix)

	all_res.append({'media': mean, 'covariancia': covariance_matrix})

	inverse_covariance = covariance_matrix.I
	det_covariance = np.sqrt(np.linalg.det(covariance_matrix))
	dimension = 3



	res = []
	for point in f_x_1.H:
		value = (point - mean.H)*covariance_matrix.I*(point - mean.H).H
		value_exp = np.exp((-1/2)*value)
		dem = (1/(np.power((2*math.pi), dimension/2)))*(1/det_covariance)
		res.append(value_exp/dem)
	all_pred.append(res)



#QUESTAO 3.2
# Para construir o naive bayes utilizando xi, iremos utilizar as médias e a covariância
# encontradas para a faixa de idade c. Iremos ter uma gaussiana multivariada.
# Para encontrar a probabilidade dado a faixa de idade c e a feature j, caimos
# no caso da gaussiana univariada. Iremos utilizar o mi_cj (média da faixa c na feature j)
# e o sigma**2_cj (a variância da faixa c na feature j). Na matriz de covariância, esses
# valores da variância estão na diagonal

#QUESTAO 3.3
# O Naive Bayes assume que as features são condicionalmente independentes dada a classe.
# Esta suposição não é feita no modelo Gaussiano

#QUESTAO 3.4
# Fazer um MLE aí. Aplicar o log, derivar e igualar a zero.
# Como fazer: https://www.cs.toronto.edu/~urtasun/courses/CSC411/tutorial4.pdf
# Ao final, a probabilidade de ser da classe j é exemplos da classe j/total de exemplos
# Média da feature i: x da feature i e da classe j/exemplos da classe j
# covariância da feature i: x da feature i e da classe j menos a média da feature i
# ao quadrado/exemplos da classe j

#QUESTAO 3.5
train_id1 = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in id_1]
train_id2 = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in id_2]
train_id3 = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in id_3]


train_id1 = np.asmatrix(normalize(train_id1))
train_id2 = np.asmatrix(normalize(train_id2))
train_id3 = np.asmatrix(normalize(train_id3))

#probabilidade de ser da classe j
pi_id1 = len(train_id1)/(len(train_id1)+len(train_id2)+len(train_id3))
pi_id2 = len(train_id2)/(len(train_id1)+len(train_id2)+len(train_id3))
pi_id3 = len(train_id3)/(len(train_id1)+len(train_id2)+len(train_id3))


#média de cada feature para cada classe
# cada id_1:
means_id1 = [np.mean(train_id1.H[1]), np.mean(train_id1.H[2]), np.mean(train_id1.H[3])]
mean_peso_id1 = means_id1[0]
mean_carga_id1 = means_id1[1]
mean_vo2_id1 = means_id1[2]

# cada id_2:
means_id2 = [np.mean(train_id2.H[1]), np.mean(train_id2.H[2]), np.mean(train_id2.H[3])]
mean_peso_id2 = means_id2[0]
mean_carga_id2 = means_id2[1]
mean_vo2_id2 = means_id2[2]

# cada id_3:
means_id3 = [np.mean(train_id3.H[1]), np.mean(train_id3.H[2]), np.mean(train_id3.H[3])]
mean_peso_id3 = means_id3[0]
mean_carga_id3 = means_id3[1]
mean_vo2_id3 = means_id3[2]

#covariância de cada feature para cada classe
#retirando a coluna de idade
train_id1_t = np.asmatrix(train_id1).H[1:]
train_id2_t = np.asmatrix(train_id2).H[1:]
train_id3_t = np.asmatrix(train_id3).H[1:]

cov_id1 = np.cov(train_id1_t)
cov_id2 = np.cov(train_id2_t)
cov_id3 = np.cov(train_id3_t)

cov_id1 = np.asmatrix(cov_id1)
cov_id2 = np.asmatrix(cov_id2)
cov_id3 = np.asmatrix(cov_id3)

#naive bayes usando a gaussiana vai ser: probabilidade de ser da classe j dados os parametros*probabilidade de x dada a classe j e os parametros
means_id1 = np.asmatrix(means_id1).H
means_id2 = np.asmatrix(means_id2).H
means_id3 = np.asmatrix(means_id3).H

test = []
for point in test_split:
	test.append([float(point[1]), float(point[2]), float(point[3])])

real_value = []
for point in test_split:
	if int(point[0]) >= 18 and int(point[0]) < 40: real_value.append(1)
	elif int(point[0]) >= 40 and int(point[0]) < 60: real_value.append(2)
	elif int(point[0]) >= 60: real_value.append(3)

test = np.asmatrix(normalize(test)).H
dimension = 3
#para id_1:
res_1 = multivariate_gaussiana(test, means_id1, cov_id1, dimension)
res_id1 = []
for i in res_1:
	res_id1.append([pi_id1*i.item(0)])

#para id_2:
res_2 = multivariate_gaussiana(test, means_id2, cov_id2, dimension)
res_id2 = []
for i in res_2:
	res_id2.append([pi_id2*i.item(0)])


#para id_3:
res_3 = multivariate_gaussiana(test, means_id3, cov_id3, dimension)
res_id3 = []
for i in res_3:
	res_id3.append([pi_id3*i.item(0)])


pred_value = []
for i in range(0, len(res_id1)):
	max_value = np.max([res_id1[i], res_id2[i], res_id3[i]])
	if res_id1[i] == max_value: pred_value.append(1)
	elif res_id2[i] == max_value: pred_value.append(2)
	elif res_id3[i] == max_value: pred_value.append(3)

count_certo = 0
for i in range(0, len(pred_value)):
	if pred_value[i] == real_value[i]: count_certo +=1


#QUESTAO 3.6
# Não porque prevê o vo2max

#QUESTAO 3.7
# também não, pelo mesmo motivo acima
