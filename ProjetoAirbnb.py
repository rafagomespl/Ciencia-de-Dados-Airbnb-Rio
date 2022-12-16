#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio
# 
# ### Ferramenta de Previsão de Preço de Imóvel

# 
# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Objetivo
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Caso queira uma outra solução, podemos olhar como referência a solução do usuário Allan Bruno do kaggle no Notebook: https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
# 
# 

# #### Importar bibliotecas e bases de dados

# In[2]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[3]:


meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = pathlib.Path('dataset')

base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    mes = meses[arquivo.name[:3]]
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)
    
display(base_airbnb)


# - Como temos muitas colunas, o modelo acaba ficando muito lento.
# - Além disso, uma análise rápida permite ver que várias colunas não são necessárias para o modelo de previsão, por este motivo, algumas colunas serão excluídas da base de dados.
# - Tipos de colunas que serão excluídas:
#     1. IDs, Links, informações não relevantes para o modelo. 
#     2. Colunas repetidas ou extremamente parecidas com outra (que possuem a mesma informação)
#     3. Colunas preenchidas com texto livre -> 
#     4. Colunas em que todos ou quase todos os valores são iguais
# - Para isso, vamos criar um arquivo em excel com os 1.000 primeiros registros e fazer uma análise qualitativa.

# In[4]:


print(list(base_airbnb.columns))
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')


# In[5]:


print(base_airbnb['experiences_offered'].value_counts())


# In[6]:


print((base_airbnb['host_listings_count']==base_airbnb['host_total_listings_count']).value_counts())


# In[7]:


print(base_airbnb['square_feet'].isnull().sum())


# #### Depois da análise qualitativa das colunas, levando em conta os critérios acima, ficamos com as seguintes colunas:

# In[8]:


colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
display(base_airbnb)


# #### Tratar valores faltando
# 
# - Visualizando os dados, percebe-se que existe uma grande disparidade em dados faltantes. as colunas com mais de 300.000 linhas com valores NaN foram retiradas da análise.
# - Para as outras colunas, como a base tem muitos dados (mais de 900.000) os dados com valores NaN terão sua linha excluída.

# In[9]:


print(base_airbnb.isnull().sum())

for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
print(base_airbnb.isnull().sum())


# In[10]:


base_airbnb = base_airbnb.dropna()

print(base_airbnb.shape)
print(base_airbnb.isnull().sum())


# #### Verificar os tipos de dados em cada coluna

# In[11]:


print(base_airbnb.dtypes)
print('-'*80)
print(base_airbnb.iloc[0])


# - Como o preço e pessoa extra estão sendo reconhecidos como objeto (ao invés de ser float) será necessário mudar o tipo de variável da coluna.

# In[12]:


#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '').str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)
#extra-people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '').str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)
#verificando os tipos
print(base_airbnb.dtypes)


# In[13]:


# Tornando o código mais leve mudando o tipo da variável de float64 para float32, o mesmo com o int
for coluna in base_airbnb:
    if base_airbnb[coluna].dtype == 'float64':
        base_airbnb[coluna] = base_airbnb[coluna].astype(np.float32, copy=False)
    if base_airbnb[coluna].dtype == 'int64':
        base_airbnb[coluna] = base_airbnb[coluna].astype(np.int32, copy=False)
print(base_airbnb.dtypes)


# #### Análise Exploratória e Tratar Outliers
# - Verificar a Correlação entre as features (característica);
# 
# - O outlier será qualquer valor que estiver abaixo do limite inferior ou acima do limite superior.
#     - Limite inferior será o 1º Quartil - 1,5 * Amplitude;
#     - Limite superior será o 3º Quartil + 1,5 * Amplitude;
#     - Amplitude = 3º Quartil - 1º Quartil;
# - Identificar se o outlier faz sentido excluir ou não.

# In[14]:


plt.figure(figsize=(18,13))
sns.heatmap(base_airbnb.corr(), annot=True, cmap='Blues')


# #### Definição de Funções para Análise de Outliers

# In[15]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5*amplitude, q3 + 1.5*amplitude

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup),:]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


# In[16]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15,5))
    sns.distplot(coluna, hist=True)
    
def grafico_barra(coluna):
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# #### Price

# In[17]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imóveis comuns, acredito que os valores acima do limite superior serão apenas os apartamentos de altíssimo luxo, que não é o objetivo principal, por este motivo, serão excluídos estes outliers.

# In[18]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print(f'{linhas_removidas} linhas removidas.')


# In[19]:


histograma(base_airbnb['price'])


# #### Extra_people

# In[20]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[21]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print(f'{linhas_removidas} linhas removidas.')


# In[22]:


histograma(base_airbnb['extra_people'])


# #### Host_listings_counts

# In[23]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# Podemos excluir os outliers, porque para o objetivo do projeto, hosts com mais de 6 imóveis no airbnb não é o público alvo do projeto (imagino que sejam imobiliários ou profissionais que gerenciam imóveis no aibnb).

# In[24]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print(f'{linhas_removidas} linhas removidas.')


# #### Accommodates

# In[25]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# In[26]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print(f'{linhas_removidas} linhas removidas.')


# #### Bathrooms

# In[27]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())


# In[28]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print(f'{linhas_removidas} linhas removidas.')


# #### Bedrooms

# In[29]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# In[30]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print(f'{linhas_removidas} linhas removidas.')


# #### Beds

# In[31]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# In[32]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print(f'{linhas_removidas} linhas removidas.')


# #### Guests_included

# In[33]:


diagrama_caixa(base_airbnb['guests_included'])
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())


# Esta feature será removida da análise. Parece que os usuários do airbnb usam muito o valor padrão do airbnb como 1 guest included. Isso pode levar o modelo a considerar uma feature que na verdade não é essencial para a definição do preço, por este motivo, parece razoável excluir a coluna da análise.

# In[34]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
print(base_airbnb.shape)


# #### Minimum_nights

# In[35]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])


# In[36]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print(f'{linhas_removidas} linhas removidas.')


# #### Maximum_nights

# In[37]:


diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])


# In[38]:


base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
print(base_airbnb.shape)


# #### Number_of_reviews

# In[39]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# In[40]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
print(base_airbnb.shape)


# ### Tratamento de Colunas de Valores de Texto

# #### Property_type

# In[41]:


plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# In[42]:


# Facilitando a visualização, os espaços com valores menores que 2000 serão agrupados na categoria outros.
tabela_tipos_espaco = base_airbnb['property_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_tipos_espaco.index:
    if tabela_tipos_espaco[tipo] < 2000:
        colunas_agrupar.append(tipo)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Other'

print(base_airbnb['property_type'].value_counts())


# #### room_type

# In[43]:


plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

print(base_airbnb['room_type'].value_counts())


# #### bed_type

# In[44]:


plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

print(base_airbnb['bed_type'].value_counts())


# Agrupando colunas de bed_type

# In[45]:


tabela_bed_type = base_airbnb['bed_type'].value_counts()
agrupar = []

for tipo in tabela_bed_type.index:
    if tabela_bed_type[tipo] < 10000:
        agrupar.append(tipo)

for tipo in agrupar:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Other'

print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# #### Cancellation_policy

# In[46]:


plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

print(base_airbnb['cancellation_policy'].value_counts())


# Como a quantidade de 'strict', 'super_strict_60' e 'super_strict_30' são bem inferiores aos demais, estas três colunas serão agrupadas como 'strict'.

# In[47]:


tabela_tipos_cancellation = base_airbnb['cancellation_policy'].value_counts()
agrupar = []

for tipo in tabela_tipos_cancellation.index:
    if tabela_tipos_cancellation[tipo] < 10000:
        agrupar.append(tipo)

for tipo in agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# #### Amenities
# 
# Como temos uma diversidade muito grande de amenities e, às vezes, as mesmas amenities escritas de formas diferentes, vamos avaliar a quantidade de amenities como o parâmetro para o nosso modelo.

# In[48]:


base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape


# In[49]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# Como o objetivo é um local comum, que não tenha tantas coisas, ou que o host colocou coisas redundantes os outliers serão excluidos da coluna de amenities

# In[50]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print(f'{linhas_removidas} linhas removidas.')


# Visualizando as propriedades no mapa.

# In[51]:


amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()}

mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5,
                        center = centro_mapa, zoom=10, mapbox_style='stamen-terrain')
mapa.show()


# #### Encoding
# 
# Ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true, false, etc.)
# 
# - Features de valores True ou False serão substituídos por 1 e 0, respectivamente;
# - Features de categoria serão substituídos por variáveis Dummies utilizando o método encoding.

# In[52]:


colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t' ,coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f' ,coluna] = 0


# In[53]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias)
display(base_airbnb_cod.head())


# ### Modelo de Previsão

# Métricas de Avaliação
# 
# - R²
# - Erro quadrático médio (RSME)

# In[54]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}: \nR2: {r2:.2%}\nRSME: {RSME:.2f}'


# Modelos de previsão que serão utilizados:
# 
# - Linear Regression
# - Random Forest Regressor
# - Extra Trees

# In[55]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)


# Separando os dados em treino (80%) e teste (20%)
# 
# Treino do modelo

# In[56]:


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# In[57]:


for nome_modelo, modelo in modelos.items():
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# O modelo escolhido como melhor modelo foi o ExtraTreesRegressor
# 
# - Este modelo obteve o maior valor de R² e ao mesmo tempo o menor valor de RSME;
# - Não houve uma grande diferença na velocidade de treino e de previsão desse modelo em comparação com o modelo de RandomForest;
# - Embora o modelo LinearRegression conseguir resolver em um tempo bem menor em comparação com os outros modelos, não obteve valores satisfatórios de R² e RSME.

# #### Ajustes e melhorias no modelo ExtraTreesRegressor

# In[58]:


importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)


# ### Análise sobre o gráfico
# - A feature de quantidade de quartos foi bem relevante no modelo e faz sentido, pois, um imóvel com mais quartos acomoda mais pessoas, é maior, tem uma valorização maior.
# - Como já visto no mapa e confirmado aqui no gráfico, a localização é um ponto bem relevante no preço. Imóveis localizados mais próximo as praias tem uma valorização maior por ser uma comodidade aos turistas que procuram conhecer as praias, e são áreas bem valorizadas, com bastante comércio etc.
# - Outro ponto interessante é a questão do 'n_amenities' que trata da quantidade de itens que são encontrados no imóvel e que podem refletir em dois pontos:
#     1. Retrata realmente que o imóvel possui bastante coisa como: tv, ar condicionado, chuveiro elétrico, e isso valoriza o imóvel;
#     2. Mostra que os hosts que dão mais relevencia a este ponto, colocando de forma detalhada os itens geralmente são os que conseguem atrair mais clientes e precificar melhor seus imóveis, poderíamos pensar que um host mais recente no aplicativo não coloque tão detalhado por não saber a relevância disto na busca dos clientes.
# - Podemos analisar também que algumas features não tem um impacto significativo no modelo, como no caso do 'is_business_travel_ready' e que precisa ser avaliado mais a fundo se é realmente necessária no modelo, treinando-o novamente sem esta feature.

# #### Ajustes finais no modelo
# - Avaliando o impacto da feature 'is_business_travel_ready' da base de dados e avaliando novamente o modelo.

# In[59]:


base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# In[60]:


base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# Feito estes ajustes, conseguimos um modelo mais simplificado e com valores de métricas ainda melhores que os valores do modelo inicial. Poderia ainda ser feito uma análise ainda mais afundo com outras features, retirando, agrupando, mas, para este caso (apenas estudo) o modelo está sufiecientemente bom.
