import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(
     page_title="Previsão de renda",
     page_icon="https://cdn-icons-png.flaticon.com/512/5408/5408783.png",
     layout="centered",
)

# Carregando os dados
renda = pd.read_csv('./input/previsao_de_renda.csv')
renda.drop(['Unnamed: 0', 'data_ref', 'id_cliente'], inplace=True, axis=1)

# Transformação de variáveis categóricas
colunas_categoricas = renda.select_dtypes(include=['object', 'category']).columns.tolist()
renda_d = pd.get_dummies(renda, columns=colunas_categoricas, drop_first=True)
media = renda_d['tempo_emprego'].mean()
renda_d['tempo_emprego'].fillna(media, inplace=True)
renda_d.isna().sum()

# Divisão dos dados
X = renda_d.drop('renda', axis=1)
y = renda_d['renda']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1993)

# Treinando o modelo
regressao = RandomForestRegressor(max_depth=10, random_state=1993)
regressao.fit(X_train, y_train)

# Função de previsão
def dummy_transformation(dados):
    novos_dados = pd.DataFrame([dados])
    colunas_categoricas = renda_d.select_dtypes(include=['object', 'category']).columns.tolist()
    novos_dados_dummy = pd.get_dummies(novos_dados, columns=colunas_categoricas, drop_first=True)
    novos_dados_dummy = novos_dados_dummy.reindex(columns=X.columns, fill_value=0)
    previsao_renda = regressao.predict(novos_dados_dummy)
    return previsao_renda[0]

# Streamlit layout
st.title('Previsão de Renda')

# Criando as abas
tab1, tab2, tab3 = st.tabs(['Previsão de Renda', 'Análises Univariadas','Análises Bivariadas'])

# Aba 1: Previsão de Renda
with tab1:
    st.header('Informe os dados para previsão')
    sexo = st.selectbox("Sexo", ["M", "F"])
    posse_de_veiculo = st.selectbox("Possui Veículo?", ["Sim", "Não"])
    posse_de_imovel = st.selectbox("Possui Imóvel?", ["Sim", "Não"])
    qtd_filhos = st.number_input("Quantidade de Filhos", min_value=0)
    tipo_renda = st.selectbox("Tipo de Renda", ['Assalariado', 'Autônomo', 'Empresário', 'Outro'])
    educacao = st.selectbox("Educação", ['Secundário', 'Superior', 'Outro'])
    estado_civil = st.selectbox("Estado Civil", ['Solteiro', 'Casado', 'Divorciado', 'Outro'])
    tipo_residencia = st.selectbox("Tipo de Residência", ['Casa', 'Apartamento', 'Com os Pais', 'Outro'])
    idade = st.number_input("Idade", min_value=0)
    tempo_emprego = st.number_input("Tempo de Emprego (anos)", min_value=0)
    qt_pessoas_residencia = st.number_input("Quantidade de Pessoas na Residência", min_value=1)


    if st.button('Prever Renda'):
        dados = {
            'sexo': sexo,
            'posse_de_veiculo': posse_de_veiculo == 'Sim',
            'posse_de_imovel': posse_de_imovel == 'Sim',
            'qtd_filhos': qtd_filhos,
            'tipo_renda': tipo_renda,
            'educacao': educacao,
            'estado_civil': estado_civil,
            'tipo_residencia': tipo_residencia,
            'idade': idade,
            'tempo_emprego': tempo_emprego,
            'qt_pessoas_residencia': qt_pessoas_residencia
        }
        previsao = dummy_transformation(dados)
        st.success(f'Previsão de Renda: R$ {previsao:.2f}')

# Aba 2: Análises e Gráficos
with tab2:
    st.header('Análises Univariadas')

    variaveis = ['renda', 'idade', 'educacao','estado_civil','idade'] 

    # Loop para automatizar a plotagem
    for var in variaveis:
        plt.clf()  # Limpa a figura atual
        plt.figure()  
        sns.displot(renda, x=var, bins=20)  # Plota a distribuição da variável
        plt.title(f'Distribuição de {var}') 
        plt.xticks(rotation=45)
        plt.xlabel(var)  
        plt.ylabel('Frequência')  
        st.pyplot(plt)  # Mostra o gráfico

# Aba 3: Análises e Gráficos
with tab3:
    st.header('Análises e Bivariadas')
       
    #'Gráfico de dispersão entre quantidade de pessoas na residência e Quantidade de filhos'
    st.subheader('Quantidade de pessoas na residência vs Quantidade de filhos')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=renda['qt_pessoas_residencia'], y=renda['qtd_filhos'])
    st.pyplot(plt)

    #'Gráfico de dispersão entre quantidade de pessoas na residência e Renda'
    st.subheader('Quantidade de pessoas na residência vs Renda')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=renda['qt_pessoas_residencia'], y=renda['renda'])
    st.pyplot(plt)
    
    # Gráfico de dispersão entre idade e renda
    st.subheader('Idade vs Renda')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=renda['idade'], y=renda['renda'])
    st.pyplot(plt)
    
    # Tempo de emprego e renda
    st.subheader('Tempo de emprego vs Renda')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=renda['tempo_emprego'], y=renda['renda'])
    st.pyplot(plt)

    # Gráfico de correlação 
    st.subheader('Heatmap de Correlação')
    # Filtrar apenas colunas numéricas
    renda_numerica = renda.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 6))
    sns.heatmap(renda_numerica.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)
