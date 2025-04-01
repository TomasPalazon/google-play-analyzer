# Importar bibliotecas necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from google_play_scraper import app, reviews, search, Sort
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import re
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos de NLTK necesarios
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar variables de estado
if 'df' not in st.session_state:
    st.session_state.df = None
if 'app_id' not in st.session_state:
    st.session_state.app_id = None

# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis de Apps de Google Play",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para centrar contenido
st.markdown("""
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin: 0 auto;
        }
        .stTitle {
            text-align: center;
        }
        .stMarkdown {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal centrado
st.markdown("<h1 style='text-align: center;'>📱 Análisis de Apps de Google Play</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analiza comentarios y obtén insights valiosos de cualquier aplicación de Google Play</p>", unsafe_allow_html=True)

# Función para extraer app_id de la URL
def extract_app_id(url):
    pattern = r"id=([^&]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

# Función para preprocesar comentarios
def preprocess_comment(comment_text):
    if pd.isna(comment_text):
        return ""
    
    # Convertir a minúsculas y eliminar caracteres especiales
    comment_text = str(comment_text).lower()
    comment_text = re.sub(r'[^\w\s]', '', comment_text)
    
    # Tokenización simple por espacios
    tokens = comment_text.split()
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

# Función para analizar sentimiento
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

# Función para clasificar sentimiento
def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positivo'
    elif compound_score <= -0.05:
        return 'Negativo'
    else:
        return 'Neutral'

# Función para obtener los detalles de la app
def get_app_details(app_id):
    try:
        app_details = app(app_id)
        return app_details
    except Exception as e:
        st.error(f"Error al obtener detalles de la app: {str(e)}")
        return None

# Función para obtener comentarios
def get_comments(app_id, count=100):
    try:
        result, _ = reviews(
            app_id,
            lang='en',
            country='US',
            sort=Sort.NEWEST,
            count=count
        )
        return pd.DataFrame(result)
    except Exception as e:
        st.error(f"Error al obtener comentarios: {str(e)}")
        return None

# Función para obtener apps del desarrollador
def get_apps_by_developer(app_id, num_results=10):
    try:
        # Obtener detalles de la app actual
        app_details = app(app_id)
        developer = app_details['developer']
        
        # Buscar apps del mismo desarrollador
        search_result = search(
            developer,
            lang="en",
            country="US",
            n_hits=num_results
        )
        
        # Filtrar solo las apps del mismo desarrollador
        developer_apps = [
            app_info for app_info in search_result
            if app_info['developer'].lower() == developer.lower()
        ]
        
        return developer_apps
    except Exception as e:
        st.error(f"Error al obtener apps del desarrollador: {str(e)}")
        return []

# Input para la URL de la app (centrado)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    app_url = st.text_input("URL de la aplicación de Google Play")
    num_comments = st.slider("Número de comentarios a analizar", 100, 10000, 1000, 100)

    if st.button("Analizar"):
        if app_url:
            try:
                # Extraer app_id de la URL
                app_id = extract_app_id(app_url)
                st.session_state.app_id = app_id
                
                # Obtener comentarios
                df = get_comments(app_id, num_comments)
                if df is not None:
                    # Preprocesar comentarios
                    df['processed_content'] = df['content'].apply(preprocess_comment)
                    
                    # Análisis de sentimiento
                    analyzer = SentimentIntensityAnalyzer()
                    df['sentiment_scores'] = df['content'].apply(analyzer.polarity_scores)
                    df['sentiment_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])
                    df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)
                    
                    # Guardar en el estado de la sesión
                    st.session_state.df = df
                    
                    # Preparar CSV para descarga
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Descargar comentarios como CSV",
                        data=csv,
                        file_name=f"comentarios_{app_id}_{len(df)}.csv",
                        mime="text/csv"
                    )
                    st.success("Análisis completado. Puedes descargar los comentarios usando el botón de arriba.")
            except Exception as e:
                st.error(f"Error al procesar la URL o obtener comentarios: {str(e)}")

if st.session_state.df is not None:
    df = st.session_state.df
    app_id = st.session_state.app_id
    
    # Obtener detalles de la app
    with st.spinner("Obteniendo detalles de la aplicación..."):
        app_details = get_app_details(app_id)
        
    if app_details:
        # Mostrar información básica de la app
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(app_details['icon'], width=100)
        with col2:
            st.subheader(app_details['title'])
            st.write(f"Desarrollador: {app_details['developer']}")
        with col3:
            st.metric("Valoración", f"{app_details['score']:.1f} ⭐")
            st.write(f"Descargas: {app_details['installs']}")
        
        # Dashboard con pestañas
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Estadísticas Generales", 
            "☁️ Nube de Palabras",
            "📈 Análisis de Sentimiento",
            "📱 Apps del Desarrollador",
            "📋 Análisis de Temas",
            "🔍 Comentarios"
        ])

        with tab1:
            # Información básica de la app
            st.subheader("Información Detallada de la App")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Estadísticas de Comentarios**")
                total_comments = len(df)
                positive_comments = len(df[df['sentiment_score'] > 0.05])
                negative_comments = len(df[df['sentiment_score'] < -0.05])
                neutral_comments = total_comments - positive_comments - negative_comments
                
                st.metric("Total de comentarios", total_comments)
                st.metric("Comentarios positivos", f"{positive_comments} ({(positive_comments/total_comments)*100:.1f}%)")
                st.metric("Comentarios negativos", f"{negative_comments} ({(negative_comments/total_comments)*100:.1f}%)")
                st.metric("Comentarios neutros", f"{neutral_comments} ({(neutral_comments/total_comments)*100:.1f}%)")
            
            with col2:
                # Distribución de valoraciones
                st.write("**Distribución de Valoraciones**")
                fig, ax = plt.subplots(figsize=(10, 6))
                df['score'].value_counts().sort_index().plot(kind='bar')
                plt.title("Distribución de Valoraciones")
                plt.xlabel("Estrellas")
                plt.ylabel("Número de reseñas")
                st.pyplot(fig)
                plt.close()

            # Evolución temporal del sentimiento
            st.subheader("Evolución del Sentimiento")
            df['date'] = pd.to_datetime(df['at'])
            sentiment_by_date = df.groupby(df['date'].dt.date)['sentiment_score'].mean()
            
            fig = plt.figure(figsize=(12, 6))
            plt.plot(sentiment_by_date.index, sentiment_by_date.values, marker='o')
            plt.title("Evolución del Sentimiento a lo largo del tiempo")
            plt.xlabel("Fecha")
            plt.ylabel("Puntuación de Sentimiento")
            plt.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()

            # Discrepancia entre Sentimiento y Score
            st.subheader("Relación entre Sentimiento y Valoración")
            fig = plt.figure(figsize=(10, 6))
            sns.kdeplot(data=df, x='sentiment_score', y='score', cmap="Blues", fill=True)
            plt.title('Discrepancia entre el Sentimiento y la Valoración')
            plt.xlabel('Sentimiento (VADER)')
            plt.ylabel('Valoración')
            plt.grid(True)
            st.pyplot(fig)
            plt.close()

        with tab2:
            # Nube de palabras y N-gramas
            st.subheader("Análisis de Palabras")
            
            # Pestañas para diferentes análisis de texto
            word_tabs = st.tabs(["☁️ Nube de Palabras", "📊 Unigramas", "📈 Bigramas", "📉 Trigramas"])
            
            with word_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Nube de Palabras General**")
                    text = " ".join(df['processed_content'].dropna())
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.write("**Nube de Palabras por Sentimiento**")
                    sentiment_option = st.selectbox(
                        "Selecciona el tipo de sentimiento",
                        ["Positivo", "Negativo", "Neutral"],
                        key="sentiment_wordcloud"
                    )
                    
                    if sentiment_option == "Positivo":
                        mask = df['sentiment_score'] > 0.05
                    elif sentiment_option == "Negativo":
                        mask = df['sentiment_score'] < -0.05
                    else:
                        mask = (df['sentiment_score'] >= -0.05) & (df['sentiment_score'] <= 0.05)
                    
                    text = " ".join(df[mask]['processed_content'].dropna())
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                    plt.close()
            
            # Función para obtener n-gramas más comunes
            def get_top_ngrams(texts, n, top_k=20):
                vec = CountVectorizer(ngram_range=(n, n)).fit(texts)
                bag_of_words = vec.transform(texts)
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) 
                            for word, idx in vec.vocabulary_.items()]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                return words_freq[:top_k]

            # Unigramas
            with word_tabs[1]:
                st.write("**Top 20 Palabras más Frecuentes**")
                unigrams = get_top_ngrams(df['processed_content'].dropna(), 1)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                words, freqs = zip(*unigrams)
                plt.barh(range(len(words)), freqs)
                plt.yticks(range(len(words)), words)
                plt.xlabel('Frecuencia')
                plt.title('Palabras más Frecuentes')
                st.pyplot(fig)
                plt.close()

            # Bigramas
            with word_tabs[2]:
                st.write("**Top 20 Pares de Palabras más Frecuentes**")
                bigrams = get_top_ngrams(df['processed_content'].dropna(), 2)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                words, freqs = zip(*bigrams)
                plt.barh(range(len(words)), freqs)
                plt.yticks(range(len(words)), words)
                plt.xlabel('Frecuencia')
                plt.title('Pares de Palabras más Frecuentes')
                st.pyplot(fig)
                plt.close()

            # Trigramas
            with word_tabs[3]:
                st.write("**Top 20 Tríos de Palabras más Frecuentes**")
                trigrams = get_top_ngrams(df['processed_content'].dropna(), 3)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                words, freqs = zip(*trigrams)
                plt.barh(range(len(words)), freqs)
                plt.yticks(range(len(words)), words)
                plt.xlabel('Frecuencia')
                plt.title('Tríos de Palabras más Frecuentes')
                st.pyplot(fig)
                plt.close()

        with tab3:
            st.subheader("Análisis Detallado del Sentimiento")
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot de sentimiento por valoración
                st.write("**Distribución de Sentimiento por Valoración**")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x='score', y='sentiment_score')
                plt.title("Distribución de Sentimiento por Valoración")
                plt.xlabel("Valoración (estrellas)")
                plt.ylabel("Puntuación de Sentimiento")
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Mostrar comentarios más positivos y negativos
                st.write("**Comentarios Destacados**")
                most_positive = df.nlargest(3, 'sentiment_score')[['content', 'sentiment_score', 'score']]
                most_negative = df.nsmallest(3, 'sentiment_score')[['content', 'sentiment_score', 'score']]
                
                st.write("🌟 Comentarios más positivos:")
                for _, row in most_positive.iterrows():
                    st.info(f"'{row['content']}'\nSentimiento: {row['sentiment_score']:.2f} | ⭐: {row['score']}")
                
                st.write("👎 Comentarios más negativos:")
                for _, row in most_negative.iterrows():
                    st.error(f"'{row['content']}'\nSentimiento: {row['sentiment_score']:.2f} | ⭐: {row['score']}")

        with tab4:
            st.subheader("Otras Apps del Desarrollador")
            apps_info = get_apps_by_developer(app_id, num_results=10)
            
            if apps_info:
                for app in apps_info:
                    col1, col2, col3 = st.columns([1,2,1])
                    with col1:
                        st.image(app['icon'], width=100)
                    with col2:
                        st.write(f"**{app['title']}**")
                        st.write(f"Categoría: {app['genre']}")
                    with col3:
                        st.metric("Valoración", f"{app['score']:.1f} ⭐")
                        st.write(f"Descargas: {app['installs']}")
                    st.markdown("---")

        with tab5:
            st.subheader("Análisis de Temas")
            
            # Configuración del modelo LDA
            num_topics = st.slider("Número de temas a identificar", 2, 10, 4)
            
            if st.button("Analizar temas"):
                with st.spinner("Analizando temas en los comentarios..."):
                    # Preparar documentos para LDA
                    docs = df['processed_content'].dropna().tolist()
                    
                    # Crear diccionario y corpus
                    texts = [doc.split() for doc in docs]
                    dictionary = corpora.Dictionary(texts)
                    corpus = [dictionary.doc2bow(text) for text in texts]
                    
                    # Entrenar modelo LDA
                    lda_model = LdaModel(
                        corpus=corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        random_state=42,
                        passes=15
                    )
                    
                    # Mostrar temas
                    for idx, topic in lda_model.print_topics(-1):
                        st.write(f"**Tema {idx + 1}:**")
                        # Procesar y mostrar palabras más relevantes
                        words = topic.split('+')
                        for word in words:
                            weight = float(word.split('*')[0])
                            term = word.split('*')[1].strip().replace('"', '')
                            st.write(f"- {term}: {weight:.3f}")
                        st.markdown("---")

        with tab6:
            # Tabla de comentarios
            st.subheader("Comentarios y Análisis")
            
            # Filtros
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_filter = st.selectbox(
                    "Filtrar por sentimiento",
                    ["Todos", "Positivos", "Negativos", "Neutros"]
                )
            with col2:
                min_rating = st.selectbox("Valoración mínima", [1,2,3,4,5])
            with col3:
                sort_by = st.selectbox(
                    "Ordenar por",
                    ["Más recientes", "Mejor valorados", "Peor valorados", "Más positivos", "Más negativos"]
                )
            
            # Aplicar filtros
            filtered_df = df.copy()
            if sentiment_filter == "Positivos":
                filtered_df = filtered_df[filtered_df['sentiment_score'] > 0.05]
            elif sentiment_filter == "Negativos":
                filtered_df = filtered_df[filtered_df['sentiment_score'] < -0.05]
            elif sentiment_filter == "Neutros":
                filtered_df = filtered_df[(filtered_df['sentiment_score'] >= -0.05) & (filtered_df['sentiment_score'] <= 0.05)]
            
            filtered_df = filtered_df[filtered_df['score'] >= min_rating]
            
            # Ordenar
            if sort_by == "Más recientes":
                filtered_df = filtered_df.sort_values('at', ascending=False)
            elif sort_by == "Mejor valorados":
                filtered_df = filtered_df.sort_values('score', ascending=False)
            elif sort_by == "Peor valorados":
                filtered_df = filtered_df.sort_values('score', ascending=True)
            elif sort_by == "Más positivos":
                filtered_df = filtered_df.sort_values('sentiment_score', ascending=False)
            elif sort_by == "Más negativos":
                filtered_df = filtered_df.sort_values('sentiment_score', ascending=True)
            
            # Mostrar tabla
            st.dataframe(
                filtered_df[['content', 'score', 'sentiment', 'sentiment_score', 'at']].rename(columns={
                    'content': 'Comentario',
                    'score': 'Valoración',
                    'sentiment': 'Sentimiento',
                    'sentiment_score': 'Puntuación Sentimiento',
                    'at': 'Fecha'
                }),
                height=400
            )
