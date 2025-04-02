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
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.preprocessing import StandardScaler

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
    """Analiza el sentimiento de un texto usando VADER con léxico actualizado"""
    scores = sentiment_analyzer.polarity_scores(text)
    return scores['compound']

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
def get_comments(app_id, count=10000):
    try:
        comments = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calcular número de iteraciones necesarias
        batch_size = 200  # Google Play Scraper obtiene comentarios en lotes
        iterations = (count + batch_size - 1) // batch_size
        
        for i in range(iterations):
            status_text.text(f'Obteniendo comentarios: {min((i+1)*batch_size, count)}/{count}')
            result, continuation_token = reviews(
                app_id,
                lang='en',
                country='US',
                sort=Sort.NEWEST,
                count=min(batch_size, count - len(comments)),
                continuation_token=None if i == 0 else continuation_token
            )
            comments.extend(result)
            progress = min((i + 1) * batch_size, count) / count
            progress_bar.progress(progress)
            
            if not continuation_token or len(comments) >= count:
                break
                
        status_text.empty()
        progress_bar.empty()
        return pd.DataFrame(comments)
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

# Función para actualizar el léxico de VADER con expresiones modernas, jerga de internet y términos específicos de gaming
def update_vader_lexicon():
    """Actualiza el léxico de VADER con expresiones modernas y de gaming"""
    new_words = {
        # Expresiones de gaming positivas
        'fun as hell': 4.0,
        'good game': 3.0,
        'great game': 4.0,
        'best game': 4.0,
        'awesome game': 4.0,
        'addictive': 3.0,
        'addicting': 3.0,
        'worth it': 3.0,
        'worth the money': 3.5,
        'worth every penny': 4.0,
        
        # Jerga moderna positiva
        'goated': 4.0,
        'fire': 3.5,
        'lit': 3.5,
        'bussin': 3.5,
        'valid': 2.5,
        'clean': 2.5,
        'based': 3.0,
        'poggers': 4.0,
        'pog': 3.5,
        'w game': 3.5,
        'big w': 3.5,
        'huge w': 4.0,
        'no cap': 2.0,
        'hits different': 3.0,
        'slaps': 3.5,
        'goes hard': 3.5,
        
        # Modificadores positivos
        'as hell': 1.5,
        'af': 1.5,
        'asf': 1.5,
        'fr': 1.0,
        'frfr': 1.5,
        'ong': 1.0,
        'ngl': 0.5,
        
        # Expresiones de gaming negativas
        'pay to win': -3.5,
        'p2w': -3.0,
        'paywall': -3.0,
        'cash grab': -3.5,
        'money grab': -3.5,
        'broken game': -3.5,
        'trash game': -3.5,
        'garbage game': -3.5,
        'waste of time': -3.5,
        'waste of money': -3.5,
        'not worth': -3.0,
        'unplayable': -3.5,
        'unbalanced': -2.5,
        'rigged': -3.0,
        
        # Jerga moderna negativa
        'mid': -2.0,
        'L game': -3.0,
        'big l': -3.0,
        'huge l': -3.5,
        'dead game': -3.0,
        'skill issue': -2.0,
        'ratio': -2.0,
        'scam': -3.5,
        
        # Expresiones sobre anuncios
        'too many ads': -3.0,
        'full of ads': -3.0,
        'ads everywhere': -3.0,
        'ad spam': -3.0,
        'ad simulator': -3.0
    }
    
    # Actualizar el léxico de VADER
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(new_words)
    return analyzer

# Crear instancia de VADER con léxico actualizado
sentiment_analyzer = update_vader_lexicon()

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
                    df['sentiment_scores'] = df['content'].apply(sentiment_analyzer.polarity_scores)
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
            try:
                score = app_details.get('score', 'N/A')
                if isinstance(score, (int, float)):
                    st.metric("Valoración", f"{score:.1f} ⭐")
                else:
                    st.metric("Valoración", "N/A")
                
                installs = app_details.get('installs', 'N/A')
                st.write(f"Descargas: {installs}")
            except Exception as e:
                st.error("Error al cargar los detalles de la app")
        
        # Dashboard con pestañas
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Estadísticas Generales", 
            "☁️ Nube de Palabras",
            "📈 Análisis de Sentimiento",
            "📱 Apps del Desarrollador",
            "📋 Análisis de Temas",
            "🔍 Comentarios",
            "📊 Métricas de Evaluación"
        ])

        with tab1:
            st.subheader("Información General de la App")
            
            # Información básica
            col1, col2 = st.columns([2,1])
            with col1:
                st.write("### Detalles Principales")
                st.write(f"**Nombre:** {app_details['title']}")
                st.write(f"**Desarrollador:** {app_details['developer']}")
                st.write(f"**Categoría:** {app_details['genre']}")
                st.write(f"**Precio:** {'Gratis' if app_details['free'] else app_details['price']}")
                st.write(f"**Versión:** {app_details['version']}")
                st.write(f"**Tamaño:** {app_details.get('size', 'No disponible')}")
                st.write(f"**Android requerido:** {app_details.get('androidVersion', 'No especificado')}")
                st.write(f"**Contenido para:** {app_details.get('contentRating', 'No especificado')}")
            
            with col2:
                st.write("### Estadísticas")
                st.metric("Valoración", f"{app_details.get('score', 'N/A')} ⭐")
                st.metric("Instalaciones", f"{app_details.get('installs', 'N/A')}")
                st.metric("Reseñas", f"{app_details.get('reviews', 'N/A')}")
                
                # Calcular tiempo desde última actualización
                if 'updated' in app_details:
                    from datetime import datetime
                    last_update = datetime.fromtimestamp(app_details['updated'])
                    days_since_update = (datetime.now() - last_update).days
                    st.metric("Última actualización", f"Hace {days_since_update} días")

            # Desglose de ratings debajo de detalles principales
            if 'histogram' in app_details and app_details['histogram']:
                st.write("### Desglose de Ratings")
                hist_data = app_details['histogram']
                total_ratings = sum(hist_data)
                
                # Crear el DataFrame con los índices correctos
                ratings_df = pd.DataFrame({
                    'Rating': range(1, 6),  # Índices de 1 a 5
                    'Estrellas': ['⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
                    'Cantidad': hist_data,  # Los datos ya vienen en orden 1 a 5
                    'Porcentaje': [count/total_ratings*100 for count in hist_data]
                })
                
                # Ordenar de 5 a 1 estrellas para la visualización
                ratings_df = ratings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
                
                st.dataframe(ratings_df.style.format({
                    'Rating': '{:.0f}',
                    'Cantidad': '{:,.0f}',
                    'Porcentaje': '{:.1f}%'
                }))
                
                # Calcular el promedio ponderado
                weighted_avg = sum(ratings_df['Rating'] * ratings_df['Cantidad']) / total_ratings
                st.metric("Rating Promedio", f"{weighted_avg:.2f} ⭐")
            
            # Información adicional
            st.write("### Información Detallada")
            tabs_info = st.tabs([" Descripción", " Historial", " Permisos", " Etiquetas", " Gráficas"])
            
            with tabs_info[0]:
                if 'description' in app_details:
                    st.markdown(app_details['description'])
                else:
                    st.write("Descripción no disponible")
            
            with tabs_info[1]:
                if 'recentChanges' in app_details:
                    st.write("#### Últimos Cambios")
                    st.markdown(app_details['recentChanges'])
                else:
                    st.write("Historial de cambios no disponible")
            
            with tabs_info[2]:
                if 'permissions' in app_details and app_details['permissions']:
                    permisos = app_details['permissions']
                    st.write(f"Total de permisos requeridos: {len(permisos)}")
                    
                    # Mostrar permisos en una tabla organizada
                    permisos_df = pd.DataFrame(permisos)
                    if not permisos_df.empty:
                        st.dataframe(permisos_df, use_container_width=True)
                else:
                    st.info("No hay información de permisos disponible para esta aplicación")

            with tabs_info[3]:
                if 'categories' in app_details and app_details['categories']:
                    st.write("#### Categorías")
                    # Mostrar como chips/tags
                    cols = st.columns(3)
                    # Asegurarse de que estamos trabajando con strings
                    categories = [str(cat) if isinstance(cat, (str, int, float)) else cat.get('name', '') 
                                for cat in app_details['categories'] if cat is not None]
                    for i, categoria in enumerate(sorted(categories)):
                        if categoria:  # Solo mostrar si no está vacío
                            cols[i % 3].markdown(f" {categoria}")
                    
                    if 'tags' in app_details and app_details['tags']:
                        st.write("#### Tags Adicionales")
                        cols = st.columns(3)
                        # Asegurarse de que estamos trabajando con strings
                        tags = [str(tag) if isinstance(tag, (str, int, float)) else tag.get('name', '')
                               for tag in app_details['tags'] if tag is not None]
                        for i, tag in enumerate(sorted(tags)):
                            if tag:  # Solo mostrar si no está vacío
                                cols[i % 3].markdown(f"#️⃣ {tag}")
                else:
                    st.write("Información de categorías no disponible")
            
            # Tab Gráficas (ya existente)
            with tabs_info[4]:
                # Distribución de Ratings
                if 'histogram' in app_details and app_details['histogram']:
                    # Crear DataFrame con los datos del desglose
                    ratings_data = {
                        'Estrellas': ['1', '2', '3', '4', '5'],  # Cambiado a texto plano
                        'Cantidad': [
                            app_details['histogram'][0],  # 1 estrella
                            app_details['histogram'][1],  # 2 estrellas
                            app_details['histogram'][2],  # 3 estrellas
                            app_details['histogram'][3],  # 4 estrellas
                            app_details['histogram'][4]   # 5 estrellas
                        ]
                    }
                    df_ratings = pd.DataFrame(ratings_data)
                    total_ratings = df_ratings['Cantidad'].sum()
                    df_ratings['Porcentaje'] = (df_ratings['Cantidad'] / total_ratings * 100).round(1)
                    
                    # Crear gráfica de barras
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(df_ratings['Estrellas'], df_ratings['Cantidad'], color='gold')
                    
                    # Añadir etiquetas con cantidad y porcentaje
                    for bar in bars:
                        height = bar.get_height()
                        percentage = (height / total_ratings * 100).round(1)
                        ax.text(bar.get_x() + bar.get_width()/2, height,
                               f'{int(height):,}\n({percentage}%)',
                               ha='center', va='bottom')
                    
                    plt.title('Distribución de Ratings')
                    plt.xlabel('Rating')
                    plt.ylabel('Número de Usuarios')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                if len(df) > 0:
                    st.write("### Análisis de Sentimiento vs Rating")
                    
                    # Calcular sentimiento si no existe
                    if 'sentiment_score' not in df.columns:
                        df['sentiment_score'] = df['content'].apply(
                            lambda x: sentiment_analyzer.polarity_scores(str(x))['compound']
                        )
                    
                    # Gráfico de densidad
                    fig = plt.figure(figsize=(10, 6))
                    sns.kdeplot(data=df, x='sentiment_score', y='score', cmap="Blues", fill=True)
                    plt.title('Discrepancia entre Sentimiento y Rating')
                    plt.xlabel('Puntuación de Sentimiento (VADER)')
                    plt.ylabel('Rating de Usuario')
                    plt.grid(True)
                    st.pyplot(fig)
                    
                    # Estadísticas de correlación
                    correlation = df['sentiment_score'].corr(df['score'])
                    st.metric("Correlación Sentimiento-Rating", f"{correlation:.3f}")
                    
                    # Tabla de discrepancias
                    st.write("### Mayores Discrepancias")
                    df['discrepancia'] = abs(df['sentiment_score'] - (df['score']/5))
                    discrepancias = df.nlargest(5, 'discrepancia')[
                        ['content', 'score', 'sentiment_score', 'discrepancia']
                    ].rename(columns={
                        'content': 'Comentario',
                        'score': 'Rating',
                        'sentiment_score': 'Sentimiento',
                        'discrepancia': 'Discrepancia'
                    })
                    st.dataframe(discrepancias.style.format({
                        'Rating': '{:.1f}',
                        'Sentimiento': '{:.3f}',
                        'Discrepancia': '{:.3f}'
                    }))
            
            # Métricas de crecimiento
            st.write("### Métricas de Crecimiento")
            growth_cols = st.columns(4)
            
            # Calcular métricas de crecimiento si es posible
            if len(df) > 0:
                with growth_cols[0]:
                    reviews_last_month = len(df[df['at'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))])
                    st.metric("Reseñas último mes", reviews_last_month)
                
                with growth_cols[1]:
                    avg_rating_recent = df[df['at'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]['score'].mean()
                    st.metric("Rating promedio reciente", f"{avg_rating_recent:.2f}")
                
                with growth_cols[2]:
                    sentiment_recent = df[df['at'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]['sentiment_score'].mean()
                    st.metric("Sentimiento reciente", f"{sentiment_recent:.2f}")
                
                with growth_cols[3]:
                    response_rate = len(df[df['repliedAt'].notna()]) / len(df) * 100
                    st.metric("Tasa de respuesta", f"{response_rate:.1f}%")

        with tab2:
            # Nube de palabras y N-gramas
            st.subheader("Análisis de Palabras")
            
            # Pestañas para diferentes análisis de texto
            word_tabs = st.tabs([" Nube de Palabras", " Unigramas", " Bigramas", " Trigramas"])
            
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
            st.subheader("Evolución del Sentimiento")
            
            # Selector de rango de fechas
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Fecha inicial",
                    min_value=df['at'].min().date(),
                    max_value=df['at'].max().date(),
                    value=df['at'].min().date()
                )
            with col2:
                end_date = st.date_input(
                    "Fecha final",
                    min_value=df['at'].min().date(),
                    max_value=df['at'].max().date(),
                    value=df['at'].max().date()
                )
            
            # Filtrar por rango de fechas
            mask = (df['at'].dt.date >= start_date) & (df['at'].dt.date <= end_date)
            df_filtered = df[mask].copy()
            
            if not df_filtered.empty:
                # Agrupar por fecha y calcular promedio de sentimiento
                sentiment_by_date = df_filtered.groupby(df_filtered['at'].dt.date)['sentiment_score'].mean().reset_index()
                
                # Calcular línea de tendencia
                X = np.arange(len(sentiment_by_date)).reshape(-1, 1)
                y = sentiment_by_date['sentiment_score'].values
                
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(X, y)
                trend_line = reg.predict(X)
                
                # Calcular R² y ecuación de la recta
                r2 = reg.score(X, y)
                slope = reg.coef_[0]
                intercept = reg.intercept_
                equation = f'y = {slope:.4f}x + {intercept:.4f}'
                
                # Crear gráfico
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Gráfico de dispersión
                ax.scatter(sentiment_by_date['at'], sentiment_by_date['sentiment_score'], 
                         alpha=0.5, color='blue', label='Sentimiento diario')
                
                # Línea de tendencia
                ax.plot(sentiment_by_date['at'], trend_line, 
                       color='red', linestyle='--', 
                       label=f'Tendencia (R² = {r2:.3f})')
                
                # Personalización del gráfico
                ax.set_xlabel('Fecha')
                ax.set_ylabel('Puntuación de Sentimiento')
                ax.grid(True, alpha=0.3)
                
                # Rotar etiquetas del eje x
                plt.xticks(rotation=45)
                
                # Añadir ecuación de la recta
                ax.text(0.05, 0.95, equation, 
                       transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.8),
                       verticalalignment='top')
                
                # Ajustar layout y mostrar
                plt.tight_layout()
                st.pyplot(fig)
                
                # Estadísticas adicionales
                st.write("### Estadísticas del período seleccionado")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentimiento promedio", f"{sentiment_by_date['sentiment_score'].mean():.2f}")
                with col2:
                    st.metric("Tendencia", "Positiva" if slope > 0 else "Negativa")
                with col3:
                    st.metric("R²", f"{r2:.3f}")
            else:
                st.warning("No hay datos para el rango de fechas seleccionado")

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
                        try:
                            score = app.get('score')
                            if score is not None:
                                st.metric("Valoración", f"{score:.1f} ⭐")
                            else:
                                st.metric("Valoración", "N/A")
                            
                            installs = app.get('installs', 'N/A')
                            st.write(f"Descargas: {installs}")
                        except Exception as e:
                            st.error("Error al mostrar valoración")
                    st.markdown("---")

        with tab5:
            st.subheader("Análisis de Temas")
            
            if len(df) > 0:
                # Definir categorías de temas comunes
                temas_categorias = {
                    'Rendimiento': [
                        'lento', 'congela', 'lag', 'crash', 'cierra', 'freezes', 'cuelga', 'pesada', 'rendimiento',
                        'fps', 'slow', 'loading', 'stuck', 'frozen', 'performance', 'sluggish', 'heavy', 'memory',
                        'ram', 'cpu', 'batería', 'battery', 'drain', 'optimization', 'optimización', 'frame rate',
                        'stuttering', 'stutters', 'laggy', 'responsive', 'unresponsive'
                    ],
                    'Publicidad': [
                        'anuncios', 'ads', 'publicidad', 'propaganda', 'pop-up', 'popup', 'advertisement',
                        'advertising', 'commercial', 'spam', 'sponsored', 'promoción', 'banner', 'invasive',
                        'invasiva', 'molesta', 'annoying', 'intrusive', 'ad-free', 'sin anuncios'
                    ],
                    'Interfaz': [
                        'interfaz', 'diseño', 'ui', 'ux', 'usuario', 'menú', 'botones', 'navegación',
                        'layout', 'design', 'user interface', 'user experience', 'usability', 'usabilidad',
                        'intuitivo', 'intuitive', 'clean', 'limpio', 'modern', 'moderno', 'aesthetic',
                        'estética', 'visual', 'responsive', 'theme', 'tema', 'dark mode', 'modo oscuro',
                        'accesible', 'accessible', 'user-friendly'
                    ],
                    'Funcionalidad': [
                        'funciona', 'feature', 'característica', 'opción', 'herramienta', 'functionality',
                        'function', 'tool', 'works', 'working', 'feature request', 'capability', 'capacidad',
                        'customization', 'personalización', 'settings', 'configuración', 'ajustes', 'options',
                        'powerful', 'potente', 'flexible', 'versatile', 'versátil', 'robust', 'robusto'
                    ],
                    'Errores': [
                        'error', 'bug', 'fallo', 'problema', 'no funciona', 'arreglen', 'issue', 'glitch',
                        'broken', 'roto', 'fix', 'arreglo', 'solution', 'solución', 'patch', 'parche',
                        'debug', 'debugging', 'defecto', 'defect', 'mal funcionamiento', 'malfunction',
                        'technical issue', 'problema técnico', 'not working'
                    ],
                    'Actualizaciones': [
                        'actualización', 'update', 'versión', 'nueva versión', 'upgrade', 'patch',
                        'release', 'latest', 'newest', 'recent', 'changelog', 'cambios', 'improvements',
                        'mejoras', 'features', 'novedades', 'rollback', 'downgrade', 'rollout', 'beta'
                    ],
                    'Contenido': [
                        'contenido', 'información', 'datos', 'material', 'content', 'info', 'data',
                        'resources', 'recursos', 'quality', 'calidad', 'quantity', 'cantidad', 'variety',
                        'variedad', 'media', 'multimedia', 'files', 'archivos', 'documents', 'documentos',
                        'library', 'biblioteca', 'collection', 'colección'
                    ],
                    'Precio': [
                        'precio', 'pago', 'gratis', 'premium', 'compra', 'costo', 'price', 'payment',
                        'free', 'paid', 'subscription', 'suscripción', 'monthly', 'mensual', 'yearly',
                        'anual', 'trial', 'prueba', 'value', 'valor', 'worth', 'vale la pena', 'expensive',
                        'caro', 'cheap', 'barato', 'affordable', 'asequible'
                    ],
                    'Soporte': [
                        'soporte', 'ayuda', 'atención', 'respuesta', 'servicio', 'support', 'help',
                        'assistance', 'customer service', 'servicio al cliente', 'feedback', 'response',
                        'contact', 'contacto', 'ticket', 'chat', 'email', 'correo', 'communication',
                        'comunicación', 'responsive', 'quick', 'rápido', 'slow', 'lento'
                    ],
                    'Permisos': [
                        'permiso', 'acceso', 'privacidad', 'datos personales', 'permission', 'access',
                        'privacy', 'personal data', 'security', 'seguridad', 'tracking', 'seguimiento',
                        'collection', 'recolección', 'consent', 'consentimiento', 'gdpr', 'policy',
                        'política', 'terms', 'términos', 'conditions', 'condiciones'
                    ]
                }
                
                # Función para clasificar comentarios
                def clasificar_comentario(texto):
                    texto = texto.lower()
                    categorias = []
                    for categoria, keywords in temas_categorias.items():
                        if any(keyword in texto for keyword in keywords):
                            categorias.append(categoria)
                    return categorias if categorias else ['Otros']
                
                # Clasificar comentarios
                df['temas'] = df['content'].apply(clasificar_comentario)
                
                # Expandir la lista de temas
                todos_temas = []
                for temas in df['temas']:
                    todos_temas.extend(temas)
                
                # Contar frecuencia de temas
                tema_counts = pd.Series(todos_temas).value_counts()
                
                # Filtrar la categoría "Otros" para la visualización
                tema_counts_sin_otros = tema_counts[tema_counts.index != "Otros"]
                
                # Visualizar distribución de temas (excluyendo "Otros")
                if not tema_counts_sin_otros.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    tema_counts_sin_otros.plot(kind='bar')
                    plt.title('Distribución de Temas en Comentarios (excluyendo Otros)')
                    plt.xlabel('Tema')
                    plt.ylabel('Número de Comentarios')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Mostrar estadísticas completas incluyendo "Otros"
                st.write("### Distribución completa de temas:")
                total_comentarios = tema_counts.sum()
                for tema, count in tema_counts.items():
                    percentage = (count / total_comentarios) * 100
                    st.write(f"- {tema}: {count} comentarios ({percentage:.1f}%)")
                
                # Mostrar ejemplos de cada tema
                st.write("### Ejemplos de Comentarios por Tema")
                for tema in tema_counts.index:
                    with st.expander(f"{tema} ({tema_counts[tema]} comentarios)"):
                        # Obtener comentarios de este tema
                        comentarios_tema = df[df['temas'].apply(lambda x: tema in x)]
                        # Mostrar los 3 comentarios más relevantes (basados en likes o longitud)
                        if 'thumbsUpCount' in comentarios_tema.columns:
                            ejemplos = comentarios_tema.nlargest(3, 'thumbsUpCount')
                        else:
                            ejemplos = comentarios_tema.sample(min(3, len(comentarios_tema)))
                        
                        for _, comentario in ejemplos.iterrows():
                            st.write(f"- {comentario['content']}")
                            if 'score' in comentario:
                                st.write(f"   Rating: {'⭐' * int(comentario['score'])}")
                
                # Análisis de sentimiento por tema
                st.write("### Sentimiento por Tema")
                sentimiento_tema = {}
                for tema in tema_counts.index:
                    comentarios_tema = df[df['temas'].apply(lambda x: tema in x)]
                    sentimiento_tema[tema] = comentarios_tema['sentiment_score'].mean()
                
                # Visualizar sentimiento por tema
                sentimiento_df = pd.DataFrame.from_dict(sentimiento_tema, orient='index', columns=['Sentimiento'])
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = ['red' if x < 0 else 'green' for x in sentimiento_df['Sentimiento']]
                sentimiento_df['Sentimiento'].plot(kind='bar', color=colors)
                plt.title('Sentimiento Promedio por Tema')
                plt.xlabel('Tema')
                plt.ylabel('Sentimiento (-1 a 1)')
                plt.xticks(rotation=45)
                st.pyplot(fig)

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
            
            # Mostrar tabla con formato mejorado
            st.dataframe(
                filtered_df[['content', 'score', 'sentiment', 'sentiment_score', 'at']].rename(columns={
                    'content': 'Comentario',
                    'score': 'Valoración',
                    'sentiment': 'Sentimiento',
                    'sentiment_score': 'Puntuación Sentimiento',
                    'at': 'Fecha'
                }).style.format({
                    'Puntuación Sentimiento': '{:.2f}',
                    'Fecha': lambda x: x.strftime('%Y-%m-%d')
                }),
                height=400
            )

        with tab7:
            st.subheader("Métricas de Evaluación")
            
            if len(df) > 0:
                # Preparar datos para el modelo
                # Crear más características para el modelo
                df['text_length'] = df['content'].str.len()
                df['word_count'] = df['content'].str.split().str.len()
                df['sentiment_class'] = df['sentiment'].map({'Negativo': 0, 'Neutral': 1, 'Positivo': 2})
                
                features = [
                    'sentiment_score',  # Puntuación de sentimiento VADER
                    'text_length',      # Longitud del texto
                    'word_count'        # Número de palabras
                ]
                
                if 'score' in df.columns:
                    df['score_norm'] = df['score'] / 5.0  # Normalizar puntuación del usuario
                    features.append('score_norm')
                
                if 'thumbsUpCount' in df.columns:
                    df['likes_norm'] = np.log1p(df['thumbsUpCount'])  # Normalizar likes con log
                    features.append('likes_norm')
                
                X = df[features]
                y = df['sentiment_class']
                
                # Asegurarse de que no hay valores nulos
                X = X.fillna(0)
                
                # Normalizar características
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Entrenar modelos
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Modelo SVM")
                    
                    # Entrenar SVM con mejores parámetros
                    svm_model = svm.SVC(
                        kernel='rbf',
                        C=1.0,
                        gamma='scale',
                        probability=True,
                        class_weight='balanced',
                        random_state=42
                    )
                    svm_model.fit(X_train, y_train)
                    
                    # Predicciones
                    y_pred_svm = svm_model.predict(X_test)
                    
                    # Métricas
                    report = classification_report(y_test, y_pred_svm, output_dict=True)
                    
                    # Mostrar métricas en formato de tabla
                    metrics_df = pd.DataFrame({
                        'Precisión': [report[str(i)]['precision'] for i in range(3)],
                        'Recall': [report[str(i)]['recall'] for i in range(3)],
                        'F1-Score': [report[str(i)]['f1-score'] for i in range(3)]
                    }, index=['Negativo', 'Neutral', 'Positivo'])
                    
                    st.dataframe(metrics_df.style.format("{:.3f}"))
                    
                    # Matriz de confusión
                    cm = confusion_matrix(y_test, y_pred_svm)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                              xticklabels=['Negativo', 'Neutral', 'Positivo'],
                              yticklabels=['Negativo', 'Neutral', 'Positivo'])
                    plt.title('Matriz de Confusión - SVM')
                    plt.xlabel('Predicción')
                    plt.ylabel('Real')
                    st.pyplot(fig)

                with col2:
                    st.write("### Modelo Random Forest")
                    
                    # Entrenar Random Forest con mejores parámetros
                    rf_model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight='balanced',
                        random_state=42
                    )
                    rf_model.fit(X_train, y_train)
                    
                    # Predicciones
                    y_pred_rf = rf_model.predict(X_test)
                    
                    # Métricas
                    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
                    
                    # Mostrar métricas en formato de tabla
                    metrics_df_rf = pd.DataFrame({
                        'Precisión': [report_rf[str(i)]['precision'] for i in range(3)],
                        'Recall': [report_rf[str(i)]['recall'] for i in range(3)],
                        'F1-Score': [report_rf[str(i)]['f1-score'] for i in range(3)]
                    }, index=['Negativo', 'Neutral', 'Positivo'])
                    
                    st.dataframe(metrics_df_rf.style.format("{:.3f}"))
                    
                    # Importancia de características
                    feature_importance = pd.DataFrame({
                        'Característica': features,
                        'Importancia': rf_model.feature_importances_
                    }).sort_values('Importancia', ascending=False)

                    st.write("### Importancia de Características")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(data=feature_importance, x='Importancia', y='Característica')
                    plt.title('Importancia de Características en la Predicción')
                    st.pyplot(fig)

                    # Explicar las características más importantes
                    st.write("### Interpretación del Modelo")
                    st.write("""
                    El modelo toma en cuenta varias características para hacer sus predicciones:
                    
                    1. **sentiment_score**: La puntuación de sentimiento de VADER (-1 a 1)
                    2. **score_norm**: La puntuación que dio el usuario (0-5, normalizada)
                    3. **text_length**: Longitud del comentario
                    4. **word_count**: Número de palabras
                    5. **likes_norm**: Número de likes (normalizado)
                    
                    El modelo aprende patrones complejos como:
                    - Comentarios largos con sentiment_score negativo y pocas estrellas suelen ser quejas detalladas
                    - Comentarios cortos con muchos likes y sentiment_score positivo suelen ser recomendaciones entusiastas
                    - Comentarios de longitud media con puntuación neutral podrían ser sugerencias o feedback constructivo
                    """)
                
                # Análisis de casos contradictorios
                st.write("### Análisis de Casos Contradictorios")
                
                # Encontrar comentarios con puntuación alta pero sentimiento negativo
                contradictorios = df[
                    (df['score'] >= 4) & (df['sentiment_score'] < -0.2)
                ].copy()  # Usar copy() del DataFrame completo
                
                if len(contradictorios) > 0:
                    st.write("#### Comentarios con puntuación alta (4-5) pero sentimiento negativo:")
                    
                    # Asegurarnos de que tenemos todas las características necesarias
                    X_contra = contradictorios[features].copy()
                    
                    # Normalizar las características
                    X_contra_scaled = scaler.transform(X_contra)
                    
                    # Hacer predicciones
                    contradictorios['Predicción SVM'] = svm_model.predict(X_contra_scaled)
                    contradictorios['Predicción RF'] = rf_model.predict(X_contra_scaled)
                    
                    # Mapear predicciones numéricas a texto
                    sentiment_map = {0: 'Negativo', 1: 'Neutral', 2: 'Positivo'}
                    contradictorios['Predicción SVM'] = contradictorios['Predicción SVM'].map(sentiment_map)
                    contradictorios['Predicción RF'] = contradictorios['Predicción RF'].map(sentiment_map)
                    
                    # Mostrar ejemplos
                    for idx, row in contradictorios.head(5).iterrows():
                        with st.expander(f"Comentario (Score: {row['score']}, Sentiment: {row['sentiment_score']:.2f})"):
                            st.write(f"**Texto**: {row['content']}")
                            st.write(f"**Score del usuario**: {row['score']}/5")
                            st.write(f"**Sentiment score**: {row['sentiment_score']:.2f}")
                            st.write(f"**Longitud del texto**: {row['text_length']} caracteres")
                            st.write(f"**Número de palabras**: {row['word_count']} palabras")
                            if 'thumbsUpCount' in row:
                                st.write(f"**Likes**: {row['thumbsUpCount']}")
                            st.write(f"**Predicción SVM**: {row['Predicción SVM']}")
                            st.write(f"**Predicción RF**: {row['Predicción RF']}")
                else:
                    st.write("No se encontraron casos contradictorios en este conjunto de datos.")
                
                # Métricas adicionales
                st.write("### Métricas Adicionales")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Cross-validation score para SVM
                    cv_scores_svm = cross_val_score(svm_model, X, y, cv=5)
                    st.metric("CV Score (SVM)", f"{cv_scores_svm.mean():.3f} ± {cv_scores_svm.std():.3f}")
                
                with col2:
                    # Cross-validation score para RF
                    cv_scores_rf = cross_val_score(rf_model, X, y, cv=5)
                    st.metric("CV Score (RF)", f"{cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")
                
                with col3:
                    # Comparación de accuracy
                    acc_svm = accuracy_score(y_test, y_pred_svm)
                    acc_rf = accuracy_score(y_test, y_pred_rf)
                    st.metric("Mejor Modelo", 
                             "SVM" if acc_svm > acc_rf else "Random Forest",
                             f"Accuracy: {max(acc_svm, acc_rf):.3f}")
