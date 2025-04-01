# 📱 Analizador de Apps de Google Play

Una aplicación web construida con Streamlit que permite analizar comentarios y obtener insights valiosos de cualquier aplicación de Google Play Store.

## 🚀 Características

- **Análisis de Sentimiento**: Utiliza VADER para analizar el sentimiento de los comentarios
- **Visualizaciones Avanzadas**:
  - 📊 Gráfico KDE de sentimiento vs. puntuación
  - ☁️ Nube de palabras interactiva
  - 📈 Análisis de n-gramas (unigrams, bigrams, trigrams)
  - 📉 Distribución de puntuaciones
  - 🔍 Análisis de temas (Topic Modeling)

- **Funcionalidades**:
  - Análisis en tiempo real de apps de Google Play
  - Carga de datos desde archivos CSV
  - Descarga de resultados en formato CSV
  - Interfaz intuitiva y responsive

## 🛠️ Tecnologías Utilizadas

- **Frontend & Backend**: Streamlit
- **Análisis de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn, WordCloud
- **NLP & ML**:
  - VADER Sentiment Analysis
  - NLTK para procesamiento de texto
  - Gensim para Topic Modeling
  - Scikit-learn para vectorización de texto

## 📦 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/google-play-analyzer.git
cd google-play-analyzer
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar la aplicación:
```bash
streamlit run app.py
```

## 🔧 Uso

1. Ingresa la URL de la aplicación de Google Play que deseas analizar
2. Selecciona el número de comentarios a analizar (100-10,000)
3. Explora las diferentes visualizaciones y análisis en las pestañas disponibles
4. Descarga los resultados en formato CSV si lo deseas

## 📊 Ejemplos de Análisis

- Análisis de sentimiento de los comentarios
- Identificación de temas principales en las reseñas
- Visualización de palabras más frecuentes
- Relación entre puntuación y sentimiento
- Tendencias temporales en las valoraciones

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios que te gustaría hacer.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## ✨ Autor

Tomás Palazón - [GitHub](https://github.com/TomasPalazon)
