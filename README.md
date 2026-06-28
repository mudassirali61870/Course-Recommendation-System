# Course-Recommendation-System

# Udemy Course Recommendation System

An AI-powered content-based recommendation system that suggests relevant Udemy courses based on text descriptions and course titles. The project utilizes Natural Language Processing (NLP) text preprocessing techniques alongside TF-IDF and Count Vectorization techniques to compute vector similarities.

## 🚀 Features
* **Text Clean-up & Normalization:** Employs `neattext` to seamlessly clear out stopwords and special characters from titles for optimal vector space representation.
* **Vector Embeddings:** Extracts structural features from textual metadata using `CountVectorizer` and `TfidfVectorizer`.
* **Similarity Engines:** Employs Cosine Similarity metrics (`cosine_similarity` and `linear_kernel`) to match and score similarities between user text-queries or chosen items against the course directory.
* **Clean Dataset Structure:** Preprocessed course parameters including specific mappings like course titles, URLs, descriptions, levels, metrics, and clean reference titles.

## 📂 Project Directory Structure
* `Course_Recommendation_System.ipynb` - Core Jupyter notebook outlining exploratory data analysis, data pre-processing, matrix mapping, and recommender logic testing.
* `UdemyCleanedTitle.csv` - The engineered dataset housing cleaned titles alongside associated structural course attributes.

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/udemy-course-recommender.git](https://github.com/your-username/udemy-course-recommender.git)
   cd udemy-course-recommender

 1.   Install Necessary Libraries:
Make sure you have Python installed. Run the following command to download all dependencies used in the notebook:

Bash
pip install pandas numpy neattext scikit-learn seaborn matplotlib

Technical Walkthrough
1. Text Preprocessing
The model filters raw course names using neattext to minimize matching noise:
import neattext.functions as nfx

# Remove stopwords and special characters
df['clean_title'] = df['course title'].astype(str).apply(nfx.remove_stopwords)
df['clean_title'] = df['clean_title'].astype(str).apply(nfx.remove_special_characters)

2. Similarity Matrix Generation
Vector matrices are derived via Count Vectorizer or TF-IDF matrix implementations:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer()
cv_mat = cv.fit_transform(df['clean_title'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(cv_mat)

3. Recommendation Execution
The system indexes target elements, tracks similarity scores across vectors, and retrieves the highest-ranking courses relative to the target item.

Dataset Specifications
The dataset maps over thousands of observations with the following metadata:

course title: Name of the course on Udemy.

course url: Direct link to the program webpage.

course description: Explanatory context about skills taught.

Ratings / reviews: Community metrics measuring quantitative feedback.

duration / Lectures: Metric length indicators.

course level: Target audience tiers (e.g., All Levels, Beginner, Intermediate).

clean_title: Normalized keyword sequences used explicitly by the AI model.

🤝 Contributing
Contributions are welcome! Feel free to open an Issue or submit a Pull Request if you've got optimizations for the vector generation process or frontend application integrations.

📄 License
This project is open source and available under the MIT License.
