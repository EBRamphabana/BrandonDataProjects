

# test_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib  # If using scikit-learn < 0.22
# from sklearn import joblib  # If using scikit-learn >= 0.22
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report



st.write("""
# DATA SCIENCE PREDICTION APP
         
CodeSoft Data Science Internship
""")

#Loading the Movie Prediction Dataset
codsost_set = pd.read_csv("C:\\Users\\202101775\\Desktop\\working\\datasets\\moviePrediction.csv")

# Load pretrained model
model_path = r'C:\Users\202101775\Downloads\linear_regress_model.pkl'
model = joblib.load(model_path)

X_train = pd.read_csv(r"C:\Users\202101775\Downloads\X_train.csv")
scaler = StandardScaler()
scaler.fit_transform(X_train)

#VISUALIZATIONS OF MOVIES RATING PREDICTIONS

def plot_scatter():
    #1
    yearly_rating = codsost_set.groupby('Year')['Rating'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_rating.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Average Rating per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Rating')
    st.pyplot(fig)

def plot_histogram():
    #2
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(codsost_set['Duration'], codsost_set['Rating'], alpha=0.5)
    ax.set_title('Relationship between Duration and Rating')
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Rating')
    st.pyplot(fig)

def plot_line_chart():
    #3
    top_10_movies = codsost_set.nlargest(10, 'Rating')[['Name', 'Year', 'Rating']]
    st.subheader("Visualization 3: Top 10 Movies According to Rating:")
    st.table(top_10_movies)

def plot_bar_chart():
    #4
    top_10_movies = codsost_set.nlargest(10, 'Rating')[['Name', 'Year', 'Rating']]
    st.subheader("Visualization 2: Top 10 Movies According to Rating:")
    fig, ax = plt.subplots(figsize=(25, 35))
    colors = np.random.rand(10, 3)
    bars = ax.barh(top_10_movies['Name'], top_10_movies['Rating'], color=colors)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Movie Name')
    ax.legend(bars, top_10_movies['Name'], title='Movies', loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)


def plot_movies_per_year_bar_chart():
    #5
    movies_per_year = codsost_set['Year'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    movies_per_year.plot(kind='bar', color='salmon', ax=ax)
    ax.set_title('Number of Movies Released Each Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Movies')
    st.pyplot(fig)

def plot_rating_vs_votes_scatter():
    #6
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(codsost_set['Rating'], codsost_set['Votes'], alpha=0.5)
    ax.set_title('Visualization 6: Relationship between Rating and Votes')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Votes')
    st.pyplot(fig)

def plot_duration_distribution_histogram():
    #7
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(codsost_set['Duration'].dropna(), bins=20, color='orange', edgecolor='black')
    ax.set_title('Duration Distribution')
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Number of Movies')
    st.pyplot(fig)

def plot_top_directors_avg_rating_bar_chart():
    #8
    director_avg_rating = codsost_set.groupby('Director')['Rating'].mean().nlargest(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    director_avg_rating.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Top Directors Based on Average Rating')
    ax.set_xlabel('Director')
    ax.set_ylabel('Average Rating')
    ax.tick_params(axis='x', rotation=45, ha='right')
    st.pyplot(fig)

def plot_correlation_heatmap():
    # Replace non-numeric values with NaN
    codsost_set_numeric = codsost_set.apply(pd.to_numeric, errors='coerce')

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(codsost_set_numeric.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    ax.set_title('Correlation Heatmap')

    # Display the plot using st.pyplot
    st.pyplot(fig)

def plot_rating_distribution_histogram():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(codsost_set['Rating'].dropna(), bins=20, color='purple', edgecolor='black')
    ax.set_title('Rating Distribution')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Movies')
    
    st.pyplot(fig)

def plot_top_directors_avg_rating():
    # Calculate average rating for each director
    director_avg_rating = codsost_set.groupby('Director')['Rating'].mean().nlargest(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(director_avg_rating.index, director_avg_rating, color='purple')
    ax.set_title('Top Directors Based on Average Rating')
    ax.set_xlabel('Director')
    ax.set_ylabel('Average Rating')
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    
    st.pyplot(fig)

def plot_top_directors_avg_rating():
    #st.header("Top Directors Based on Average Rating")

    # Group by director and calculate average rating
    director_avg_rating = codsost_set.groupby('Director')['Rating'].mean().nlargest(10)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    director_avg_rating.plot(kind='bar', color='purple')
    plt.xlabel('Director')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    
    # Display the plot using Streamlit
    st.pyplot(fig)

# Function to make predictions
def make_prediction(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    return prediction[0]


def most_movies_actor():
    st.header("Actor Starring in the Most Movies")

    # Find the actor starring in the most movies
    most_movies_actor = codsost_set[['Actor 1', 'Actor 2', 'Actor 3']].stack().value_counts().idxmax()

    # Display the result using Streamlit
    st.write(f"The actor starring in the most movies is: {most_movies_actor}")

def most_movies_director():
    st.header("Director with the Most Movies")
    
    # Find the director with the most movies
    most_movies_director = codsost_set['Director'].value_counts().idxmax()

    # Display the result using Streamlit
    st.write(f"The director with the most movies is: {most_movies_director}")



# Function to train a linear regression model and display results
def make_prediction(genre_label, director_label, actor1_label, actor2_label, actor3_label, votes):
    # Create a dictionary with user input
    user_input = {
        'Genre_Label': genre_label,
        'Director_Label': director_label,
        'Actor1_Label': actor1_label,
        'Actor2_Label': actor2_label,
        'Actor3_Label': actor3_label,
        'Votes': votes
    }

    # Convert user input to a DataFrame
    input_df = pd.DataFrame([user_input])

    expected_features = ['Genre_Label', 'Director_Label', 'Actor1_Label', 'Actor2_Label', 'Actor3_Label', 'Votes']
    
    if not set(expected_features).issubset(input_df.columns):
        st.error(f"Invalid feature names. Expected features: {', '.join(expected_features)}")
        return None

    input_df = input_df[expected_features]

    input_df_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df_scaled)

    return prediction[0]


def make_batch_prediction(input_df):

    input_df = input_df[['Genre_Label', 'Director_Label', 'Actor1_Label', 'Actor2_Label', 'Actor3_Label','Votes']]  # Update with other features if necessary

    # Scale the input data
    input_df_scaled = scaler.transform(input_df)

    # Make predictions
    predictions = model.predict(input_df_scaled)
    return predictions




#VISUALIZATIONS OF IRIS SPECIES PREDICTION

iris_data = pd.read_csv(r"C:\Users\202101775\Downloads\iris_data.csv")

lowest_values = iris_data.groupby('species').min()

def plot_lowest_sepal_length():

    # Plot lowest sepal length
    st.subheader("Lowest Sepal Length")
    st.bar_chart(lowest_values['sepal_length'], height=300, width=400)
    

    
def plot_lowest_sepal_width():

    # Plot lowest sepal width
    st.subheader("Lowest Sepal Width")
    st.bar_chart(lowest_values['sepal_width'],  height=300, width=400)


def plot_lowest_petal_length():

    # Plot lowest petal length
    st.subheader("Lowest Petal Length")
    st.bar_chart(lowest_values['petal_length'], height=300, width=400)


def plot_lowest_petal_width():

    # Plot lowest petal width
    st.subheader("Lowest Petal Width")
    st.bar_chart(lowest_values['petal_width'], height=300, width=400)

# Function to generate correlation heatmap
def generate_correlation_heatmap():
    # Exclude non-numeric columns from the correlation matrix
    numeric_columns = iris_data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()

    # Create a heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

def collect_user_input():
    st.sidebar.header("Enter Iris Flower Details")
    sepal_length = st.sidebar.slider("Sepal Length", min_value=0, max_value=10, value=1)
    sepal_width = st.sidebar.slider("Sepal Width", min_value=0, max_value=10, value=1)
    petal_length = st.sidebar.slider("Petal Length", min_value=0, max_value=10, value=1)
    petal_width = st.sidebar.slider("Petal Width", min_value=0, max_value=10, value=1)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

#VISUALIZATIONS OF CREDICT FRAUD PREDICTION


def load_and_preprocess_data(dataset_path, xgb_model_path, rf_model_path, ensemble_model_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Data Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Load pre-trained models
    xgb_model = joblib.load(xgb_model_path)
    rf_model = joblib.load(rf_model_path)
    ensemble_model = joblib.load(ensemble_model_path)

    return X_test, y_test, xgb_model, rf_model, ensemble_model

# Example usage
dataset_path = r"C:\Users\202101775\Downloads\creditcard.csv"
xgb_model_path = r"C:\Users\202101775\Downloads\xgboost_model.joblib"
rf_model_path = r"C:\Users\202101775\Downloads\random_forest_model.joblib"
ensemble_model_path = r"C:\Users\202101775\Downloads\voting_classifier_model.joblib"



def main():

    st.sidebar.title(" CODSOFT FT EMMANUEL BRANDON RAMPHABANA")

    # Create a sidebar with a menu
    menu_selection = st.sidebar.radio("Select a page", ["Home", "MOVIE RATING PREDICTION", "IRIS FLOWER CLASSIFICATION", "CREDIT CARD FRAUD DETECTION"])

    # Handle different menu selections
    if menu_selection == "Home":
        st.title("Streamlit App Home Page")
        st.header("Projects Summary")
    
    # Project summaries
        project_summaries = """
        ### Project 1: Movie Rating Prediction with Python
        
        I will be building a machine learning model to predict the rating of a movie based on features such as genre, director, and actors. I will utilize regression techniques to analyze historical movie data, exploring data analysis, preprocessing, feature engineering, and machine learning modeling. The goal is to accurately estimate movie ratings and gain insights into the factors influencing them.

        ### Project 2: Iris Flower Classification
        
        I will develop a machine learning model using the Iris flower dataset to classify flowers into three species: setosa, versicolor, and virginica. Train the model based on measurements of sepals and petals to accurately categorize Iris flowers.

        ### Project 3: Credit Card Fraud Detection
        
        I will build a machine learning model to identify fraudulent credit card transactions. Preprocess and normalize transaction data, handle class imbalance, and split the dataset into training and testing sets. Train a classification algorithm, such as logistic regression or random forests, to distinguish between fraudulent and genuine transactions. Evaluate the model's performance using precision, recall, and F1-score, considering techniques like oversampling or undersampling for improved results.
        """
        st.markdown(project_summaries)

    elif menu_selection == "MOVIE RATING PREDICTION":
        st.title("Streamlit App Project 1")
        st.header("Visualization 1: MOVIE RATING PREDICTION WITH PYTHON")
        
        plot_scatter()
        project_summary1 = """The graph illustrates the stability of the average movie rating on Rotten Tomatoes over the last 15 years, with a slight upward trajectory. Starting at 52% in 2005, the average rating increased to 63% in 2019. Potential explanations for this trend include improved movie quality by studios or an inflation of Rotten Tomatoes' ratings over time, with both factors possibly contributing. The graph also depicts yearly fluctuations, like the 64% average rating in 2016 compared to 59% in 2017, potentially influenced by the release of exceptional or subpar movies in specific years. In summary, while the average rating has shown relative stability, year-to-year variations may stem from a combination of factors, including shifts in movie quality and alterations in Rotten Tomatoes' rating criteria."""
        st.markdown(project_summary1)

        # First row of visualizations (side by side)
        col1, col2 = st.columns(2)

        with col1:
            plot_bar_chart()

        with col2:
            plot_line_chart()
        
        project_summary2 = """Both the list and the graph capture the essence of cinematic excellence, providing a valuable reference in the domain of highly-rated films. The compilation presents an outstanding array of cinematic achievements. Leading the selection is "The Shawshank Redemption" (1994) with an exceptional 9.3 rating, closely followed by the enduring classic "The Godfather" (1972) at 9.2. Notable entries include "The Dark Knight" (2008) and "The Godfather Part II" (1974), both securing a solid 9.0 rating. Other noteworthy mentions encompass "Pulp Fiction" (1994), "12 Angry Men" (1957), "Schindler's List" (1993), and "The Lord of the Rings: The Return of the King" (2003), each boasting an impressive 8.9 rating. Completing the top 10 are "The Lord of the Rings: The Fellowship of the Ring" (2001) and "Forrest Gump" (1994), each earning an 8.8 rating. This compilation and graphical representation stand as a testament to the excellence found within these highly-regarded films. Feel free to inquire for further insights!"""
        st.markdown(project_summary2)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Relationship between Duration and Rating")
            project_summary4 = """The scatter plot reveals a subtle positive correlation between movie duration and ratings, suggesting that, on average, longer movies tend to receive slightly higher ratings. However, this relationship is not consistently strong. The data points are widely dispersed, indicating considerable variability in ratings for movies of all lengths. Despite the overall trend, there are noticeable clusters, potential outliers, and a concentration of ratings between 6 and 10, with a scarcity of movies below 5. The non-linear nature of the relationship and the influence of factors like genre and budget underscore that movie length alone is not a decisive predictor of ratings."""
            st.markdown(project_summary4)

            
        with col4:
            st.subheader("Visualization 4: Interpretation of Relationship between Duration and Rating")
            plot_histogram()
            

        # Fourth row of visualizations (side by side)
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Visualization 5: Duration Distribution")
            plot_duration_distribution_histogram()

        with col6:
            st.subheader("Interpretation of Duration Distribution")
            project_summary4 = """The graph illustrates the distribution of movie durations, showcasing the number of movies within different time ranges. Notably, a significant majority of movies fall within the average duration of 120 to 150 minutes. There is a noticeable decline in the number of movies above 150 minutes, whereas those below 120 to 40 minutes are more prevalent. In essence, the majority of movies are clustered in the 120 to 150-minute interval, highlighting a common duration trend in the dataset."""
            st.markdown(project_summary4)
        # Fourth row of visualizations (side by side)
        col7, col8 = st.columns(2)

        with col7:
            st.subheader("Interpretation of Rating Distribution")
            project_summary3 = """The picture gives us a quick look at how movies are rated in a specific group. Most movies fall in the middle range, around 6 to 7, with about 8,000 of them. Many also have ratings between 5 and 6 or between 7 and 8. But not many movies have really low or really high ratings. There are less than 2,000 movies rated below 4 and less than 1,000 rated above 8. This suggests that most movies are seen as okay or a bit better, and there are fewer that are considered really bad or really good."""
            st.markdown(project_summary3)

        with col8:
            st.subheader("Visualization 6: Rating Distribution")
            plot_rating_distribution_histogram()

        st.subheader("Visualization 7: Number of Movies Released Each Year")   
        plot_movies_per_year_bar_chart()

        project_summary5 = """The graph indicates a general upward trajectory in the number of movie releases over time, with noticeable fluctuations. Notable observations include a consistent rise from 2000 to 2019, peaking at 792 movies in 2019. However, there was a decline in 2020 and 2021, likely influenced by the COVID-19 pandemic, with 406 and 449 movies released, respectively. In 2022, there was a partial recovery with 490 movies, still 30.2% lower than the 2019 peak. Estimates for 1991 suggest a surge in movie releases, ranging from 540 to 630 in the US and Canada, surpassing both the 2000 baseline and the 2019 peak. This spike may be attributed to factors such as increased competition, a saturation of popular genres, and technological advancements, fostering a more competitive and diverse movie landscape."""
        st.markdown(project_summary5)

        col9, col10 = st.columns(2)

        with col9:
            st.subheader("Interpretation of Rating Distribution")
            project_summary3 = """The picture gives us a quick look at how movies are rated in a specific group. Most movies fall in the middle range, around 6 to 7, with about 8,000 of them. Many also have ratings between 5 and 6 or between 7 and 8. But not many movies have really low or really high ratings. There are less than 2,000 movies rated below 4 and less than 1,000 rated above 8. This suggests that most movies are seen as okay or a bit better, and there are fewer that are considered really bad or really good."""
            st.markdown(project_summary3)

        with col10:
            st.subheader("Visualization 8: Top Directors Based on Average Rating")
            plot_top_directors_avg_rating()

        col11, col12 = st.columns(2)

        with col11:
            most_movies_director()

        with col12:
            most_movies_actor()

        st.title("Movie Rating Prediction")

        # Sidebar for user input
        st.sidebar.header("Enter Movie Details")
        genre_label = st.sidebar.number_input("Genre Label", min_value=0, max_value=10000, value=1)
        director_label = st.sidebar.number_input("Director Label", min_value=0, max_value=10000, value=1)
        actor1_label = st.sidebar.number_input("Actor1 Label", min_value=0, max_value=10000, value=1)
        actor2_label = st.sidebar.number_input("Actor2 Label", min_value=0, max_value=10000, value=1)
        actor3_label = st.sidebar.number_input("Actor3 Label", min_value=0, max_value=10000, value=1)
        votes = st.sidebar.number_input("Votes", min_value=0, max_value=500, value=0)


        # Create a dictionary with user input
        user_input = {
            'Genre_Label': genre_label,
            'Director_Label': director_label,
            'Actor1_Label': actor1_label,
            'Actor2_Label': actor2_label,
            'Actor3_Label': actor3_label,
            'Votes': votes
        }

        # Convert user input to a DataFrame
        input_data = pd.DataFrame([user_input])

        # Make prediction on user input
        prediction = make_prediction(genre_label, director_label, actor1_label, actor2_label, actor3_label, votes)

        # Display uploaded data
        st.subheader("Entered Data:")
        st.write(input_data)

        # Display prediction
        st.subheader("Predicted Movie Rating:")
        st.write(f"The predicted rating is: {prediction:.4f}")


        st.title("Movie Rating Batch Prediction")

        #Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read CSV file into DataFrame
            input_data = pd.read_csv(uploaded_file)

            # Display uploaded data
            st.subheader("Uploaded Data:")
            st.write(input_data)

            # Make batch predictions
            predictions = make_batch_prediction(input_data)

            # Add predictions to the DataFrame
            input_data['Predicted Rating'] = predictions

            # Display the DataFrame with predictions
            st.subheader("Predicted Movie Ratings:")
            st.write(input_data)  # Update with other features if necessary












    elif menu_selection == "IRIS FLOWER CLASSIFICATION":
        st.title("Streamlit App Project 2")
        st.header("IRIS FLOWER CLASSIFICATION WITH PYTHON")

       
        # First row of visualizations (side by side)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Intepretation: Lowest Sepal Length")
            project_summary6 = """The first subplot (top-left) displays the species with the lowest sepal length. The bar plot will reveal that Setosa typically has the shortest sepal length compared to Versicolor and Virginica. The distinctive bar for Setosa will be noticeably shorter, indicating its characteristic shorter sepal length."""
            st.markdown(project_summary6)
            
        with col2:
            plot_lowest_sepal_length()

        # Second row of visualizations (side by side)
        col1, col2 = st.columns(2)

        with col1:
            plot_lowest_sepal_width()

        with col2:
            st.subheader("Interpretation: Lowest Sepal Width")
            project_summary7 = """The second subplot (top-right) concentrates on the species with the lowest sepal width. The bar plot will illustrate that Setosa typically has the narrowest sepals compared to Versicolor and Virginica. The bar for Setosa will be the shortest, indicating its characteristic narrow sepal width."""
            st.markdown(project_summary7)

        # Third row of visualizations (side by side)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Interpretation: Lowest Petal Length")
            project_summary8 = """Moving to the bottom-left subplot, it showcases the species with the lowest petal length. The bar plot provides insights into the species that usually exhibit the shortest petals. In this case, Setosa is likely to have the shortest petal length, as reflected by the shortest bar."""
            st.markdown(project_summary8)

        with col2:
            plot_lowest_petal_length()

        # Fourth row of visualizations (side by side)
        col3, col4 = st.columns(2)

        with col3:
            plot_lowest_petal_width()

        with col4:
            st.subheader("Interpretation: Lowest Petal Width")
            project_summary9 = """Finally, the bottom-right subplot focuses on the species with the lowest petal width. The bar plot will indicate that Setosa tends to have the narrowest petals compared to Versicolor and Virginica. The bar for Setosa will be the shortest, specifying its characteristic narrow petal width."""
            st.markdown(project_summary9)

        st.markdown("<h1 style='text-align: center;'>Iris Dataset Correlation Heatmap</h1>", unsafe_allow_html=True)
        generate_correlation_heatmap()
        st.write("Interpretation: The correlation heatmap reveals the correlation matrix of features. Strong correlations between features are represented by brighter colors, while weak correlations are depicted by darker shades. The bright region corresponds to petal length and petal width, indicating a strong positive correlation between these two features. This means that as the petal length increases, the petal width tends to increase as well. The overall heatmap provides a visual guide to understanding the strength and direction of correlations between various pairs of features, aiding in the identification of key relationships within the dataset.")

        st.markdown("<h1 style='text-align: center;'>PREDICTION OF SINGLE AND BATCH SPECIES OF IRIS FLOWER</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Streamlit Single Iris Flower Prediction App</h1>", unsafe_allow_html=True)
    

        # Collect a single user input
        df = collect_user_input()

        # Display user input
        st.subheader("Iris Flower Parameters")
        st.write(df)

        # Load Iris dataset
        iris = datasets.load_iris()
        X = iris.data 
        Y = iris.target 

        # Create and train the classifier
        classifier = RandomForestClassifier()
        classifier.fit(X, Y)

        # Make prediction for a single user input
        prediction = classifier.predict(df)
        prediction_proba = classifier.predict_proba(df)

        # Display prediction for a single user input
        st.subheader("Predicted Iris Flower Species")
        st.write(iris.target_names[prediction])

        # Display prediction probabilities for a single user input
        st.subheader("Prediction Probabilities of Iris Flower Species")
        prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
        st.write(prob_df)

        # Batch Prediction using CSV file
        
        st.markdown("<h1 style='text-align: center;'>Batch Prediction of Iris Flower Species Using CSV File</h1>", unsafe_allow_html=True)

        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file for Batch Prediction", type=["csv"])

        if uploaded_file is not None:
            # Read the uploaded CSV file
            batch_df = pd.read_csv(uploaded_file)

            # Display batch user inputs
            st.subheader("Batch User Inputs from CSV")
            st.write(batch_df)

            # Make batch prediction
            batch_prediction = classifier.predict(batch_df)
            batch_prediction_proba = classifier.predict_proba(batch_df)

            # Add predicted species and prediction probability columns to the table
            batch_df['Predicted Species'] = [iris.target_names[p] for p in batch_prediction]
            batch_df['Prediction Probability'] = [max(prob) for prob in batch_prediction_proba]

            # Display the entire table with predicted species and prediction probability columns
            st.subheader("Batch Prediction And Probabilities")
            
            st.write(batch_df)




            

    elif menu_selection == "CREDIT CARD FRAUD DETECTION":
        st.title("Streamlit App Project 3")
        st.header("CREDIT CARD FRAUD DETECTION WITH PYTHON")

        X_test, y_test, xgb_model, rf_model, ensemble_model = load_and_preprocess_data(
        dataset_path, xgb_model_path, rf_model_path, ensemble_model_path)

        # Batch Prediction Section
        st.sidebar.subheader("Batch Prediction")

        # Input for Batch Prediction
        batch_file = st.sidebar.file_uploader("Upload Batch Data (CSV)", type=["csv"])

        # Sidebar with model selection Project 3
        selected_model = st.sidebar.selectbox("Select Model", ["XGBoost", "Random Forest", "Voting Classifier"])

        # Model prediction and metrics
        if selected_model == "XGBoost":
            predictions = xgb_model.predict(X_test)
            # Display metrics
            st.subheader(f"{selected_model} Metrics:")
            st.write("Precision:", precision_score(y_test, predictions))
            st.write("Recall:", recall_score(y_test, predictions))
            st.write("F1-Score:", f1_score(y_test, predictions))

            project_summary10 = """The XGBoost model exhibits a respectable performance in credit fraud detection, as evidenced by precision, recall, and F1-Score metrics. With a precision of 0.80, the model accurately identifies fraudulent transactions around 80% of the time, while maintaining a commendable recall of 0.85, indicating its ability to capture approximately 85% of actual fraud cases. The F1-Score of 0.82 underscores a balanced trade-off between precision and recall, highlighting the model's reliability in handling the imbalanced nature of the dataset."""
            st.markdown(project_summary10)

            st.write("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
            project_summary12 = """The confusion matrix provides a more detailed perspective, indicating that out of 98 actual fraudulent cases (class 1), the model correctly identified 83, but it also produced 21 false positives. The balance between true positives and false positives is crucial in credit fraud detection, as it impacts both user experience and the effectiveness of fraud prevention."""
            st.markdown(project_summary12)

            # Display Classification Report
            st.text("Classification Report:")
            st.text(classification_report(y_test, predictions))
            project_summary11 = """The classification report reveals a solid credit fraud detection model with an 80% precision for identifying fraudulent transactions and an 85% recall rate. The F1-Score of 0.82 reflects a balanced trade-off between precision and recall. The model achieves perfect precision and recall for non-fraudulent transactions, attaining an overall accuracy of 1.00. The report's macro and weighted averages consistently highlight the model's reliability. Continuous monitoring and adaptation are advised to ensure the model's ongoing effectiveness in detecting credit fraud."""
            st.markdown(project_summary11)

            if st.sidebar.button("Run Batch Prediction") and batch_file is not None:
                batch_data = pd.read_csv(batch_file)
                predictions = xgb_model.predict(batch_data)
                batch_data["XGBoost Prediction"] = predictions
                st.subheader("XGBoost Batch Predictions:")
                st.write(batch_data)


        elif selected_model == "Random Forest":
            predictions = rf_model.predict(X_test)
            # Display metrics
            st.subheader(f"{selected_model} Metrics:")
            st.write("Precision:", precision_score(y_test, predictions))
            st.write("Recall:", recall_score(y_test, predictions))
            st.write("F1-Score:", f1_score(y_test, predictions))

            project_summary10 = """The results presented for the Random Forest model are in the context of credit fraud detection. The Precision of 0.91 indicates that when the model predicts a transaction as fraudulent, it is correct around 91% of the time. The Recall of 0.85 implies that the model captures about 85% of the actual fraudulent transactions. It highlights its ability to identify potential fraud cases effectively. The F1-Score, which combines precision and recall, is 0.88, demonstrating a balanced performance between precision and recall."""
            st.markdown(project_summary10)

            st.write("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
            project_summary12 = """Examining the confusion matrix provides additional insights. Out of the 98 actual fraudulent cases (class 1), the model correctly identified 83, but there were 15 instances where it failed to detect fraud. On the positive side, the model performed exceptionally well in classifying non-fraudulent cases (class 0), with only 8 false positives out of 56,856."""
            st.markdown(project_summary12)

            # Display Classification Report
            st.text("Classification Report:")
            st.text(classification_report(y_test, predictions))
            project_summary11 = """The classification report reflects a highly effective machine learning model for credit fraud detection, with precision, recall, and F1-Score metrics indicating strong performance. The model demonstrates impeccable precision (1.00) in identifying non-fraudulent transactions, minimizing false positives, and maintaining high accuracy (1.00). While the recall for fraudulent transactions is slightly lower at 0.85, suggesting some missed fraud cases, the overall balanced F1-Score of 0.94 underscores the model's ability to effectively navigate the imbalanced nature of the dataset. This indicates a reliable tool for detecting both legitimate and fraudulent credit card transactions, though continuous monitoring and adaptation to evolving fraud patterns remain crucial for sustained effectiveness."""
            st.markdown(project_summary11)

            if st.sidebar.button("Run Batch Prediction") and batch_file is not None:
                batch_data = pd.read_csv(batch_file)
                predictions = rf_model.predict(batch_data)
                batch_data["Random Forest Prediction"] = predictions
                st.subheader("Random Forest Batch Predictions:")
                st.write(batch_data)
            
        else:
            predictions = ensemble_model.predict(X_test)
            # Display metrics
            st.subheader(f"{selected_model} Metrics:")
            st.write("Precision:", precision_score(y_test, predictions))
            st.write("Recall:", recall_score(y_test, predictions))
            st.write("F1-Score:", f1_score(y_test, predictions))

            project_summary10 = """The Voting Classifier presents a strong set of metrics for credit fraud detection. With a precision of 0.87, the model ensures that when it predicts a transaction as fraudulent, it is correct approximately 87% of the time. The recall of 0.86 indicates that the model captures around 86% of the actual fraudulent cases, demonstrating its ability to effectively identify instances of credit card fraud. The F1-Score, standing at 0.86, signifies a well-balanced trade-off between precision and recall, reflecting the model's overall reliability in navigating the imbalanced nature of the dataset."""
            st.markdown(project_summary10)

            st.write("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
            project_summary12 = """The confusion matrix provides a detailed breakdown of the model's performance. Among the 98 actual fraudulent cases (class 1), the Voting Classifier correctly identified 84 instances, but produced 13 false positives. Additionally, for non-fraudulent transactions (class 0), the model achieved a near-perfect result with 56,851 true negatives and only 14 false negatives. The balance between true positives and false positives is crucial in credit fraud detection, as it influences both the accuracy of predictions and the impact on end-users."""
            st.markdown(project_summary12)

            # Display Classification Report
            st.text("Classification Report:")
            st.text(classification_report(y_test, predictions))
            project_summary11 = """he classification report underscores the Voting Classifier's robustness, revealing an accuracy of 1.00, suggesting accurate predictions for both classes. The precision and recall for class 1 are both strong at 0.87 and 0.86, respectively, showcasing the model's effectiveness in correctly identifying and capturing fraudulent transactions. The macro and weighted averages for precision, recall, and F1-Score consistently emphasize the model's reliability across the dataset. """
            st.markdown(project_summary11)

            if st.sidebar.button("Run Batch Prediction") and batch_file is not None:
                batch_data = pd.read_csv(batch_file)
                predictions = ensemble_model.predict(batch_data)
                batch_data["Voting Classifier Prediction"] = predictions
                st.subheader("Voting Classifier Batch Predictions:")
                st.write(batch_data)

if __name__ == "__main__":
    main()

