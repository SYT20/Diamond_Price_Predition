import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("🌟 Navigation")
page = st.sidebar.radio("Go to", ["📊 Analysis", "🔮 Prediction"])

# Paths to models and dataset
scaler_path = "/media/suyash/DATA/Suyash/Document/Programming/Software/Python/Projects/Project-4_Diamond_Price_Prediction/artifacts/preprocessor.pkl"
model_path = "/media/suyash/DATA/Suyash/Document/Programming/Software/Python/Projects/Project-4_Diamond_Price_Prediction/artifacts/model.pkl"
data_path = "/media/suyash/DATA/Suyash/Document/Programming/Software/Python/Projects/Project-4_Diamond_Price_Prediction/notebooks/data/gemstone.csv"

# Load models
@st.cache_resource
def load_models():
    try:
        scaler = pickle.load(open(scaler_path, "rb"))
        model = pickle.load(open(model_path, "rb"))
        return scaler, model
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.stop()

diamond_scaler, diamond_model = load_models()

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(data_path)
        # If needed, perform additional cleaning here (or rely on your EDA notebook's output)
        # For example, convert categorical features to numeric if required.
        # Assuming target variable is "price"
        X = df.drop("price", axis=1)
        y = df["price"]
        return X, y, df
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        st.stop()

X, y, df = load_data()

# Scale data for visualization
scaled_data = pd.DataFrame(diamond_scaler.transform(X.select_dtypes(include=[np.number])), 
                           columns=X.select_dtypes(include=[np.number]).columns)

# Analysis Page
if page == "📊 Analysis":
    st.title("📈 Diamond Data Analysis Dashboard")
    st.markdown("Gain insights into the **diamonds dataset** with various visualizations.")

    # Display basic dataset information
    st.subheader("📜 Dataset Overview")
    with st.expander("View Dataset Details"):
        st.dataframe(df.head())
        st.write(f"**Dataset shape:** {df.shape}")
        st.write("**Statistical Summary:**")
        st.write(df.describe())

    # Correlation Plot
    st.subheader("🔗 Correlation Plot")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(),c annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Box Plot (Scaled Numeric Data)
    st.subheader("📦 Box Plot (Scaled Numeric Features)")
    fig, ax = plt.subplots()
    sns.boxplot(data=scaled_data, palette="Set3", ax=ax)
    st.pyplot(fig)

    # Histogram for each numeric feature
    st.subheader("📊 Feature Distribution (Histograms)")
    st.markdown("Visualize the distribution of numeric features across the dataset.")
    fig = plt.figure(figsize=(20, 15))
    num_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(num_cols, 1):
        plt.subplot(3, 3, i)
        plt.hist(df[col].dropna(), bins=30, color="skyblue", edgecolor="black")
        plt.title(col)
    st.pyplot(fig)

# Prediction Page
elif page == "🔮 Prediction":
    st.title("💎 Diamond Price Prediction")
    st.markdown("Provide input values to predict the **diamond price**.")

    st.sidebar.subheader("Input Parameters")
    user_input = {}

    # For each numeric feature, create a slider.
    # You may also need to adjust for categorical variables (e.g., cut, color, clarity)
    for col in X.columns:
        # For demonstration, we assume numeric features get sliders.
        # For categorical features, you can use selectbox.
        if np.issubdtype(X[col].dtype, np.number):
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            default_val = (min_val + max_val) / 2  # Default midpoint
            user_input[col] = st.slider(f"{col}", min_val, max_val, default_val)
        else:
            options = list(X[col].unique())
            user_input[col] = st.selectbox(f"{col}", options)

    input_df = pd.DataFrame([user_input])

    # Preprocess categorical features if needed
    # For example, if your model was trained on encoded features,
    # you might need to transform the categorical variables here.
    # This sample assumes that any necessary preprocessing is handled within the scaler.

    # Scale input and make prediction
    try:
        scaled_input = diamond_scaler.transform(input_df)
        prediction = diamond_model.predict(scaled_input)
        st.write(f"### 💎 Predicted Diamond Price: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
