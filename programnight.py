import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score

# --- Supervised Models ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- Unsupervised Models ---
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Streamlit page configuration
st.set_page_config(
    page_title="ML Model Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Dictionaries for Model Mapping ---
SUPERVISED_MODELS = {
    "Decision Tree Classifier": DecisionTreeClassifier,
    "Random Forest Classifier": RandomForestClassifier,
    "Support Vector Machine (SVM)": SVC
}

UNSUPERVISED_MODELS = {
    "KMeans": KMeans,
    "Agglomerative Clustering": AgglomerativeClustering,
    "DBSCAN": DBSCAN
}

# --- Global Preprocessing Pipeline Definition ---
# This function prepares data for model training
def create_preprocessing_pipeline():
    """Defines a universal preprocessing pipeline for any mixed dataset."""
    
    # Transformers for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Fill missing values with mean
        ('scaler', StandardScaler()) # Scale data
    ])

    # Transformers for categorical data
    categorical_transformer = Pipeline(steps=[
        # Fill missing data with 'missing'
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
        # Convert categories to numerical
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer applies all transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, lambda X: X.select_dtypes(include=np.number).columns.tolist()),
            ('cat', categorical_transformer, lambda X: X.select_dtypes(include=['object', 'category']).columns.tolist())
        ],
        remainder='passthrough' # Leave remaining columns as is
    )
    
    return preprocessor

# --- Main Application Functions ---

def load_data():
    """Handles file upload and data preview."""
    st.sidebar.header("📊 1. Data Input")
    
    # File Uploader with unique key to avoid internal errors
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV Dataset", 
        type="csv", 
        key="csv_file_uploader"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset loaded successfully!")
            
            st.subheader("📋 Data Preview")
            st.markdown(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Enhanced data preview with tabs
            tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Data Types", "Statistical Summary"])
            
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                st.write("**Data Types Overview**")
                dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                st.dataframe(dtype_df, use_container_width=True)
            
            with tab3:
                st.write("**Statistical Summary**")
                st.dataframe(df.describe(), use_container_width=True)
            
            return df
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return None
    return None

def select_model_and_params(df):
    """Handles model selection and hyperparameters, stores in Session State."""
    
    st.sidebar.header("🤖 2. Model Configuration")
    
    # Store learning type in Session State
    if 'learning_type' not in st.session_state:
        st.session_state['learning_type'] = "Supervised (Classification)"

    learning_type = st.sidebar.radio(
        "Select Learning Type",
        ("Supervised (Classification)", "Unsupervised (Clustering)"),
        key="learning_type_radio"
    )
    st.session_state['learning_type'] = learning_type

    model_params = {}
    
    # Initialize algorithm name and parameters in session state
    if 'algorithm_name' not in st.session_state:
         st.session_state['algorithm_name'] = None
    if 'model_params' not in st.session_state:
         st.session_state['model_params'] = {}
    if 'target_column' not in st.session_state:
         st.session_state['target_column'] = None
    
    if learning_type == "Supervised (Classification)":
        
        # --- Supervised Learning (Target Column Selection) ---
        if df is not None:
            all_columns = df.columns.tolist()
            target_column = st.sidebar.selectbox(
                "Select Target Column (Y)", 
                [''] + all_columns,
                index=0,
                key='target_col_select'
            )
            st.session_state['target_column'] = target_column
            
            if not target_column:
                st.session_state['algorithm_name'] = None
                return 
        
        # --- Supervised Model Selection ---
        algorithm = st.sidebar.selectbox(
            "Select Classification Model",
            tuple(SUPERVISED_MODELS.keys()),
            key="supervised_model_select"
        )
        st.session_state['algorithm_name'] = algorithm
        
        # --- Hyperparameter Tuning ---
        st.sidebar.subheader("⚙️ Hyperparameters")
        
        if algorithm == "Decision Tree Classifier":
            model_params['max_depth'] = st.sidebar.slider("Max Depth", 1, 30, 5, key='dtc_max_depth_slider')
            model_params['min_samples_leaf'] = st.sidebar.slider("Min Samples Leaf", 1, 20, 2, key='dtc_min_samples_leaf_slider')
            
        elif algorithm == "Random Forest Classifier":
            model_params['n_estimators'] = st.sidebar.slider("N Estimators", 10, 500, 100, key='rfc_n_estimators_slider')
            model_params['max_depth'] = st.sidebar.slider("Max Depth", 1, 30, 10, key='rfc_max_depth_slider')
            model_params['random_state'] = 42
            
        elif algorithm == "Support Vector Machine (SVM)":
            model_params['C'] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.01, key='svc_c_slider')
            model_params['kernel'] = st.sidebar.selectbox("Kernel", ('linear', 'rbf', 'poly'), key='svc_kernel_select')
            model_params['random_state'] = 42

        st.session_state['model_params'] = model_params
            
    else: # Unsupervised Learning
        st.session_state['target_column'] = None
        
        # --- Unsupervised Model Selection ---
        algorithm = st.sidebar.selectbox(
            "Select Clustering Model",
            tuple(UNSUPERVISED_MODELS.keys()),
            key="unsupervised_model_select"
        )
        st.session_state['algorithm_name'] = algorithm
        
        # --- Hyperparameter Tuning ---
        st.sidebar.subheader("⚙️ Hyperparameters")
        
        if algorithm == "KMeans":
            model_params['n_clusters'] = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3, key='kmeans_n_clusters_slider')
            model_params['random_state'] = 42
            model_params['n_init'] = 'auto'
            
        elif algorithm == "Agglomerative Clustering":
            model_params['n_clusters'] = st.sidebar.slider("Number of Clusters", 2, 10, 3, key='agg_n_clusters_slider')
            model_params['linkage'] = st.sidebar.selectbox("Linkage", ('ward', 'complete', 'average', 'single'), key='agg_linkage_select')
            
        elif algorithm == "DBSCAN":
            model_params['eps'] = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1, key='dbscan_eps_slider')
            model_params['min_samples'] = st.sidebar.slider("Min Samples", 1, 10, 5, key='dbscan_min_samples_slider')
            
        st.session_state['model_params'] = model_params


def run_supervised_model(df, target_column, model_instance, model_params):
    """Displays Supervised Learning results and evaluation."""
    
    st.subheader("🎯 Supervised Learning: Results & Evaluation")
    
    # 1. Split data into train and test
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Error if target column has too many unique values
    if y.nunique() > 50 and y.dtype in [np.number, 'object']:
        st.error("Selected target column has too many unique values. Please choose a valid classification target.")
        return

    # Data splitting (with stratification if possible)
    if min(y.value_counts()) < 2:
        st.warning("⚠️ Warning: Some classes have only one sample. Stratification skipped.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    st.info(f"📊 Data Split: Training ({X_train.shape[0]} samples), Testing ({X_test.shape[0]} samples).")

    # 2. Create Full Pipeline (Preprocessing + Model)
    preprocessor = create_preprocessing_pipeline()
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_instance) 
    ])

    # 3. Train Model
    with st.spinner(f"🔄 Training {model_instance.__class__.__name__}..."):
        full_pipeline.fit(X_train, y_train)

    # 4. Predict and Evaluate
    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create tabs for different evaluation metrics
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["📈 Performance Metrics", "📊 Classification Report", "🎭 Confusion Matrix"])
    
    with eval_tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Model Accuracy", value=f"{accuracy:.4f}")
        
        with col2:
            # Calculate precision, recall, f1 from classification report
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                weighted_f1 = report['weighted avg']['f1-score']
                st.metric(label="Weighted F1-Score", value=f"{weighted_f1:.4f}")
            except:
                st.metric(label="Weighted F1-Score", value="N/A")
        
        with col3:
            try:
                weighted_precision = report['weighted avg']['precision']
                st.metric(label="Weighted Precision", value=f"{weighted_precision:.4f}")
            except:
                st.metric(label="Weighted Precision", value="N/A")
        
    with eval_tab2:
        st.subheader("Detailed Classification Report")
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        except ValueError:
            st.error("Classification Report cannot be displayed: Only one class might be present in target column after splitting.")
        
    with eval_tab3:
        st.subheader("Confusion Matrix")
        try:
            cm = confusion_matrix(y_test, y_pred)
            
            labels = sorted(y.unique())
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': 'Count'})
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for {model_instance.__class__.__name__}')
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Confusion Matrix cannot be displayed: {e}")
            
    st.success("✅ Supervised Model evaluation completed.")

def run_unsupervised_model(df, model_instance, model_params):
    """Displays Unsupervised Learning (Clustering) results and evaluation."""
    
    st.subheader("🔍 Unsupervised Learning: Results & Evaluation")
    
    # 1. Preprocess Data
    X = df.copy()
    preprocessor = create_preprocessing_pipeline()
    
    with st.spinner("🔄 Data Preprocessing in progress..."):
        X_processed = preprocessor.fit_transform(X)
        
        try:
            feature_names = preprocessor.get_feature_names_out()
            X_df = pd.DataFrame(X_processed, columns=feature_names)
        except AttributeError:
            X_df = pd.DataFrame(X_processed)
            st.warning("⚠️ Feature names not available after preprocessing. Plot axes will be generic.")

    # 2. Train Model
    try:
        model_name = model_instance.__class__.__name__
        if model_name == 'DBSCAN':
            labels = model_instance.fit_predict(X_df)
        else:
            model_instance.fit(X_df)
            labels = model_instance.labels_
            
    except Exception as e:
        st.error(f"❌ Error in model training. Please check hyperparameters. Error: {e}")
        return

    # 3. Evaluate and Visualize
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Create tabs for clustering results
    cluster_tab1, cluster_tab2 = st.tabs(["📊 Cluster Metrics", "📈 Visualization"])
    
    with cluster_tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Identified Clusters", value=n_clusters)
        
        with col2:
            # Calculate silhouette score
            if n_clusters > 1:
                try:
                    score = silhouette_score(X_df, labels)
                    st.metric(label="Silhouette Score", value=f"{score:.4f}", 
                             help="Higher score indicates better separation between clusters.")
                except Exception as e:
                    st.metric(label="Silhouette Score", value="N/A")
            else:
                st.metric(label="Silhouette Score", value="N/A")
        
        with col3:
            # Count noise points for DBSCAN
            if model_name == 'DBSCAN':
                noise_points = np.sum(labels == -1)
                st.metric(label="Noise Points", value=noise_points)
            else:
                st.metric(label="Total Samples", value=len(labels))

    with cluster_tab2:
        st.subheader("Cluster Visualization (First 2 Principal Components)")
        
        if X_df.shape[1] < 2:
            st.warning("Dataset has less than 2 features. Scatter plot cannot be created.")
            return

        # Use PCA for visualization
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_df)
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = labels.astype(str)
            
            # Create scatter plot
            plt.figure(figsize=(10, 7))
            scatter = sns.scatterplot(
                x="PC1", 
                y="PC2", 
                hue="Cluster", 
                palette="viridis",
                data=pca_df, 
                legend="full", 
                alpha=0.8,
                s=60
            )
            plt.title(f'{model_instance.__class__.__name__} Clustering Results (via PCA)')
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            st.pyplot(plt.gcf())
            
        except Exception as e:
            st.error(f"❌ Error during PCA visualization: {e}")
            
    st.success("✅ Unsupervised Model evaluation completed.")

# --- Streamlit UI Main Execution ---

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">✨ ML Model Explorer Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("A comprehensive platform for exploring Supervised and Unsupervised Learning models with automatic data preprocessing.")

    # 1. Load Data
    df = load_data()
    
    # 2. Configure Model (Only sets session state)
    select_model_and_params(df)

    # Get necessary variables from session state
    learning_type = st.session_state.get('learning_type')
    algorithm_name = st.session_state.get('algorithm_name')
    model_params = st.session_state.get('model_params', {})
    target_column = st.session_state.get('target_column')

    st.sidebar.markdown("---")
    
    if df is not None:
        
        # Check if model is ready to run
        is_ready = True
        
        if learning_type == "Supervised (Classification)" and (not target_column or target_column == ''):
            is_ready = False
        
        if not algorithm_name:
             is_ready = False
             
        # 3. Run Button
        if is_ready:
            # Button with unique key
            if st.sidebar.button("🚀 Run Model Analysis", key='run_model_button_final', type="primary"):
                st.header("🎯 Model Execution")
                
                # --- Instantiate model before running ---
                if learning_type == "Supervised (Classification)":
                    ModelClass = SUPERVISED_MODELS.get(algorithm_name)
                    if ModelClass:
                        model_instance = ModelClass(**model_params)
                        st.subheader(f"Selected Model: {algorithm_name} (Target: {target_column})")
                        run_supervised_model(df, target_column, model_instance, model_params)
                    else:
                        st.error(f"❌ Supervised model not found: {algorithm_name}")
                    
                elif learning_type == "Unsupervised (Clustering)":
                    ModelClass = UNSUPERVISED_MODELS.get(algorithm_name)
                    if ModelClass:
                        model_instance = ModelClass(**model_params)
                        st.subheader(f"Selected Model: {algorithm_name}")
                        run_unsupervised_model(df, model_instance, model_params)
                    else:
                        st.error(f"❌ Unsupervised model not found: {algorithm_name}")
        else:
             if learning_type == "Supervised (Classification)" and df is not None:
                st.sidebar.error("❌ Model cannot run. Please select a valid target column.")
             
    else:
        st.info("⬆️ To get started, please upload a CSV file in the sidebar.")


if __name__ == "__main__":
    main()
