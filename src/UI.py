import streamlit as st
import numpy as np
import pandas as pd
from utils.loaddataset import (
    load_synthetic_blobs_dataset,
    load_synthetic_moons_dataset,
    load_synthetic_circles_dataset,
    load_synthetic_classification_dataset,
)
from sklearn.model_selection import train_test_split
from algorithms.svm import SVM
from algorithms.knn import KNN, euclidean_distance, manhattan_distance, cosine_distance
from algorithms.decisiontree import DecisionTree
import plotly.express as px
import plotly.graph_objects as go
from utils.streamlitutils import print_tree

st.set_page_config(layout="wide")


# open external css file and load it into streamlit (the css file is in the same folder as this script)
with open("/home/gabe/VScodeMisc/Classification/src/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.markdown(
    '<div class="Title">Machine Learning Classification - Interactive Demo</div>',
    unsafe_allow_html=True,
)

main_tab, models_tab, datasets_tab, About_us_tab = st.tabs(
    ["Main", "Models", "Datasets", "About us"]
)

# Add radio options for the following parameters: model, search, dataset
model_options = ["SVM", "KNN", "DecisionTree"]
dataset_options = [
    "synthetic blobs",
    "synthetic moons",
    "synthetic circles",
    "synthetic classification",
]
distance_options = ["euclidean", "manhattan", "cosine"]

st.sidebar.markdown(
    """
    <div class="sidebarTitle">
        <p>Side Bar</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("Dataset and Model")

dataset = st.sidebar.selectbox("Select dataset", dataset_options)

model = st.sidebar.selectbox("Select model", model_options)

st.sidebar.header("Hyperparameters")
if model == "SVM":
    lr = st.sidebar.number_input(
        "Learning rate",
        min_value=0.0001,
        max_value=10.0,
        value=0.001,
        format="%f",
    )
    tol = st.sidebar.number_input(
        "Tolerance", format="%f", min_value=0.0, max_value=10.0, value=0.01
    )
    lambda_param = st.sidebar.number_input(
        "Lambda parameter",
        min_value=0.0001,
        max_value=10.0,
        value=0.001,
        format="%f",
    )
    max_iter = st.sidebar.slider("Max iterations", 1, 1000, 10)
    clf = SVM(learning_rate=lr, max_iters=max_iter, tol=tol, lambda_param=lambda_param)
elif model == "KNN":
    k = st.sidebar.slider("k", 1, 20, 5)
    distance_function = st.sidebar.selectbox(
        "Select distance function", distance_options
    )
    clf = KNN(k=k, metric=distance_function)
elif model == "DecisionTree":
    max_depth = st.sidebar.slider("max_depth", 1, 20, 5)
    clf = DecisionTree(max_depth=max_depth)
with main_tab:
    test_size = st.slider("test size", min_value=0.1, max_value=0.9, value=0.2)
    random_state = st.slider("random state", min_value=1, max_value=100, value=42)

    plotly_container = st.container()
    if dataset == "synthetic blobs":
        X_train, X_test, y_train, y_test = load_synthetic_blobs_dataset(
            test_size=test_size, random_state=random_state
        )
        fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
    elif dataset == "synthetic moons":
        X_train, X_test, y_train, y_test = load_synthetic_moons_dataset(
            test_size=test_size, random_state=random_state
        )
        fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
    elif dataset == "synthetic circles":
        X_train, X_test, y_train, y_test = load_synthetic_circles_dataset(
            test_size=test_size, random_state=random_state
        )
        fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
    elif dataset == "synthetic classification":
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = load_synthetic_classification_dataset(
            test_size=test_size, random_state=random_state
        )
        fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
    else:
        raise ValueError("Invalid dataset name.")

    with plotly_container:
        st.header("Dataset visualized")
        st.plotly_chart(fig)
        placeholder = st.empty()
        with placeholder.form(key="my_form"):
            run_script_button = st.form_submit_button(label="Run script")
            if run_script_button:
                clf.fit(X_train, y_train)
                y_pred = clf.score(X_test, y_test)
                y_pred = round(y_pred, 2) * 100
                st.markdown(
                    f"""
                            <div class="results">
                            <p>Accuracy: {y_pred}%</p>
                            </div>
                            """,
                    unsafe_allow_html=True,
                )
                if model == "SVM":
                    # plot in fig the line that separates the two classes
                    w = clf.weights
                    b = clf.bias
                    x = np.linspace(-10, 10, 10000)
                    y = -w[0] / w[1] * x - b / w[1]
                    fig.add_scatter(x=x, y=y, mode="lines")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig)

                elif model == "DecisionTree":
                    tree_strings = print_tree(clf.tree, spacing="\n")
                    for t_string in tree_strings:
                        st.markdown(t_string)

                elif model == "KNN":
                    # plot contour plot
                    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
                    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
                    xx, yy = np.meshgrid(
                        np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
                    )
                    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig = go.Figure(
                        data=go.Contour(
                            z=Z,
                            x=np.arange(x_min, x_max, 0.1),
                            y=np.arange(y_min, y_max, 0.1),
                            colorscale="Viridis",
                            opacity=0.5,
                        )
                    )
                    fig.add_scatter(
                        x=X_train[:, 0],
                        y=X_train[:, 1],
                        mode="markers",
                        marker=dict(color=y_train),
                    )
                    st.plotly_chart(fig)

with models_tab:
    st.markdown(
        """
                <div class="models_tab">
                <h1>Models</h1>
                <p>In ML, a model is a mathematical representation of a real-world process. It's a program that improves its performance at a task through experience. In supervised learning, a model learns from labeled training data to predict outcomes for unseen data.</p>
                </div>
                """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
                <div class="modelTitle">SVM</div>
                <div class="modelDescription">Support Vector Machines are powerful classifiers that work by finding the hyperplane that best separates data points of different classes in a high-dimensional space. SVM aims to maximize the margin between classes, making it robust to outliers. </div>
                """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("""<div class="modelTitle">KNN</div>""", unsafe_allow_html=True)
    st.markdown(
        """<div class="modelDescription">k-NN is a simple yet effective algorithm for classification and regression tasks. It classifies a data point based on the majority class of its k nearest neighbors in the feature space. The choice of k influences the model's sensitivity to noise and its ability to capture underlying patterns. k-NN is non-parametric and easy to understand, making it a versatile choice for various applications.</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """<div class="modelTitle">Decision Tree</div>""", unsafe_allow_html=True
    )
    st.markdown(
        """<div class="modelDescription">Decision Trees are versatile models that recursively split the dataset based on feature values, creating a tree-like structure for decision-making. Each internal node represents a decision based on a feature, and each leaf node corresponds to the predicted class. Decision Trees are interpretable and can handle both classification and regression tasks. However, they are prone to overfitting, which can be mitigated using techniques like pruning.</div>""",
        unsafe_allow_html=True,
    )

with datasets_tab:
    st.markdown(
        """
                <div class="datasets_tab">
                <h1>Datasets</h1>
                <p>A dataset is a collection of data points that share a common theme. Each data point represents a single instance of data, such as an observation or an event. Datasets are used to train and evaluate machine learning models. They are also used to test the performance of algorithms in a controlled environment.</p>
                </div>
                """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
                <div class="datasetTitle">Synthetic Blobs</div>
                <div class="datasetDescription">The synthetic blobs dataset is a collection of randomly generated, isotropic Gaussian blobs. It's often used to demonstrate clustering algorithms. Each blob represents a cluster of data points, making it suitable for tasks where the goal is to identify and group similar patterns. This dataset is particularly useful for understanding the behavior of clustering algorithms in different scenarios.</div>
                """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """<div class="datasetTitle">Synthetic Moons</div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="datasetDescription">The synthetic moons dataset is designed for binary classification tasks. It consists of crescent moon-shaped clusters of points, making it challenging for linear classifiers. This dataset is commonly used to illustrate the effectiveness of non-linear classification algorithms, such as support vector machines with a non-linear kernel or decision trees.</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """<div class="datasetTitle">Synthetic Circles</div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="datasetDescription">Similar to synthetic moons, the synthetic circles dataset is crafted for binary classification. It contains concentric circles of points with varying degrees of noise, making it a useful benchmark for algorithms that need to capture complex, non-linear decision boundaries. This dataset is often employed to showcase the capabilities of classifiers that can handle circular decision boundaries.</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """<div class="datasetTitle">Synthetic Classification</div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="datasetDescription">The synthetic classification dataset is a general-purpose synthetic dataset for classification tasks. It allows users to control various parameters, such as the number of samples, features, and classes, making it customizable for specific experimentation needs. This dataset is versatile and can be used to evaluate the performance of classifiers in a controlled environment.</div>""",
        unsafe_allow_html=True,
    )

with About_us_tab:
    st.markdown(
        """
                <div class="About_us_tab">
                <h1>About us</h1>
                <p> I am a Computer Science Student, and I am currently working on this project to visualize the different classification algorithms.
                </div>
                """,
        unsafe_allow_html=True,
    )

    # Footer section

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p> Made by Gabriel Resch</p>
        <p>Github: <a href=https://github.com/Resch-Gabriel-Z>"<span class="highlight">Resch-Gabriel-Z</span>"</a></p>
        <p>🚀 Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)
