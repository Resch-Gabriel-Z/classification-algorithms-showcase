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
import plotly.express as px


def main():
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
    model_options = ["SVM", "KNN"]
    dataset_options = [
        "synthetic_blobs",
        "synthetic_moons",
        "synthetic_circles",
        "synthetic_classification",
    ]
    distance_options = ["euclidean", "manhattan", "cosine"]

    with main_tab:
        st.header("Input")

        dataset = st.selectbox("Select dataset", dataset_options)
        uploaded_dataset = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_dataset is not None:
            uploaded_dataset = pd.read_csv(uploaded_dataset)
            st.dataframe(data=uploaded_dataset, height=200)

        model = st.selectbox("Select model", model_options)

        st.header("Hyperparameters")
        if model == "SVM":
            st.header("SVM Hyperparameters")
            lr = st.number_input(
                "Learning rate",
                min_value=0.0001,
                max_value=1.0,
                value=0.001,
                format="%f",
            )
            tol = st.number_input("Tolerance", format="%f")
            lambda_param = st.number_input(
                "Lambda parameter",
                min_value=0.0001,
                max_value=1.0,
                value=0.001,
                format="%f",
            )
            max_iter = st.slider("Max iterations", 1, 1000, 10)
            clf = SVM(
                learning_rate=lr, max_iters=max_iter, tol=tol, lambda_param=lambda_param
            )
        elif model == "KNN":
            st.header("KNN Hyperparameters")
            k = st.slider("k", 1, 20, 5)
            distance_function = st.selectbox(
                "Select distance function", distance_options
            )
            clf = KNN(k=k, metric=distance_function)
        test_size = st.slider("test_size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = st.slider("random_state", min_value=1, max_value=100, value=42)

        if uploaded_dataset is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                uploaded_dataset.iloc[:, :-1],
                uploaded_dataset.iloc[:, -1],
                test_size=test_size,
                random_state=random_state,
            )
        else:
            if dataset == "synthetic_blobs":
                X_train, X_test, y_train, y_test = load_synthetic_blobs_dataset(
                    test_size=test_size, random_state=random_state
                )
                fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
            elif dataset == "synthetic_moons":
                X_train, X_test, y_train, y_test = load_synthetic_moons_dataset(
                    test_size=test_size, random_state=random_state
                )
                fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
            elif dataset == "synthetic_circles":
                X_train, X_test, y_train, y_test = load_synthetic_circles_dataset(
                    test_size=test_size, random_state=random_state
                )
                fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
            elif dataset == "synthetic_classification":
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

        st.header("Dataset visualized")
        st.plotly_chart(fig)
        if st.button("Run Script"):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write("Accuracy:", clf.score(X_test, y_test))

            if model == "SVM":
                # plot in fig the line that separates the two classes
                w = clf.weights
                b = clf.bias
                x = np.linspace(-10, 10, 10000)
                y = -w[0] / w[1] * x - b / w[1]
                fig.add_scatter(x=x, y=y, mode="lines")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig)

    with models_tab:
        st.markdown(
            """
                    <div class="models_tab">
                    <h1>Models</h1>
                    <p>look at those models poggers
                    </div>
                    """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
                    <div class="modelTitle">SVM</div>
                    """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class=svmdescription>Svm is cool</div>
        """,
            unsafe_allow_html=True,
        )

    with datasets_tab:
        st.markdown(
            """
                    <div class="datasets_tab">
                    <h1>Datasets</h1>
                    <p>look at those datasets poggers
                    </div>
                    """,
            unsafe_allow_html=True,
        )

    with About_us_tab:
        st.markdown(
            """
                    <div class="About_us_tab">
                    <h1>About us</h1>
                    <p>look at those About us poggers
                    </div>
                    """,
            unsafe_allow_html=True,
        )

        # Footer section

    st.sidebar.markdown(
        """
        <div class="sidebarTitle">
            <p>Side Bar</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>Footer</p>
            <p>Footer.</p>
            <p>🚀 Powered by Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
