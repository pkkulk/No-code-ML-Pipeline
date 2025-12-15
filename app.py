import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="No-Code ML Pipeline", layout="centered")

st.title("🧠 No-Code ML Pipeline Builder")
st.caption("Data → Preprocessing → Split → Model → Results")

for step in ["step1", "step2", "step3", "step4"]:
    if step not in st.session_state:
        st.session_state[step] = False
st.header("Step 1: Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset loaded successfully")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.write("Column Names:", list(df.columns))
        st.dataframe(df.head())

        st.session_state.df = df
        st.session_state.step1 = True

    except:
        st.error("❌ Invalid file format")

if st.session_state.step1:
    st.header("Step 2: Data Preprocessing")

    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found.")
    else:
        method = st.radio(
            "Choose preprocessing method:",
            ["None", "Standardization", "Normalization"]
        )

        if st.button("Apply Preprocessing"):
            df_processed = df.copy()

            if method == "Standardization":
                scaler = StandardScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.success("✅ Standardization applied")

            elif method == "Normalization":
                scaler = MinMaxScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.success("✅ Normalization applied")

            else:
                st.info("No preprocessing applied")

            st.dataframe(df_processed.head())
            st.session_state.df_processed = df_processed
            st.session_state.step2 = True
# ---------------- STEP 3: TRAIN TEST SPLIT ----------------
if st.session_state.step2:
    st.header("Step 3: Train–Test Split")

    df = st.session_state.df_processed

    target = st.selectbox("Select target column", df.columns)

    split_ratio = st.slider("Test size (%)", 20, 40, 30)

    if st.button("Split Dataset"):
        # Separate X and y
        X = df.drop(columns=[target])
        y = df[target]

        # 🔥 KEEP ONLY NUMERIC FEATURES
        X = X.select_dtypes(include=["int64", "float64"])

        if X.shape[1] == 0:
            st.error("❌ No numeric feature columns available for training.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_ratio / 100, random_state=42
            )

            st.success("✅ Dataset split completed")
            st.write(f"Features used: {list(X.columns)}")
            st.write(f"Training samples: {len(X_train)}")
            st.write(f"Testing samples: {len(X_test)}")

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.step3 = True

# ---------------- STEP 4: MODEL TRAINING ----------------
if st.session_state.step3:
    st.header("Step 4: Model Selection & Training")

    model_choice = st.radio(
        "Choose model:",
        ["Logistic Regression", "Decision Tree Classifier"]
    )

    if st.button("Train Model"):
        y_train = st.session_state.y_train

        # 🔥 HARD STOP if target is continuous
        if y_train.dtype in ["float64", "int64"] and y_train.nunique() > 10:
            st.error(
                "❌ Invalid target column selected.\n\n"
                "You selected a continuous (numeric) target.\n"
                "This app supports only *classification* models.\n\n"
                "Please select a categorical target column "
                "(e.g., 0/1, Yes/No, class labels)."
            )
            st.stop()   # ⛔ THIS IS THE FIX

        # ---- SAFE TO TRAIN BELOW ----
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = DecisionTreeClassifier()

        model.fit(st.session_state.X_train, y_train)
        predictions = model.predict(st.session_state.X_test)

        accuracy = accuracy_score(st.session_state.y_test, predictions)
        cm = confusion_matrix(st.session_state.y_test, predictions)

        st.session_state.accuracy = accuracy
        st.session_state.cm = cm
        st.session_state.step4 = True

        st.success("✅ Model trained successfully")

if st.session_state.step4:
    st.header("Step 5: Results")

    st.metric("Accuracy", f"{st.session_state.accuracy * 100:.2f}%")

    fig, ax = plt.subplots()
    ax.imshow(st.session_state.cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.success("🎉 ML Pipeline executed successfully!")
