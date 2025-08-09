import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ===== Load and Prepare Data =====
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# Train model
X = df.iloc[:, :-1]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ===== Streamlit UI =====
st.title("🌸 Iris Flower Classification App")
st.write("Enter the measurements of the flower to predict its species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict Species"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_species = model.predict(sample)[0]
    st.success(f"Predicted Species: **{predicted_species}**")

# ===== Model Accuracy =====
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"**Model Accuracy:** {accuracy:.2%}")

# ===== Confusion Matrix =====
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris_data.target_names,
            yticklabels=iris_data.target_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ===== EDA (Optional) =====
with st.expander("Show Data Exploration"):
    st.write("### Pair Plot")
    fig2 = sns.pairplot(df, hue='species', markers=["o", "s", "D"])
    st.pyplot(fig2)
