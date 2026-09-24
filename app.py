import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation App", page_icon="🛍️", layout="wide"
)

if "k_value" not in st.session_state:
  st.session_state["k_value"] = 5


def set_k(val: int):
  st.session_state["k_value"] = int(val)


@st.cache_data
def load_data():
  df = pd.read_csv("data/customers.csv")
  df.columns = df.columns.str.strip()
  df = df.rename(
      columns={
          "Annual Income (k$)": "Annual Income",
          "Spending Score (1-100)": "Spending Score",
          "CustomerID": "Customer ID",
      }
  )
  return df


df = load_data()

st.sidebar.header("Settings")

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    step=1,
    key="k_value",
)

st.title("🛍️ Mall Customer Segmentation App")
st.markdown(
    "This application performs customer segmentation using **K-Means"
    " Clustering**."
)

st.header("1. Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Age Range", f"{df['Age'].min()}–{df['Age'].max()}")
col3.metric(
    "Income Range",
    f"{df['Annual Income'].min()}k–{df['Annual Income'].max()}k",
)
col4.metric(
    "Spending Score Range",
    f"{df['Spending Score'].min()}–{df['Spending Score'].max()}",
)

st.dataframe(df.head(10), use_container_width=True)

X_raw = df[["Annual Income", "Spending Score"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

st.markdown("---")
st.header("2. Elbow Method — Choosing K")

ks = list(range(2, 11))
inertias = []
for kk in ks:
  km_tmp = KMeans(n_clusters=kk, random_state=42, n_init="auto")
  km_tmp.fit(X_scaled)
  inertias.append(km_tmp.inertia_)

knee = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
optimal_k = knee.knee

col_left, col_right = st.columns([2, 1])

with col_left:
  fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4), dpi=120)
  ax_elbow.plot(
      ks, inertias, marker="o", color="steelblue", linewidth=2, markersize=8
  )
  ax_elbow.axvline(
      st.session_state["k_value"],
      color="orange",
      linestyle="--",
      linewidth=2,
      label=f"Selected K = {st.session_state['k_value']}",
  )
  ax_elbow.set_xlabel("Number of Clusters (K)", fontsize=11)
  ax_elbow.set_ylabel("Inertia", fontsize=11)
  ax_elbow.set_title("Elbow Curve", fontsize=13)
  ax_elbow.legend()
  ax_elbow.grid(True, linestyle="--", alpha=0.3)
  plt.tight_layout()
  st.pyplot(fig_elbow, use_container_width=True)

with col_right:
  st.markdown("### Suggested K")
  if optimal_k is not None:
    st.metric("Optimal K (KneeLocator)", int(optimal_k))
    st.button(
        "Use suggested K",
        on_click=set_k,
        args=(int(optimal_k),),
        key="use_optimal_k_btn",
    )
  else:
    st.warning("No clear elbow found.")

st.markdown("---")
st.header(f"3. Dynamic K-Means Clustering — K = {st.session_state['k_value']}")

km = KMeans(
    n_clusters=int(st.session_state["k_value"]), random_state=42, n_init="auto"
)
labels = km.fit_predict(X_scaled)

df_out = df.copy()
df_out["Cluster"] = labels
df_out["Cluster"] = df_out["Cluster"].apply(lambda x: f"Cluster {x+1}")

centers = scaler.inverse_transform(km.cluster_centers_)

st.subheader("Cluster Visualization")
fig_scatter, ax_scatter = plt.subplots(figsize=(9, 5), dpi=120)
palette = sns.color_palette("Set2", st.session_state["k_value"])

for i, cluster in enumerate(sorted(df_out["Cluster"].unique())):
  subset = df_out[df_out["Cluster"] == cluster]
  ax_scatter.scatter(
      subset["Annual Income"],
      subset["Spending Score"],
      label=cluster,
      color=palette[i],
      s=60,
      alpha=0.85,
      edgecolors="white",
      linewidth=0.5,
  )

ax_scatter.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="*",
    s=300,
    color="black",
    label="Centroid",
    zorder=5,
)
ax_scatter.set_xlabel("Annual Income (k$)", fontsize=11)
ax_scatter.set_ylabel("Spending Score (1-100)", fontsize=11)
ax_scatter.set_title("Annual Income vs Spending Score", fontsize=13)
ax_scatter.legend(title="Cluster")
ax_scatter.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
st.pyplot(fig_scatter, use_container_width=True)

st.markdown("### Cluster Profiles")
profile = (
    df_out[["Age", "Annual Income", "Spending Score", "Cluster"]]
    .groupby("Cluster")
    .mean()
    .round(1)
)
profile.insert(0, "Count", df_out.groupby("Cluster").size())
st.dataframe(profile, use_container_width=True)
