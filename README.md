
# Mall Customer Segmentation & Streamlit Application

## Overview
This project performs customer segmentation using K-Means clustering and an interactive Streamlit web interface to analyze customer behaviors based on their annual income and spending score.

## Dataset
- File: `data/customers.csv`
- Total samples: 200
- Clustering features:
  - Annual Income (k$)
  - Spending Score (1-100)

## Method
1. Scale the clustering features with StandardScaler.
2. Compare several K values using the Elbow Method.
3. Run K-Means using the K selected in the Streamlit app.
4. Visualize the resulting customer clusters.

## Streamlit Application
**Live App:** [https://mall-customer-segmentation-a7eumzydkgydkuweq9jv4e.streamlit.app/]

### Screenshot
![Streamlit Application](<img width="1885" height="863" alt="2026-09-24_21-23-31" src="https://github.com/user-attachments/assets/24ec046b-1eb3-433a-8436-6ea1a15bebe4" />
)

## Installation
```bash
git clone [https://github.com/Ridwanul-hoque/Mall-Customer-Segmentation](https://github.com/Ridwanul-hoque/Mall-Customer-Segmentation)
cd Mall-Customer-Segmentation
pip install -r requirements.txt

```

## Usage

```bash
streamlit run app.py

```

## Project Structure

```text
mall-customer-segmentation/
|-- data/
|   `-- customers.csv
|-- app.py
|-- screenshots/
|   `-- <img width="1885" height="863" alt="2026-09-24_21-23-31" src="https://github.com/user-attachments/assets/d2f4b388-1745-4498-b95d-6f8c26f7c68b" />

|-- requirements.txt
|-- mall-customer-segmentation.ipynb
`-- README.md

```

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit

```

