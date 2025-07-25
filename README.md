
 **Intelligent Phishing Website Detection Model**

This repository contains the implementation of an **intelligent phishing website detection model** using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. It is developed to detect phishing websites by analyzing linguistic, domain-based, and content-based features from URLs and webpages.

The project is part of a final year capstone project at **Sunway University**.

---

 **Features**

* Extracts **66 different features** from URLs and website content:

  * Linguistic & URL-based features (e.g., URL length, obfuscation indicators, suspicious TLDs)
  * Domain & security-based features (e.g., domain age, HTTPS presence)
  * Content & UI-based features (e.g., keywords, hidden form fields, favicon presence)
* Implements multiple ML algorithms:

  * XGBoost
  * Random Forest
  * Decision Tree
  * Logistic Regression
  * Naïve Bayes
* Includes full data preprocessing, feature extraction, model training, and evaluation pipelines.

---

 **Project Structure**

```
phishing-detection/
├── scripts/
│   ├── preprocess.py
│   ├── feat_extraction.py
│   ├── new_feat_extraction.py
│   ├── model_training.py
│   ├── evaluate.py
│   ├── splitting.py
│   ├── combine_dataset.py
│   └── ...
├── models/
│   └── phishing_detection_pipeline.pkl
├── data/
│   └── final_combined_dataset.csv (not included - large dataset)
├── README.md
├── requirements.txt
└── .gitignore
```

---

 **Setup Instructions**

 **1. Clone this repository**

```bash
git clone https://github.com/<your-username>/phishing-detection.git
cd phishing-detection
```

 **2. Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

 **3. Install dependencies**

```bash
pip install -r requirements.txt
```

#### **4. (Optional) Download datasets**

This project uses:

* **[UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)**
* **[OpenPhish URLs](https://openphish.com/)**

Place your datasets in the `data/` folder.

---

 **Usage**

 **1. Preprocess datasets**

```bash
python scripts/preprocess.py
```

 **2. Extract features**

```bash
python scripts/feat_extraction.py
```

 **3. Train models**

```bash
python scripts/model_training.py
```

 **4. Evaluate models**

```bash
python scripts/evaluate.py
```

---

 **Results**

* **XGBoost** achieved the best accuracy of **97.84%**, with high precision and recall.
* Random Forest was the second-best model with **97.73%** accuracy.

For detailed performance metrics, refer to the `Results and Discussion` section of the final report.

---

 **Author**

**Jeff Chooi Loke Peng (21008164)**
BSc (Hons) Information Technology (Computer Networking and Security)
**Supervisor:** Dr. Saad Aslam

---

 **License**

This project is for academic purposes. Please contact the author for permission before reusing any code or datasets.


