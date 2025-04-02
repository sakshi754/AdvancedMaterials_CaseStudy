# AdvancedMaterials_CaseStudy
# Manufacturing Defect Prediction in Advanced Materials

This project demonstrates a machine learning pipeline to predict defects in composite material manufacturing using synthetic IoT sensor data. It includes anomaly detection, supervised classification, and MLflow-based governance logging to simulate a real-world industrial analytics use case.

## 📌 Objective
To proactively identify potential defects in the production line using sensor data, enabling predictive quality control and reducing rework or scrap rates in manufacturing.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- Gradient Boosting, Isolation Forest
- MLflow for experiment tracking

## 🧪 Dataset (Synthetic)
Features:
- `temperature` (°F)
- `pressure` (psi)
- `humidity` (%)
- `vibration` (m/s²)
- `speed` (units/min)
- `defect_label` (0 = OK, 1 = Defective)

Data is generated synthetically using NumPy to simulate realistic sensor behavior in a production environment.

## 📊 Model Workflow
1. **Data Generation**: Simulated sensor readings for 1,000 production units.
2. **Anomaly Detection**: Applied Isolation Forest to flag potential anomalies.
3. **Supervised Learning**: Trained a Gradient Boosting Classifier on labeled data.
4. **Evaluation**: Printed confusion matrix and classification report.
5. **Governance & Logging**: Logged experiment with MLflow (params, accuracy, model artifact).

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn mlflow
python manufacturing_defect_prediction.py
```

## 📈 Results
- Demonstrates how early detection of defects can be achieved with simple ML models.
- Sets a foundation for integrating this into larger smart manufacturing pipelines.

## 🔐 Compliance Focus
The pipeline uses MLflow for traceability and reproducibility, simulating a governance-compliant workflow often required in regulated industrial environments.

## 📂 File Structure
```
├── manufacturing_defect_prediction.py   # Main script
├── README.md                            # Project overview
```

## 💡 Future Improvements
- Replace synthetic data with real factory sensor datasets
- Add SHAP-based explainability
- Build a Streamlit dashboard for real-time monitoring
- Integrate with a production-grade CI/CD pipeline

---

**Author**: Sakshi Naik  
**Role**: Data Analytics Leader | ML | MLOps | Governance | Manufacturing Applications
