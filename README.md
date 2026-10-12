# Food Preservation Safety

An interactive Streamlit project for exploring food-storage patterns and
estimating preservation outcomes. The repository includes a 200,000-row
synthetic dataset, data generation, analytics, model training and evaluation,
and a saved scikit-learn prediction pipeline.

## Project overview

1. `generate_dataset.py` creates records for eight food categories from
   documented simulation assumptions.
2. `analytics.py` calculates dataset summaries and grouped preservation rates;
   `visualizations.py` turns them into interactive Plotly charts.
3. `model_training.py` validates the data, trains classification pipelines,
   compares them with training-only cross-validation, and evaluates them on a
   held-out split. The selected model prioritizes F1 for the Not Preserved
   class.
4. `app.py` provides Dashboard, Predict, Analytics, Model Performance, and
   Project Guide pages.

The prediction pipeline uses ten input features: food type, temperature,
humidity, storage duration, pH, packaging, storage condition, microbial count,
preservative use, and days remaining on the stated shelf life. Categorical
features are one-hot encoded; numeric features are scaled. The target is the
simulated Preserved / Not Preserved label. The preservation score is used in
analytics but excluded from model training.

## Run the dashboard

The bundled model was saved with scikit-learn 1.9.0. Use Python 3.13 (or another
Python version supported by scikit-learn 1.9.0) so the model and runtime use the
same scikit-learn version.

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit, usually <http://localhost:8501>.

## Train the model

Training also requires Java 17 or 21 and `requirements-training.txt`, which adds
PySpark and Matplotlib. Dataset regeneration is optional and overwrites the
bundled CSV:

```bash
python -m pip install -r requirements-training.txt
python model_training.py
```

The training script writes the model, evaluation reports, and plots beside the
script.

```bash
python generate_dataset.py
```

The model file is stored with Git LFS because it exceeds GitHub's regular file
size limit. Install Git LFS before cloning so the model is downloaded along
with the source files.

## Deploy on Streamlit Community Cloud

After the project is pushed to GitHub:

1. Connect your GitHub account at <https://share.streamlit.io/>.
2. Create an app using repository `AditiJogad/Food-preservation`, branch
   `main`, and entrypoint `app.py`.
3. In Advanced settings, select Python 3.13 to match the saved model's
   scikit-learn version.
4. Deploy. Community Cloud installs the app dependencies from the root
   `requirements.txt`; training-only PySpark is kept in
   `requirements-training.txt`.

The saved model uses Git LFS, which Streamlit Community Cloud supports.
