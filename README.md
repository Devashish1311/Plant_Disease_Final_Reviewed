# LeafLab: Plant Disease Identification Using Handcrafted Image Features

ES 335 undergraduate project comparing handcrafted colour and texture features for three potato-leaf classes from PlantVillage: Early blight, Healthy, and Late blight.

## Project question

Which handcrafted image representation—HSV colour statistics, GLCM texture, LBP texture, or their combination—works best with traditional classifiers for this three-class task?

## Dataset and observed class counts

The project uses the public [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset). The run included 2,152 unique, readable potato images:

| Class | Images |
|---|---:|
| Early blight | 1,000 |
| Healthy | 152 |
| Late blight | 1,000 |

One observation is one image; the target label comes from its PlantVillage class folder. The healthy class is substantially underrepresented. Images are resized to 128 × 128. Exact byte-identical files are removed using SHA-1 hashes; this does not guarantee removal of all near-duplicates.

## Feature extraction

- **Colour (54 features):** for each HSV channel, mean, standard deviation and a 16-bin histogram.
- **GLCM (48 features):** contrast, dissimilarity, homogeneity, energy, correlation and ASM for two distances and four angles.
- **LBP (10 features):** uniform Local Binary Pattern histogram, P=8 and R=1.
- **Combined (112 features):** concatenation of all three groups.

## Models and evaluation

- Majority-class baseline.
- Logistic Regression.
- RBF Support Vector Machine.
- Random Forest.

The experiment uses an 80/20 stratified train/test split with random seed 42. Hyperparameter selection uses stratified 5-fold cross-validation on the training set. StandardScaler is inside the Logistic Regression and SVM pipelines to avoid fitting scaling parameters on validation/test data. Metrics include accuracy, macro-F1, macro precision, macro recall and confusion matrices.

The current exported run found the highest test macro-F1 for combined features + Random Forest (0.9887), while combined features + SVM had the highest test accuracy (0.9930). These are results on this particular PlantVillage split, not a claim of field-ready disease diagnosis.

## Reproduce the experiment

### Google Colab or local Python

1. Download/clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download PlantVillage. The experiment script recursively searches under `data/PlantVillage` for folders named `Potato___Early_blight`, `Potato___Late_blight`, and `Potato___healthy`.

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage
```

4. Run the complete experiment:

```bash
python src/run_experiments.py --data-dir data/PlantVillage
```

Optional quick smoke test (not for final reported results):

```bash
python src/run_experiments.py --data-dir data/PlantVillage --max-per-class 100
```

5. The script writes `outputs/results.json`, `outputs/metrics.csv`, `outputs/class_counts.csv`, and confusion matrices under `outputs/confusion_matrices/`. Copy the updated JSON into the website folder:

```bash
cp outputs/results.json site/results.json
```

On Windows PowerShell:

```powershell
Copy-Item outputs/results.json site/results.json
```

## Website and GitHub Pages

The static website is in `site/`. It reads `site/results.json` and provides interactive feature-set/classifier selectors, metric cards, macro-F1 comparisons, confusion matrices and per-class precision/recall.

**GitHub Pages does not support selecting an arbitrary `/site` folder as the publishing source with the branch-based deployment option.** This repository includes `.github/workflows/pages.yml`, which deploys the `site/` directory using GitHub Actions.

1. Push the repository to GitHub.
2. Open repository **Settings → Pages**.
3. Under **Build and deployment → Source**, select **GitHub Actions**.
4. Push to the `main` branch (or manually run the workflow from the Actions tab).
5. Wait for the Pages deployment workflow to finish. The published URL will be shown in the workflow/Pages settings.

The website can also be previewed locally from the repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/site/` in a browser. A local HTTP server is needed because the page fetches `results.json`.

## Project limitations

- PlantVillage has controlled image conditions; performance may not transfer to field images.
- The healthy class has only 152 images, so class-wise metrics and the confusion matrix should be examined carefully.
- Exact duplicate detection does not guarantee removal of near-duplicates.
- Global image features do not localise lesions.
- No independent farm-image test set was used.

## AI-use disclosure

Adapt this disclosure to match actual use and course policy:

Generative AI (ChatGPT) was used as an assisting tool during this project for brainstorming and refining the project approach, developing parts of the Python implementation, debugging code, and preparing the project website and documentation. I selected and worked with the dataset, executed the experiments, reviewed the resulting outputs, and is responsible for the analysis, interpretation, and conclusions presented in this project. AI-generated suggestions and code were reviewed and adapted as needed.

## References

- PlantVillage dataset: https://github.com/spMohanty/PlantVillage-Dataset
- scikit-learn documentation: https://scikit-learn.org/stable/
- scikit-image feature documentation: https://scikit-image.org/docs/stable/api/skimage.feature.html

Check and cite the dataset repository's current licence/terms in the final submission.
