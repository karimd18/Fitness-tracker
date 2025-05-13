# Fitness Tracker

An exploration into the realm of strength training analytics, leveraging wristband accelerometer and gyroscope data. This project harnesses modern **AI/ML technologies**—including Support Vector Machines (SVM), Neural Networks (MLPClassifier), Random Forests, Decision Trees, K-Means clustering, PCA, Butterworth filtering, and outlier detection techniques (Chauvenet’s criterion, Local Outlier Factor)—to classify exercises, count repetitions, and detect improper form, aiming to create a digital personal trainer experience.

<img src="https://images.unsplash.com/photo-1519222970733-f546218fa6d3?auto=format&fit=crop&w=1000&q=80" alt="wristband sensor" width="100%">

# Table of Contents
- [Introduction, Goal, and Context](#part-1)
- [Data Processing](#part-2)
- [Outlier Handling](#part-3)
- [Feature Engineering](#part-4)
- [Predictive Modeling and Repetition Counting](#part-5)
- [Conclusion](#part-6)

# Introduction <a id="part-1"></a>

In the past decade, breakthroughs in sensor technology have made wearable devices like accelerometers, gyroscopes, and GPS receivers more feasible and accessible. Such advancements have propelled the monitoring and classification of human activities to the forefront of pattern recognition and machine learning research. This is majorly due to the immense commercial potential of context-aware applications and evolving user interfaces. Beyond commerce, there's a broader societal impact: addressing challenges related to rehabilitation, sustainability, elderly care, and health.

Historically, the focus was largely on tracking aerobic exercises. Systems existed to monitor running pace, track exertion, and even automate some functionalities of exercise machines. However, the domain of free weight exercises remained relatively uncharted. There's a notable gap: while aerobic exercises have been well-addressed by wearables, strength training—a crucial component of a balanced fitness regime—hasn’t been explored to its full potential.

Digital personal trainers might soon be a reality, with advancements in context-aware AI. While there have been significant strides towards this future, there remains a vital component yet unaddressed: tracking workouts effectively and ensuring safety and progress.

Inspired by Dave Ebbelaar and his classmates at Vrije Universiteit Amsterdam, this project sets its sight on a niche yet profound aspect of fitness technology. By tapping into the potential of the strength training arena and leveraging wristband accelerometer and gyroscope data, the foundation is formed. This pivotal data, amassed during free weight workouts from five distinct participants, offers invaluable insights. The paramount ambition of this endeavor? To architect AI-driven models—Support Vector Machines, Neural Networks, Random Forests—that emulate the precision and expertise of human personal trainers: models adept at tracking exercises, enumerating repetitions, and discerning improper form. Here is the [dataset](https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/tree/main/data/raw).

---

# Data Processing <a id="part-2"></a>

## Data Collection

This study uses MbientLab’s wristband sensor research kit, simulating a smartwatch’s placement and orientation. The accelerometer was sampled at **12.5 Hz** and the gyroscope at **25 Hz**. Five participants (Table 1) performed bench press, squat, row, overhead press (OHP), and deadlift in both 3×5 reps (heavy) and 3×10 reps (medium) to evaluate model generalization. ‘Rest’ data was collected between sets.

| Participant | Gender | Age | Weight (kg) | Height (cm) | Experience (years) |
|-------------|--------|-----|-------------|-------------|--------------------|
| A           | Male   | 23  | 95          | 194         | 5+                 |
| B           | Male   | 24  | 76          | 183         | 5+                 |
| C           | Male   | 16  | 65          | 181         | < 1                |
| D           | Male   | 21  | 85          | 197         | 3                  |
| E           | Female | 20  | 58          | 165         | 1                  |

*Table 1. Participant demographics (N=5).*

## Preparing the Dataset

- **File ingestion & parsing**: Read all CSVs in `data/raw/MetaMotion`, extract metadata (participant, exercise label, category) from filenames.
- **Concatenation**: Build separate DataFrames for accelerometer and gyroscope streams.
- **Timestamping**: Convert epoch milliseconds to a `DatetimeIndex`.
- **Merging & Resampling**: Merge accelerometer & gyroscope on time, resample to **200 ms** windows (mean aggregation) for synchronized analysis.
- **Export**: Save the processed time series to `01_data_processed.pkl`.

<table>
  <tr>
    <td><img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/d2e278cd-811b-4828-8afb-15e551497641" height="300" width="300"></td>
    <td><img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/3d62c5cc-6e74-44a3-a82f-e29bbb963555" height="300" width="300"></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center">Figure 1: Data ingestion & resampling</td>
  </tr>
</table>

---

# Outlier Handling <a id="part-3"></a>

We evaluated three methods to detect and clean anomalies:

- **Interquartile Range (IQR)**: Marks points outside [Q1−1.5 IQR, Q3+1.5 IQR].  
- **Chauvenet’s Criterion**: Assumes normality; flags points with extremely low probability.  
- **Local Outlier Factor (LOF)**: Density-based detection via nearest-neighbors.

Outliers are replaced with `NaN` and then interpolated. Cleaned data is saved to `02_outliers_removed_chauvenets.pkl`.

<table>
  <tr>
    <td><img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/0e38d223-5800-4324-b459-4a91fb7e77c0" height="250"></td>
    <td><img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/29b8a40f-2cb1-41a8-a41d-bdaf6a3f8efe" height="250"></td>
  </tr>
  <tr>
    <td align="center">Chauvenet’s Criterion</td>
    <td align="center">Local Outlier Factor</td>
  </tr>
</table>

---

# Feature Engineering <a id="part-4"></a>

1. **Imputation & Duration**: Interpolate missing values, compute set durations.  
2. **Butterworth Low‐Pass Filter**: Remove high-frequency noise.  
3. **Principal Component Analysis (PCA)**: Reduce six raw axes to three orthogonal components.  
4. **Vector Magnitudes**: Compute `acc_r` and `gyr_r`.  
5. **Temporal Abstraction**: Rolling‐window mean & std (window = 5 samples).  
6. **Frequency Abstraction**: FFT-based features (dominant freq, weighted freq, spectral entropy) with a 2 s window.  
7. **Clustering**: K-Means (K=5) on accelerometer axes for unsupervised pattern discovery.  
8. **Export**: Save engineered features to `03_data_features.pkl`.

![clustering](https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/db0069ef-7bba-4281-9233-f8eac8bc183d)
*Figure: K-Means clustering in PCA-reduced space.*

---

# Predictive Modeling & Repetition Counting <a id="part-5"></a>

## Classification Models

- **Feature sets**: Basic (filtered axes), Magnitudes, PCA, Temporal, Frequency, Cluster.  
- **Algorithms**:  
  - **Support Vector Machine (SVM)** with RBF & linear kernels  
  - **Feed‐forward Neural Network** (MLPClassifier)  
  - **Random Forest**  
  - **Decision Tree**  
  - **Naive Bayes**  

We used **GridSearchCV** for hyperparameter tuning and **forward feature selection** to find the optimal subset. The Random Forest achieved **98.5% accuracy** on unseen test sets. Confusion matrices were used to analyze misclassifications, especially between similar lifts (bench vs. OHP).

## Repetition Counting

A simple **peak detection** on low-pass filtered magnitude signals (`acc_r`, `gyr_r`) yields repetition counts. Customized cut-off thresholds per exercise delivered a **Mean Absolute Error (MAE)** of ≈ 1 rep across all sets.

---

# Conclusion <a id="part-6"></a>

This project demonstrates that **AI-driven analysis** of wristband IMU data can accurately classify barbell exercises (98%+ accuracy), count reps (≈ 95% accuracy), and detect form anomalies (98.5% on bench press). By integrating signal processing, dimensionality reduction, temporal/frequency abstraction, and a suite of ML models (SVMs, Neural Nets, Random Forests), we pave the way for a **digital personal trainer** embedded in everyday wearables.

Future work: expand to more exercises, collect diverse participant demographics, and integrate real‐time feedback loops.
