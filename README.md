# Statistical Analysis of Vehicle Energy Consumption

## 📌 Project Overview
This project applies statistical inference techniques to a large-scale real-world dataset to investigate fuel consumption patterns across three categories of vehicle powertrains: Internal Combustion Engine (ICE), Hybrid Electric Vehicle (HEV), and Plug-in Hybrid Electric Vehicle (PHEV).Utilizing Mass Air Flow (MAF, measured in grams per second) as the primary proxy for instantaneous fuel consumption, the study quantifies differences in energy efficiency and evaluates the impact of driving behavior on overall fuel usage.

This project was developed as part of the Statistical Inference course at IIT Kharagpur.

## 📊 Dataset
The analysis utilizes the **Vehicle Energy Dataset (VED)**, originally collected by the University of Michigan Transportation Research Institute (UMTRI). It captures second-by-second OBD-II (On-Board Diagnostics) signals from a diverse fleet of vehicles driven across urban, suburban, and highway routes. 

* **Dataset Link:** [VED Segregated on Kaggle](https://www.kaggle.com/datasets/yashseth25/ved-segregated)

## 🎯 Problem Statements
1. **Comparative Efficiency of Vehicle Types:** To determine whether different powertrain architectures (ICE, HEV, PHEV) exhibit statistically significant differences in mean fuel consumption, and to establish a formal efficiency ranking among them.
2. **Impact of Driving Behaviour:** To evaluate whether aggressive driving behavior—characterized by high acceleration patterns reflected in elevated MAF and Engine RPM values—leads to a statistically significant increase in energy consumption compared to smooth driving.

## 🛠️ Methodology
To ensure structural integrity, the continuous data was engineered to classify driving styles, where observations with RPM > 2,000 or MAF > dataset mean were labeled "Aggressive" and the rest as "Smooth". The statistical framework consists of:

* **Welch's Two-Sample t-Test:** Selected over the standard equal-variance t-test due to extremely large sample sizes, unequal group sizes, heterogeneous variances, and the presence of skewness and outliers in the data.
* **Chi-Square Test of Independence:** Implemented to complement the parametric analysis non-parametrically by discretizing MAF into Low, Medium, and High consumption bins and cross-tabulating against Engine Type.
* **Kolmogorov-Smirnov (KS) Test:** Implemented in the codebase to evaluate distributional robustness across powertrain variants.

## 🚀 Key Findings
All hypotheses tested were supported at the 0.01% significance level (p < 0.001).

* **Powertrain Efficiency Hierarchy:** The results establish a structural and distributionally consistent efficiency hierarchy: `PHEV < HEV < ICE`.
* **Effect Sizes:** ICE vehicles consume approximately 23% more fuel than HEVs, HEVs consume 47% more than PHEVs, and ICE vehicles consume 59% more fuel than PHEVs overall.
* **Driving Behavior:** Aggressive driving increases consumption by a factor of approximately six compared to smooth driving. The absolute difference in consumption between driving styles dwarfs the differences between powertrains, making human behavior the strongest single determinant of real-world fuel usage. 

## 💻 Codebase & Execution

The analysis is driven by `main.py`, which handles data preprocessing, feature engineering, statistical testing, and visualization using `pandas`, `numpy`, `scipy.stats`, and `matplotlib`.

### Dependencies
Ensure you have the following libraries installed before running the script:
```bash
pip install pandas numpy scipy matplotlib