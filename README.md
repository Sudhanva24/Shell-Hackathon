# SHELL Hackathon Report

This report details the systematic methodology developed to predict ten key
properties of a five-component fluid mixture. The primary objective was to optimize a
leaderboard score derived from the Mean Absolute Percentage Error (MAPE). Through
an iterative process of feature engineering, advanced modeling, and robust data
cleaning, we successfully improved the initial baseline score of 88 to a final,
high-performing score of 94.01. The final solution leverages the TabPFN model, which
is enhanced by a pipeline that includes targeted outlier removal.

## 1. Initial Baseline and Foundational Feature Engineering

Our initial approach involved using a standard gradient-boosted model, LightGBM,
which established a baseline leaderboard score of 78. We transitioned from
tree-based models to a more advanced architecture: TabPFN (Tabular Prior-Fitted
Network) . This model was selected for its state-of-the-art performance on
small-to-medium tabular datasets without requiring extensive hyperparameter
tuning.The Score instantly rose to 88. While solid, this score indicated that the model
was not fully capturing the complex interactions within the fluid mixture.
To address this, we implemented a foundational feature engineering strategy. New
features were created by multiplying the fraction of each component by its
corresponding intrinsic properties.
- Technique: New Feature = Component_Fraction * Component_Property
- Impact: This step was highly effective, immediately boosting the score from 88 to
91.
- Rationale: This confirmed a critical hypothesis: the contribution of a
component's property to the final blend is directly proportional to its fraction in
the mixture. This initial success guided all subsequent modeling efforts toward
better capturing these interaction effects.

## 2. Transition to Advanced Modeling and Target Transformation

A crucial innovation at this stage was the transformation of the target variables. Given
that the evaluation metric was MAPE and the data contained negative values, a
standard logarithm was not applicable. We implemented a Shift-and-Log
Transformation.
- Technique: Transformed_Target = log(1 + Target + Shift), where Shift is a constant
calculated to make all target values positive.
- Rationale: This transformation reshapes the target distribution, forcing the
model to minimize relative errors rather than absolute errors, which aligns
perfectly with the MAPE metric's sensitivity. It proved to be a vital component for
stabilizing the model and improving predictive accuracy.

## 3. Data Cleaning via Outlier Removal

The most significant breakthrough came from robust data cleaning. We hypothesized
that certain data points in the training set were anomalous and were negatively
impacting the model's ability to generalize. We used the Isolation Forest algorithm to
identify and remove these outliers.

- Technique: IsolationForest was applied to the training data before any
transformations or model fitting.
- Tuning: The contamination hyperparameter, which represents the expected
proportion of outliers, was carefully tuned. Through systematic experimentation, a
value of 0.1 was identified as optimal, indicating that a significant portion of the
data was considered anomalous.
- Impact: This single step was the key to our final performance gain, elevating the
score from the low 90s to our final score of 94. It demonstrated that providing the
TabPFN model with a cleaner, more consistent dataset was paramount.

## 4. Final Methodology Summary

The final, high-performing pipeline is a multi-stage process that combines the
successful elements from each phase of development:
- Feature Set: The model is trained on the engineered feature set, which includes
the original data plus the foundational Fraction * Property interaction terms.
- Data Cleaning: The IsolationForest algorithm is applied to the training data with
a contamination rate of 0.1 to remove outliers.
- Target Transformation: The target variables (y_
train) are transformed using the
Shift-and-Log method to prepare them for MAPE-optimized training.
- Modeling: A MultiOutputRegressor wrapping the TabPFNRegressor (with
device='cuda') is trained on the cleaned features and transformed targets.
- Prediction: The trained model generates predictions, which are then
inverse-transformed (exp(pred) - 1 - shift) to be returned to their original scale
for evaluation.

## 5. Conclusion

The successful outcome was not the result of a single technique but rather a
systematic and iterative refinement of the entire modeling pipeline. The journey from a
score of 88 to 94 underscores the importance of a layered approach: starting with
insightful feature engineering, moving to an appropriate and powerful model, and
finally, achieving peak performance through meticulous data cleaning and
preprocessing.
