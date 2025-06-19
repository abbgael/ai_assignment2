# Disease Outbreak Prediction System for SDG 3 (Good Health and Well-being)
# AI-driven solution to predict disease outbreaks using machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DiseaseOutbreakPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        
    def generate_synthetic_data(self, n_samples=2000):
        """
        Generate synthetic disease outbreak data for demonstration
        In a real project, you would load actual health data from WHO, CDC, or other sources
        """
        print("Generating synthetic disease outbreak data...")
        
        # Generate features that could influence disease outbreaks
        data = {
            'population_density': np.random.normal(1000, 500, n_samples),
            'temperature': np.random.normal(25, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall': np.random.exponential(50, n_samples),
            'sanitation_index': np.random.uniform(0, 100, n_samples),
            'healthcare_access': np.random.uniform(0, 100, n_samples),
            'vaccination_rate': np.random.uniform(40, 95, n_samples),
            'poverty_rate': np.random.uniform(5, 60, n_samples),
            'water_quality_index': np.random.uniform(20, 100, n_samples),
            'previous_outbreak_count': np.random.poisson(2, n_samples),
            'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
            'urban_rural': np.random.choice(['Urban', 'Rural'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create outbreak probability based on realistic factors
        outbreak_prob = (
            0.3 * (df['population_density'] > 1200).astype(int) +
            0.2 * (df['sanitation_index'] < 50).astype(int) +
            0.2 * (df['healthcare_access'] < 60).astype(int) +
            0.15 * (df['vaccination_rate'] < 70).astype(int) +
            0.1 * (df['poverty_rate'] > 40).astype(int) +
            0.05 * (df['water_quality_index'] < 60).astype(int)
        )
        
        # Add some randomness and create binary outcome
        outbreak_prob += np.random.normal(0, 0.1, n_samples)
        df['outbreak'] = (outbreak_prob > 0.4).astype(int)
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data for machine learning
        """
        print("Preprocessing data...")
        
        # Handle categorical variables
        df_processed = df.copy()
        df_processed = pd.get_dummies(df_processed, columns=['season', 'urban_rural'], drop_first=True)
        
        # Separate features and target
        X = df_processed.drop('outbreak', axis=1)
        y = df_processed['outbreak']
        
        # Handle missing values (if any)
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models and compare performance
        """
        print("Training multiple models...")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        model_scores = {}
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            model_scores[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'model': model
            }
            print(f"{name}: CV AUC = {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['mean_cv_score'])
        self.model = model_scores[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        
        # Train the best model
        self.model.fit(X_train, y_train)
        
        return model_scores
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        print("Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Test AUC Score: {auc_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return y_pred, y_pred_proba, auc_score
    
    def visualize_results(self, X_test, y_test, y_pred_proba):
        """
        Create visualizations for the results
        """
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {auc_score:.2f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve - Disease Outbreak Prediction')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True)
        
        # Feature Importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[0, 1].barh(importance_df['feature'], importance_df['importance'])
            axes[0, 1].set_title('Feature Importance')
            axes[0, 1].set_xlabel('Importance')
        
        # Prediction Distribution
        axes[1, 0].hist(y_pred_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Distribution of Outbreak Probabilities')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        y_pred = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Outbreak', 'Outbreak'],
                   yticklabels=['No Outbreak', 'Outbreak'], ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('disease_outbreak_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_outbreak_risk(self, input_data):
        """
        Predict outbreak risk for new data
        """
        # Preprocess input data
        input_scaled = self.scaler.transform(input_data)
        
        # Predict probability
        risk_probability = self.model.predict_proba(input_scaled)[:, 1]
        
        return risk_probability
    
    def ethical_analysis(self):
        """
        Discuss ethical considerations and potential biases
        """
        print("\n" + "="*60)
        print("ETHICAL ANALYSIS & BIAS CONSIDERATIONS")
        print("="*60)
        
        ethical_considerations = """
        1. DATA BIAS CONCERNS:
           - Historical health data may be biased toward certain populations
           - Underreporting in rural or marginalized communities
           - Socioeconomic factors may create systematic biases
        
        2. FAIRNESS & EQUITY:
           - Model should not discriminate against vulnerable populations
           - Equal access to early warning benefits across all communities
           - Consider healthcare infrastructure disparities
        
        3. PRIVACY & SECURITY:
           - Health data requires strict privacy protection
           - Anonymization and secure data handling essential
           - Compliance with healthcare data regulations (HIPAA, GDPR)
        
        4. TRANSPARENCY & ACCOUNTABILITY:
           - Clear explanation of model decisions for health officials
           - Regular model auditing and performance monitoring
           - Human oversight in critical health decisions
        
        5. POTENTIAL POSITIVE IMPACTS:
           - Early intervention can save lives
           - Resource allocation optimization
           - Reduced healthcare costs through prevention
           - Global health security improvement
        
        6. MITIGATION STRATEGIES:
           - Regular bias testing and model retraining
           - Diverse and representative training data
           - Community engagement in model development
           - Continuous monitoring of real-world outcomes
        """
        
        print(ethical_considerations)

def main():
    """
    Main function to run the complete disease outbreak prediction pipeline
    """
    print("="*80)
    print("DISEASE OUTBREAK PREDICTION SYSTEM FOR SDG 3")
    print("AI-Driven Solution for Global Health Security")
    print("="*80)
    
    # Initialize predictor
    predictor = DiseaseOutbreakPredictor()
    
    # Generate synthetic data (replace with real data in production)
    df = predictor.generate_synthetic_data(n_samples=2000)
    
    # Display data info
    print(f"\nDataset shape: {df.shape}")
    print(f"Outbreak cases: {df['outbreak'].sum()} ({df['outbreak'].mean()*100:.1f}%)")
    print("\nDataset overview:")
    print(df.head())
    
    # Preprocess data
    X, y = predictor.preprocess_data(df)
    
    # Scale features
    X_scaled = predictor.scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    model_scores = predictor.train_models(X_train, y_train)
    
    # Evaluate model
    y_pred, y_pred_proba, auc_score = predictor.evaluate_model(X_test, y_test)
    
    # Create visualizations
    predictor.visualize_results(X_test, y_test, y_pred_proba)
    
    # Example prediction for a new region
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION FOR A NEW REGION")
    print("="*60)
    
    # Create example input (high-risk scenario)
    example_input = pd.DataFrame({
        'population_density': [1500],
        'temperature': [28],
        'humidity': [85],
        'rainfall': [120],
        'sanitation_index': [30],
        'healthcare_access': [40],
        'vaccination_rate': [55],
        'poverty_rate': [50],
        'water_quality_index': [35],
        'previous_outbreak_count': [3],
        'season_Spring': [0],
        'season_Summer': [1],
        'season_Winter': [0],
        'urban_rural_Urban': [1]
    })
    
    risk_prob = predictor.predict_outbreak_risk(example_input)[0]
    print(f"Outbreak Risk Probability: {risk_prob:.3f} ({risk_prob*100:.1f}%)")
    
    if risk_prob > 0.7:
        print("⚠️  HIGH RISK: Immediate intervention recommended")
    elif risk_prob > 0.4:
        print("⚡ MODERATE RISK: Enhanced monitoring advised")
    else:
        print("✅ LOW RISK: Continue routine surveillance")
    
    # Ethical analysis
    predictor.ethical_analysis()
    
    print("\n" + "="*80)
    print("PROJECT IMPACT ON SDG 3 (GOOD HEALTH AND WELL-BEING)")
    print("="*80)
    
    impact_summary = """
    This AI-driven disease outbreak prediction system contributes to SDG 3 by:
    
    🎯 TARGET 3.3: Combat communicable diseases
       - Early warning system for epidemic prevention
       - Proactive public health response
    
    🎯 TARGET 3.D: Strengthen health emergency preparedness
       - Predictive analytics for resource allocation
       - Risk assessment for vulnerable populations
    
    📊 MEASURABLE OUTCOMES:
       - Reduced outbreak response time
       - Optimized healthcare resource distribution
       - Improved population health surveillance
       - Enhanced global health security
    
    🌍 SCALABILITY:
       - Adaptable to different diseases and regions
       - Integration with existing health systems
       - Real-time monitoring capabilities
    """
    
    print(impact_summary)
    
    print("\nProject completed successfully! 🎉")
    print("Next steps: Deploy as web application, integrate real-time data sources")

if __name__ == "__main__":
    main()
