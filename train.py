import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def load_data(filepath='Data/Processed_data15.csv', sample_size=600000):
    """Load and clean flight data"""
    logger.info(f"Loading data from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} records")
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    
    # Required columns
    required_cols = ['ARR_DELAY', 'AIRLINE', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'FL_DATE', 'CANCELLED']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean data
    df = df.dropna(subset=['ARR_DELAY', 'AIRLINE', 'ORIGIN', 'DEST', 'CRS_DEP_TIME'])
    
    # Sample if needed
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled to {len(df):,} records")
    
    return df

def engineer_features(df):
    """Create model features"""
    logger.info("Starting feature engineering...")
    
    # Date features
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['month'] = df['FL_DATE'].dt.month
    df['day_of_week'] = df['FL_DATE'].dt.dayofweek
    df['departure_hour'] = (df['CRS_DEP_TIME'] // 100).astype(int).clip(0, 23)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Holiday periods
    df['is_holiday_period'] = df.apply(
        lambda x: 1 if (
            (x['FL_DATE'].month == 12 and x['FL_DATE'].day >= 20) or
            (x['FL_DATE'].month == 1 and x['FL_DATE'].day <= 5) or
            (x['FL_DATE'].month == 11 and 22 <= x['FL_DATE'].day <= 29) or
            (x['FL_DATE'].month == 7 and 1 <= x['FL_DATE'].day <= 7)
        ) else 0,
        axis=1
    )
    
    # Target variables
    df['is_delayed'] = (df['ARR_DELAY'] > 15).astype(int)
    df['is_cancelled'] = df['CANCELLED'].astype(int)
    
    # Aggregate statistics
    route_freq = df.groupby(['ORIGIN', 'DEST']).size().to_dict()
    airline_delay = df.groupby('AIRLINE')['ARR_DELAY'].mean().to_dict()
    
    df['route_frequency'] = df.apply(
        lambda row: route_freq.get((row['ORIGIN'], row['DEST']), 0),
        axis=1
    )
    df['airline_delay_history'] = df['AIRLINE'].map(airline_delay).fillna(0)
    
    logger.info("Feature engineering complete")
    return df, route_freq, airline_delay

def prepare_features(df):
    """Encode and scale features"""
    logger.info("Preparing features...")
    
    # Initialize encoders
    encoders = {}
    for col in ['AIRLINE', 'ORIGIN', 'DEST']:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col])
    
    # Define feature columns
    feature_columns = [
        'month',
        'day_of_week',
        'departure_hour',
        'is_weekend',
        'is_holiday_period',
        'route_frequency',
        'airline_delay_history',
        'AIRLINE_encoded',
        'ORIGIN_encoded',
        'DEST_encoded'
    ]
    
    # Verify all features exist
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Scale features
    scaler = StandardScaler()
    X = df[feature_columns]
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Final feature set: {len(feature_columns)} features")
    return X_scaled, feature_columns, encoders, scaler

def train_model(X, y, model_name):
    """Train classification model with SMOTE"""
    logger.info(f"Training {model_name} model...")
    
    # Check class distribution
    class_dist = pd.Series(y).value_counts().to_dict()
    logger.info(f"Class distribution: {class_dist}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Apply SMOTE for imbalanced data
    minority_class_ratio = min(class_dist.values()) / sum(class_dist.values())
    if minority_class_ratio < 0.15:
        logger.info("Applying SMOTE for class imbalance...")
        min_neighbors = min(5, (y_train == min(class_dist.keys())).sum() - 1)
        smote = SMOTE(random_state=42, k_neighbors=max(1, min_neighbors))
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info(f"{model_name} AUC: {auc:.3f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return model, auc

def create_chart_data(df):
    """Create visualization data"""
    logger.info("Creating chart data...")
    
    charts_data = {
        'delay_by_hour': df.groupby('departure_hour')['is_delayed'].mean().to_dict(),
        'delay_by_airline': df.groupby('AIRLINE')['is_delayed'].mean().to_dict(),
        'delay_by_month': df.groupby('month')['is_delayed'].mean().to_dict(),
        'cancel_by_airline': df.groupby('AIRLINE')['is_cancelled'].mean().to_dict(),
        'delay_by_day': df.groupby('day_of_week')['is_delayed'].mean().to_dict()
    }
    
    return charts_data

def main():
    """Main training pipeline"""
    try:
        # Load data
        df = load_data()
        
        # Engineer features
        df, route_freq, airline_delay = engineer_features(df)
        
        # Prepare features
        X, feature_columns, encoders, scaler = prepare_features(df)
        
        # Train models
        logger.info("\n" + "="*50)
        logger.info("TRAINING DELAY PREDICTION MODEL")
        logger.info("="*50)
        delay_model, delay_auc = train_model(X, df['is_delayed'], 'DELAY')
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING CANCELLATION PREDICTION MODEL")
        logger.info("="*50)
        cancel_model, cancel_auc = train_model(X, df['is_cancelled'], 'CANCELLATION')
        
        # Create chart data
        charts_data = create_chart_data(df)
        
        # Save artifacts
        logger.info("\nSaving artifacts...")
        artifacts = {
            'delay_model': delay_model,
            'cancel_model': cancel_model,
            'scaler': scaler,
            'encoders': encoders,
            'features': feature_columns,
            'route_frequency_map': route_freq,
            'airline_avg_delay_map': airline_delay,
            'charts_data': charts_data,
            'model_performance': {
                'delay_auc': delay_auc,
                'cancel_auc': cancel_auc
            }
        }
        
        joblib.dump(artifacts, 'prediction_artifacts.pkl')
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*50)
        logger.info(f"Delay Model AUC: {delay_auc:.3f}")
        logger.info(f"Cancellation Model AUC: {cancel_auc:.3f}")
        logger.info("Artifacts saved to: prediction_artifacts.pkl")
        logger.info("\nNext steps:")
        logger.info("1. Set environment variables for API keys")
        logger.info("2. Run: python app.py")
        logger.info("3. Open: http://localhost:5000")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()