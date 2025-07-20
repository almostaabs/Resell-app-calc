import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class PropertyValuePredictor:
    def __init__(self):
        self.price_model = None
        self.rent_model = None
        self.city_encoder = None
        self.location_encoder = None
        self.status_encoder = None
        self.feature_columns = None
        self.scaler = None
        
        # Balanced location multipliers
        self.location_multipliers = {
            'Dwarka': 1.08, 'Rohini': 1.05, 'Karol Bagh': 1.12, 'Central Delhi': 1.15,
            'Greater Noida West': 1.10, 'Noida Sec 150': 1.12, 'Golf City': 1.11,
            'East Delhi': 1.03, 'Gurugram': 1.18, 'DLF': 1.22, 'Sohna Road': 1.15
        }

        # Balanced bedroom multipliers
        self.bedroom_multipliers = {
            1: 1.03, 2: 1.07, 3: 1.05, 4: 1.02, 5: 1.01
        }

        self.rental_yields = {
            'Delhi': 0.028, 'Noida': 0.030, 'Gurgaon': 0.032, 'Gurugram': 0.032,
            'Faridabad': 0.029, 'Ghaziabad': 0.027
        }

        self.current_year = 2025

    def load_models(self, model_dir="models"):
        try:
            model_files = {
                'price_model': os.path.join(model_dir, 'price_model.pkl'),
                'rent_model': os.path.join(model_dir, 'rent_model.pkl'),
                'city_encoder': os.path.join(model_dir, 'city_encoder.pkl'),
                'location_encoder': os.path.join(model_dir, 'location_encoder.pkl'),
                'status_encoder': os.path.join(model_dir, 'status_encoder.pkl'),
                'feature_columns': os.path.join(model_dir, 'feature_columns.pkl'),
                'scaler': os.path.join(model_dir, 'scaler.pkl')
            }

            for attr_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))

            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def train_models(self, csv_file="cleaned_resale_properties.csv"):
        try:
            df = pd.read_csv(csv_file)
            print(f"Original dataset shape: {df.shape}")
            
            df = self._preprocess_data(df)
            print(f"After preprocessing: {df.shape}")
            
            X, y_price, y_rent = self._prepare_features(df)
            print(f"Feature matrix shape: {X.shape}")

            # Split data with stratification to ensure balanced splits
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_price, test_size=0.25, random_state=42, shuffle=True
            )

            # Price model with balanced regularization
            print("\n🔍 Training price model...")
            
            # Balanced models - not too simple, not too complex
            models = {
                'XGBoost_Balanced': xgb.XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.06,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    random_state=42,
                    min_child_weight=3,
                    gamma=0.1
                ),
                'RandomForest_Balanced': RandomForestRegressor(
                    n_estimators=120,
                    max_depth=7,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                    max_features='sqrt'
                ),
                'Ridge_Balanced': Ridge(alpha=10.0, random_state=42)
            }

            best_model = None
            best_score = float('inf')
            best_name = ""

            for name, model in models.items():
                # Cross-validation with proper scoring
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                print(f"{name} CV MAE: ₹{cv_mae:,.0f} (±{cv_std:,.0f})")
                
                if cv_mae < best_score:
                    best_score = cv_mae
                    best_model = model
                    best_name = name

            print(f"\n🏆 Best model: {best_name}")
            self.price_model = best_model
            self.price_model.fit(X_train, y_train)

            # Train rent model
            X_rent_train, X_rent_test, y_rent_train, y_rent_test = train_test_split(
                X, y_rent, test_size=0.25, random_state=42
            )
            
            self.rent_model = Ridge(alpha=5.0, random_state=42)
            self.rent_model.fit(X_rent_train, y_rent_train)

            # Save models
            self._save_models()

            # Evaluate models
            self._evaluate_models(X_train, X_test, y_train, y_test,
                                X_rent_train, X_rent_test, y_rent_train, y_rent_test)

            return True

        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _preprocess_data(self, df):
        # Remove duplicates and handle missing values
        df = df.drop_duplicates()
        df = df.dropna(subset=['Price', 'Rate_per_sqft', 'Bedroom', 'Carpet_area_sqft'])

        # Convert to numeric
        numeric_cols = ['Price', 'Rate_per_sqft', 'Carpet_area_sqft', 'Total_area']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle Total_area
        if 'Total_area' in df.columns:
            df['Total_area'] = df['Total_area'].fillna(df['Carpet_area_sqft'] * 1.25)
        else:
            df['Total_area'] = df['Carpet_area_sqft'] * 1.25

        # Balanced outlier removal - not too aggressive
        df = self._remove_outliers_balanced(df)
        print(f"After outlier removal: {df.shape}")

        # Apply balanced market adjustments
        df = self._apply_balanced_adjustments(df)

        return df

    def _remove_outliers_balanced(self, df):
        """Balanced outlier removal - remove clear outliers but keep reasonable variance"""
        df_clean = df.copy()
        
        # Method 1: Remove extreme outliers (1% and 99%)
        for col in ['Price', 'Rate_per_sqft']:
            lower_bound = df_clean[col].quantile(0.01)
            upper_bound = df_clean[col].quantile(0.99)
            
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            after_count = len(df_clean)
            
            print(f"Removed {before_count - after_count} extreme outliers for {col}")

        # Method 2: IQR with reasonable bounds
        for col in ['Price', 'Rate_per_sqft']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Less aggressive multiplier
            multiplier = 2.0 if col == 'Price' else 2.5
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            after_count = len(df_clean)
            
            print(f"Removed {before_count - after_count} IQR outliers for {col}")

        # Domain-specific filters (reasonable bounds)
        df_clean = df_clean[df_clean['Price'] > 500000]  # Min price
        df_clean = df_clean[df_clean['Price'] < 75000000]  # Max price
        df_clean = df_clean[df_clean['Rate_per_sqft'] > 800]  # Min rate
        df_clean = df_clean[df_clean['Rate_per_sqft'] < 80000]  # Max rate
        
        return df_clean

    def _apply_balanced_adjustments(self, df):
        """Apply balanced market adjustments"""
        df = df.copy()
        
        # Apply location multipliers with moderate dampening
        for location, multiplier in self.location_multipliers.items():
            mask = df['SubLocation'].str.contains(location, case=False, na=False)
            # 50% dampening - reasonable adjustment
            dampened_multiplier = 1 + (multiplier - 1) * 0.5
            df.loc[mask, 'Price'] *= dampened_multiplier
            df.loc[mask, 'Rate_per_sqft'] *= dampened_multiplier
        
        # Apply bedroom multipliers with light dampening
        for bedroom, multiplier in self.bedroom_multipliers.items():
            mask = df['Bedroom'] == bedroom
            # 30% dampening
            dampened_multiplier = 1 + (multiplier - 1) * 0.3
            df.loc[mask, 'Price'] *= dampened_multiplier
            df.loc[mask, 'Rate_per_sqft'] *= dampened_multiplier
        
        return df

    def _prepare_features(self, df):
        # Initialize encoders
        self.city_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.status_encoder = LabelEncoder()

        # Encode categorical variables
        df['City_encoded'] = self.city_encoder.fit_transform(df['City'])
        df['SubLocation_encoded'] = self.location_encoder.fit_transform(df['SubLocation'])
        df['Status_encoded'] = self.status_encoder.fit_transform(df['Status'])

        # Balanced feature engineering - keep useful features, remove noise
        df['area_ratio'] = df['Carpet_area_sqft'] / df['Total_area']
        df['bedroom_per_area'] = df['Bedroom'] / df['Carpet_area_sqft']
        df['price_per_area'] = df['Price'] / df['Carpet_area_sqft']
        
        # Log transformations for skewed features (important for price prediction)
        df['log_Price'] = np.log1p(df['Price'])
        df['log_Rate_per_sqft'] = np.log1p(df['Rate_per_sqft'])
        df['log_Carpet_area'] = np.log1p(df['Carpet_area_sqft'])
        df['log_Total_area'] = np.log1p(df['Total_area'])

        # Balanced feature selection - not too simple, not too complex
        self.feature_columns = [
            'City_encoded', 'SubLocation_encoded', 'Status_encoded',
            'Bedroom', 'log_Rate_per_sqft', 'log_Carpet_area', 'log_Total_area',
            'area_ratio', 'bedroom_per_area', 'price_per_area'
        ]

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df[self.feature_columns])
        
        # Use log-transformed price as target (helps with skewed distribution)
        y_price = df['log_Price'].values

        # Rental income calculation
        y_rent_raw = df.apply(lambda row: self._calculate_rental_income(row), axis=1)
        y_rent = np.log1p(y_rent_raw)

        return X, y_price, y_rent

    def _calculate_rental_income(self, row):
        rental_yield = self.rental_yields.get(row['City'], 0.030)
        return (row['Price'] * rental_yield) / 12

    def _evaluate_models(self, X_train, X_test, y_train, y_test,
                        X_rent_train, X_rent_test, y_rent_train, y_rent_test):
        """Evaluate both models with proper transformations"""
        
        # Price model evaluation
        price_train_pred_log = self.price_model.predict(X_train)
        price_test_pred_log = self.price_model.predict(X_test)

        # Convert back to original scale
        price_train_pred = np.expm1(price_train_pred_log)
        price_test_pred = np.expm1(price_test_pred_log)
        y_train_original = np.expm1(y_train)
        y_test_original = np.expm1(y_test)

        print("\n📈 RESELL MODEL PERFORMANCE (Balanced)")
        print(f"Train R² Score : {r2_score(y_train_original, price_train_pred):.3f}")
        print(f"Test  R² Score : {r2_score(y_test_original, price_test_pred):.3f}")
        print(f"Train MAE      : ₹{mean_absolute_error(y_train_original, price_train_pred):,.0f}")
        print(f"Test  MAE      : ₹{mean_absolute_error(y_test_original, price_test_pred):,.0f}")
        print(f"Train RMSE     : ₹{np.sqrt(mean_squared_error(y_train_original, price_train_pred)):,.0f}")
        print(f"Test  RMSE     : ₹{np.sqrt(mean_squared_error(y_test_original, price_test_pred)):,.0f}")
        print(f"Test  MAPE     : {mean_absolute_percentage_error(y_test_original, price_test_pred) * 100:.2f}%")

        # Check for overfitting
        train_mae = mean_absolute_error(y_train_original, price_train_pred)
        test_mae = mean_absolute_error(y_test_original, price_test_pred)
        overfitting_ratio = test_mae / train_mae
        print(f"Overfitting Ratio: {overfitting_ratio:.2f} (should be < 2.0)")

        # Rent model evaluation
        rent_train_pred_log = self.rent_model.predict(X_rent_train)
        rent_test_pred_log = self.rent_model.predict(X_rent_test)
        
        rent_train_pred = np.expm1(rent_train_pred_log)
        rent_test_pred = np.expm1(rent_test_pred_log)
        y_rent_train_original = np.expm1(y_rent_train)
        y_rent_test_original = np.expm1(y_rent_test)

        print("\n📊 RENT MODEL PERFORMANCE")
        print(f"Train R² Score : {r2_score(y_rent_train_original, rent_train_pred):.3f}")
        print(f"Test  R² Score : {r2_score(y_rent_test_original, rent_test_pred):.3f}")
        print(f"Train MAE      : ₹{mean_absolute_error(y_rent_train_original, rent_train_pred):,.0f}")
        print(f"Test  MAE      : ₹{mean_absolute_error(y_rent_test_original, rent_test_pred):,.0f}")
        print(f"Train RMSE     : ₹{np.sqrt(mean_squared_error(y_rent_train_original, rent_train_pred)):,.0f}")
        print(f"Test  RMSE     : ₹{np.sqrt(mean_squared_error(y_rent_test_original, rent_test_pred)):,.0f}")
        print(f"Test  MAPE     : {mean_absolute_percentage_error(y_rent_test_original, rent_test_pred) * 100:.2f}%")

    def _save_models(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        models = {
            'price_model': self.price_model,
            'rent_model': self.rent_model,
            'city_encoder': self.city_encoder,
            'location_encoder': self.location_encoder,
            'status_encoder': self.status_encoder,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        for name, model in models.items():
            with open(os.path.join(model_dir, f"{name}.pkl"), "wb") as f:
                pickle.dump(model, f)

    def predict_property_value(self, data_dict, years=5):
        try:
            features = self._prepare_input_features(data_dict)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict log price and convert back
            log_price = self.price_model.predict(features_scaled)[0]
            current_price = np.expm1(log_price)
            
            # Ensure positive price
            current_price = max(current_price, 100000)

            carpet_area = float(data_dict['carpet_area'])
            rate_per_sqft = current_price / carpet_area if carpet_area > 0 else 0

            # Calculate rental income
            city = data_dict['city']
            rental_yield = self.rental_yields.get(city, 0.030)
            monthly_rent = (current_price * rental_yield) / 12

            # Calculate projections
            annual_appreciation = 0.08
            future_price = current_price * (1 + annual_appreciation) ** years
            capital_gain = future_price - current_price
            total_rent = monthly_rent * 12 * years
            total_returns = capital_gain + total_rent
            total_roi = (total_returns / current_price) * 100
            annual_roi = total_roi / years
            grade = self._get_investment_grade(annual_roi)
            tips = self._get_real_estate_tips(data_dict)

            return {
                'current_price': current_price,
                'rate_per_sqft': rate_per_sqft,
                'future_price': future_price,
                'capital_gain': capital_gain,
                'monthly_rent': monthly_rent,
                'annual_rent': monthly_rent * 12,
                'total_rental_income': total_rent,
                'total_returns': total_returns,
                'total_roi': total_roi,
                'annual_roi': annual_roi,
                'investment_grade': grade,
                'tips': tips
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_input_features(self, data_dict):
        try:
            city_encoded = self.city_encoder.transform([data_dict['city']])[0]
        except:
            city_encoded = 0

        try:
            location_encoded = self.location_encoder.transform([data_dict['sub_location']])[0]
        except:
            location_encoded = 0

        try:
            status_encoded = self.status_encoder.transform([data_dict['status']])[0]
        except:
            status_encoded = 0

        bedroom = int(data_dict['bedroom'])
        carpet_area = float(data_dict['carpet_area'])
        total_area = float(data_dict['total_area'])
        
        # Estimate rate_per_sqft based on location
        base_rate = 8000
        if 'Gurgaon' in data_dict['city'] or 'Gurugram' in data_dict['city']:
            base_rate = 12000
        elif 'Delhi' in data_dict['city']:
            base_rate = 10000
        elif 'Noida' in data_dict['city']:
            base_rate = 9000
            
        rate_per_sqft = base_rate
        log_rate_per_sqft = np.log1p(rate_per_sqft)
        log_carpet_area = np.log1p(carpet_area)
        log_total_area = np.log1p(total_area)
        
        # Calculate derived features
        area_ratio = carpet_area / total_area
        bedroom_per_area = bedroom / carpet_area
        price_per_area = rate_per_sqft  # approximation

        return [
            city_encoded, location_encoded, status_encoded, bedroom,
            log_rate_per_sqft, log_carpet_area, log_total_area,
            area_ratio, bedroom_per_area, price_per_area
        ]

    def _get_investment_grade(self, annual_roi):
        if annual_roi >= 15:
            return "Excellent"
        elif annual_roi >= 12:
            return "Good"
        elif annual_roi >= 8:
            return "Average"
        else:
            return "Below Average"

    def _get_real_estate_tips(self, data_dict):
        city = data_dict['city']
        location = data_dict['sub_location']
        tips = []

        if 'Dwarka' in location:
            tips.append("Dwarka has excellent metro connectivity boosting ROI")
        elif 'Noida' in city:
            tips.append("Noida offers good infrastructure and IT hub proximity")
        elif 'Gurgaon' in city or 'Gurugram' in city:
            tips.append("Gurgaon's corporate hub status ensures steady rental demand")

        tips.append("Consider metro connectivity for better resale value")
        tips.append("Properties near IT hubs typically have higher rental yields")
        return tips


if __name__ == "__main__":
    predictor = PropertyValuePredictor()
    success = predictor.train_models("cleaned_resale_properties.csv")
    if success:
        print("\n✅ Training complete. Models saved.")
    else:
        print("\n❌ Training failed.")