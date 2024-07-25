import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import os


MODEL_PATH = "output/clasificator_model.pkl"
ENCODERS_FOLDER = "output/encoders/"

TEN_FEATURES = [
    'Demand', 
    'Charges', 
    'Contract', 
    'TechSupport', 
    'Security', 
    'PaymentMethod', 
    'OnlineBackup', 
    'DeviceProtection', 
    'Dependents', 
    'Partner'
]


class ClassificatorModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self.model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        self.encoders_folder = os.path.join(os.path.dirname(__file__), ENCODERS_FOLDER)
        self._label_encoders = {}
        self._class_le = None
        self._scaler = None
        try:
            self._model = load(self.model_path)
        except Exception as e:
            self._model = None
            print(f"Warning: Error loading the model - {str(e)}. Model will be set to None. Train it.")
        self._load_encoders_and_scaler()


    def _load_encoders_and_scaler(self):
        try:
            for encoder_file in os.listdir(self.encoders_folder):
                if encoder_file.endswith('_label_encoder.pkl'):
                    column = encoder_file.replace('_label_encoder.pkl', '')
                    self._label_encoders[column] = load(os.path.join(self.encoders_folder, encoder_file))
                    if encoder_file == 'class_label_encoder.pkl':
                        self._class_le = load(os.path.join(self.encoders_folder, encoder_file))
                elif encoder_file == 'features_scaler.pkl':
                    self._scaler = load(os.path.join(self.encoders_folder, encoder_file))
        except Exception as e:
            print(f"Warning: Error loading encoders or scaler - {str(e)}")
    

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): Raw data.
            target_column (str, optional): If set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target.
            or
            pd.DataFrame: Features.
        """
        
        # Codificar la variable objetivo
        if 'Class' in data.columns and self._class_le is not None:
            data.dropna() 

            # Clean Data to Preprocessing
            data.drop(['autoID'], axis=1, inplace=True)
            data['Demand'] = pd.to_numeric(data['Demand'], errors='coerce')
            data.dropna(subset=['Demand'], inplace=True)
            data['Demand'] = data['Demand'].astype('float64')
            data['Class'] = self._class_le.transform(data['Class'])

        # Codificar variables categóricas
        for column in data.select_dtypes(include=['object']).columns:
            if column != 'Class' and column in self._label_encoders:
                le = self._label_encoders[column]
                data[column] = le.transform(data[column])
        
        # Características y variable objetivo
        # Initialize a new DataFrame with the desired columns and fill it with 0
        processed_features = pd.DataFrame(0, index=range(len(data)), columns=TEN_FEATURES)
        # Copy data from the original DataFrame to the corresponding columns in the processed DataFrame
        processed_features.update(data)
        target = None

        # Crear el retorno dependeindo tipo de preprocemaiento
        result = processed_features
        if target_column:
            target = data[[target_column]]
        # Normalizar características si el scaler está disponible
        if self._scaler:
            processed_features = self._scaler.transform(processed_features)
        
        # Pasar el resultado final
        if target:
            result = (processed_features, target)
        else:
            result = processed_features
                
        return result
    

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Dividir los datos en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
        y_train = y_train.squeeze()

        # Balancear los datos con SMOTE
        smote = SMOTE(random_state=42)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

        # Normalizar las características
        self._scaler = StandardScaler()
        x_train_scaled = self._scaler.fit_transform(x_train_resampled)
        x_test_scaled = self._scaler.transform(x_test)

        # Crear y entrenar el modelo RandomForest
        estimators = 20
        self._model = RandomForestClassifier(n_estimators=estimators, random_state=42)
        self._model.fit(x_train_scaled, y_train_resampled)

        # Guardar el modelo en un archivo
        dump(self._model, self.model_path)
        dump(self._scaler, os.path.join(self.encoders_folder, 'features_scaler.pkl'))
        
        return


    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict Class Alpha-Betha for new data.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise Exception("Model not found. Cannot make predictions. Train model first.")
        try:
            
            predictions = self._model.predict(features)
            if self._class_le:
                decoded_predictions = self._class_le.inverse_transform(predictions)
            return predictions.tolist()
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")