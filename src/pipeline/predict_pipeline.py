import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Get the absolute path to the root directory of the project
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            model_path = os.path.join(root_path, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(root_path, "artifacts", "preprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Debug: print the features before scaling
            print("Features before scaling:")
            print(features)

            # Ensure the features are in the same order as during training
            expected_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", 
                                "test_preparation_course", "reading_score", "writing_score"]
            for col in expected_columns:
                if col not in features.columns:
                    raise CustomException(f"Expected column {col} is missing from the input data")

            # Transform features
            data_scaled = preprocessor.transform(features)

            # Debug: print the features after scaling
            print("Features after scaling:")
            print(data_scaled)

            preds = model.predict(data_scaled)

            # Debug: print the predictions
            print("Predictions:")
            print(preds)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
