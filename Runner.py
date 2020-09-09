from data_service import DataService
from model_service import ModelService

data, char_to_ix, ix_to_char = DataService.get_data()
parameters = ModelService.model(data, ix_to_char, char_to_ix, verbose=True)