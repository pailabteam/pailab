"""
Machine learning repository
"""
from enum import Enum
import logging

LOGGER = logging.getLogger('repo')


class MLObjectType(Enum):
    """Enums describing all ml object types.
    """
    TRAINING_DATA = 'training_data'
    TEST_DATA = 'test_data'
    MODEL_PARAM = 'model_param'
    TRAINING_PARAM = 'training_parame'
    MODEL_CALIBRATOR = 'model_calibrator'
    MODEL_EVALUATOR = 'model_evaluator'
    PREP_PARAM = 'prep_param'
    PREPROCESSOR = 'preprocessor'



    


