"""
Machine learning repository
"""
import logging

LOGGER = logging.getLogger('repo')


class MLObjectType(Enum):
    """Enums describing all ml object types.
    """
    TRAINING_DATA = 'training_data'
    TEST_DATA = 'test_data'
    MODEL_PARAMETER = 'model_parameter'
    TRAINING_PARAMETER = 'training_parameter'




