"""Constants related to the GHG pipeline shared across tasks.

License: BSD
"""
MIN_YEAR = 2005

REGIONS = [
    'China',
    'EU30',
    'NAFTA',
    'RoW'
]

SECTORS = [
    'Agriculture',
    'Building_Construction',
    'Electrical_Electronic',
    'Household_Leisure_Sports',
    'Others',
    'Packaging',
    'Textiles',
    'Transportation'
]

POLYMERS = [
    '60% LDPE, 40% HDPE',
    'PP',
    'PS',
    'PVC',
    '100% OTP',
    '50% OTP, 50% OTS',
    'PET'
    '100% OTP',
    'PUR'
]

SUBTYPES = SECTORS + POLYMERS

CHANGE_COLS = [
    'region',
    'subtype',
    'year',
    'years',
    'gdpChange',
    'populationChange',
    'beforeRatio',
    'afterRatio'
]

INPUTS_REGIONS = ['region_%s' % x for x in REGIONS]

INPUTS_SECTORS = ['subtype_%s' % x for x in SECTORS]

INPUTS_POLYMERS = ['polymer_%s' % x for x in POLYMERS]

INPUTS = [
    'years',
    'gdpChange',
    'populationChange',
    'beforeRatio',
] + INPUTS_REGIONS + INPUTS_SECTORS + INPUTS_POLYMERS

RESPONSE = 'afterRatio'

TASK_DIR = 'task'
DEPLOY_DIR = 'deploy'

CONFIG_NAME = 'job.json'
TRADE_FRAME_NAME = 'trade_inputs.csv'

SUPPORTED_ALGORITHMS = [
    'linear',
    'svr',
    'tree',
    'random forest',
    'adaboost'
]

ALLOWED_TEST_ERROR = 0.1
ALLOWED_OUT_SAMPLE_ERROR = 0.1

EXPECTED_SWEEP_COLS = [
    'algorithm',
    'depth',
    'estimators',
    'features',
    'alpha',
    'kernel',
    'degree',
    'trainMae',
    'validationMae',
    'testMae'
]

NUM_YEARS_INFERENCE_WINDOW = 5

EXPECTED_PROJECTION_COLS = [
    'year',
    'region',
    'subtype',
    'ratioSubtype',
    'gdp',
    'population'
]

RATIO_NONE_STRS = ['', 'none']
