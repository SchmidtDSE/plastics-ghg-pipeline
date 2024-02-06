"""Constants related to the GHG pipeline shared across tasks.

License: BSD
"""
# Column which serves as an internal check that all polymers are accounted for
OTHER_SUBTYPE = 'MISC'

# Allowed size of the unknown ratio
UNKNOWN_RATIO_TOLLERANCE = 0.0001

# Assert that full ratios are available within at least these years
ACTUALS_REQUIRED_MIN_YEAR = 2005
ACTUALS_REQUIRED_MAX_YEAR = 2019

# Columns expected in the raw data input for trade
EXPECTED_RAW_DATA_COLS = [
    'year',
    'region',
    'series',
    'subtype',
    'ratioSubtype',
    'gdp',
    'population'
]

# Minimum year for output series
MIN_YEAR = 2005

# Expected regions
REGIONS = [
    'China',
    'EU30',
    'NAFTA',
    'RoW'
]

# Expected subtypes for goods
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

# Expected subtypes for resin
POLYMERS = [
    '60% LDPE, 40% HDPE',
    'PP',
    'PS',
    'PVC',
    '100% OTP',
    '50% OTP, 50% OTS',
    'PET',
    'PUR'
]

# Combined expected subtypes across series
SUBTYPES = SECTORS + POLYMERS

# Expected columns for change records in dictionary form
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

# Names of columns for region one hot encoding
INPUTS_REGIONS = ['region_%s' % x for x in REGIONS]

# Names of columns for sector one hot encoding
INPUTS_SECTORS = ['subtype_%s' % x for x in SECTORS]

# Names of columns for polymer one hot encoding
INPUTS_POLYMERS = ['polymer_%s' % x for x in POLYMERS]

# Combined expected ordered list of inputs
INPUTS = [
    'years',
    'gdpChange',
    'populationChange',
    'beforeRatio',
] + INPUTS_REGIONS + INPUTS_SECTORS + INPUTS_POLYMERS

# Response variable across tasks
RESPONSE = 'afterRatio'

# Common file names
TASK_DIR = 'task'
DEPLOY_DIR = 'deploy'
CONFIG_NAME = 'job.json'
TRADE_FRAME_NAME = 'trade_inputs.csv'

# Modeling supported algorithm strategy names
SUPPORTED_ALGORITHMS = [
    'linear',
    'svr',
    'tree',
    'random forest',
    'adaboost'
]

# Use ceiling that covers both resin (0.3) and goods trade (0.1)
ALLOWED_TEST_ERROR = 0.3
ALLOWED_OUT_SAMPLE_ERROR = 0.3

# Expected output columns for sweep records
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

# Number of years to include when predicting / imputing a ratio
NUM_YEARS_INFERENCE_WINDOW = 5

# Expected output columns for projection records
EXPECTED_PROJECTION_COLS = [
    'year',
    'region',
    'subtype',
    'ratioSubtype',
    'gdp',
    'population'
]

# Strings (lowercased) that should be interpreted as ratio is not known
RATIO_NONE_STRS = ['', 'none']

# URL at which the raw trade inputs can be found
TRADE_INPUTS_URL = 'https://global-plastics-tool.org/data/trade_inputs.csv'
