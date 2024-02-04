"""Constants related to the GHG pipeline shared across tasks.

License: BSD
"""

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

CHANGE_COLS = [
    'region',
    'sector',
    'year',
    'years',
    'gdpChange',
    'populationChange',
    'beforeRatio',
    'afterRatio'
]

INPUTS_REGIONS = ['region_%s' % x for x in REGIONS]

INPUTS_SECTORS = ['sector_%s' % x for x in SECTORS]

INPUTS = [
    'years',
    'gdpChange',
    'populationChange',
    'beforeRatio',
] + INPUTS_REGIONS + INPUTS_SECTORS

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
