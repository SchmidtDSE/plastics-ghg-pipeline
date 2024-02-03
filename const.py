"""Constants related to the GHG pipeline shared across tasks.

License: BSD
"""

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

INPUTS = [
    'years',
    'gdpChange',
    'populationChange',
    'beforeRatio',
    'region_China',
    'region_EU30',
    'region_NAFTA',
    'region_RoW',
    'sector_Agriculture',
    'sector_Building_Construction',
    'sector_Electrical_Electronic',
    'sector_Household_Leisure_Sports',
    'sector_Others',
    'sector_Packaging',
    'sector_Textiles',
    'sector_Transportation'
]

RESPONSE = 'afterRatio'

TASK_DIR = 'task'
DEPLOY_DIR = 'deploy'

CONFIG_NAME = 'job.json'
TRADE_FRAME_NAME = 'trade_inputs.csv'
