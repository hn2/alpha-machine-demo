DELIMITER = '-'
ENV_VERBOSE = True

MODELS_DIR = 'E:/alpha-machine-files/models/forex/oanda/hour/sb3-train'
STATS_FILE = 'E:/alpha-machine-files/stats/stats_1_lev_20-sb3-train-Oanda-OandaBrokerage-hour-20-12-2022-0-1000-.csv'
NUM_OF_MODELS = 10
FILES_LOOKBACK_HOURS = 2400
INCLUDE_PATTERNS = ['']
START_DATE = [2020, 3, 25]
END_DATE = [2022, 12, 20]
LOT_SIZE = "Micro"
LEVERAGE = 20
MAX_SLIPPAGE_PERCENT = 0.01
CASH = 1000.0

exec(open("../../alpha-machine/src/drl-pf-fx-sb3/test/test-ensamble-main.py").read())