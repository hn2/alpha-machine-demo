DELIMITER = "-"
MODELS_DIR = "E:/alpha-machine-files/models/forex/oanda/daily/sb3-train"
MODEL_NAME = "fx_wk.51_lr.0.00001_b.32_ns.none_cb.True_res.True_lev.20-28-100100-5000_0-100-2-oanda-daily-on_algo.td3-c_pos.long_and_short-c_ind.all_full-c_rew.log_returns-m_rl.True-f9cc6f85"
START_DATE = [2020, 3, 26]
END_DATE = [2022, 12, 21]
LOT_SIZE = "Micro"
LEVERAGE = 20
MAX_SLIPPAGE_PERCENT = 0.01
CASH = 1000.0

exec(open("../../alpha-machine/src/drl-pf-fx-sb3/test/test-main.py").read())
