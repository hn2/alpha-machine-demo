DELIMITER = "-"
MODELS_DIR = "E:/alpha-machine-files/models/forex/oanda/hour/sb3-train"
MODEL_NAME = "fx_wk.51_lr.0.00001_b.512_sde.False_cb.True_res.True_lev.20-7-300100-10000_0-120-2-oanda-hour-on_algo.ppo-c_pos.long_and_short-c_ind.all_full-c_rew.log_returns-m_rl.True-9bbe338c"
START_DATE = [2021, 9, 17]
END_DATE = [2021, 10, 29]
LOT_SIZE = "Micro"
LEVERAGE = 20
MAX_SLIPPAGE_PERCENT = 0.01
CASH = 1000.0

exec(open("../../alpha-machine/src/drl-pf-fx-sb3/test/test-main.py").read())
