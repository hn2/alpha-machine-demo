DELIMITER = "-"
MODELS_DIR = "E:/alpha-machine-files/models/forex/oanda/daily/sb3-train"
MODEL_NAME = "fx_wk.51_lr.0.00001_b.512_sde.False_cb.True_res.True_lev.20-7-300100-5000_0-100-2-oanda-daily-on_algo.ppo-c_pos.long_and_short-c_ind.all_full-c_rew.log_returns-m_rl.True-d5d2186a"
START_DATE = [2006, 7, 17]
END_DATE = [2009, 4, 12]
LOT_SIZE = "Micro"
LEVERAGE = 20
MAX_SLIPPAGE_PERCENT = 0.01
CASH = 1000.0

exec(open("../../alpha-machine/src/drl-pf-fx-sb3/test/test-main.py").read())
