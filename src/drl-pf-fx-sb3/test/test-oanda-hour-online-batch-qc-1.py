import platform

CONFIG_FILE = '../settings/test/config-test-oanda-hour-online-batch-1.ini'

if platform.system() == 'Windows':
    TEST_SCRIPT_DIR = 'C:/alpha-machine/src/drl-pf-fx-sb3/test'
    TEST_SCRIPT = 'C:/alpha-machine/src/drl-pf-fx-sb3/test/test-oanda-hour-online-sb3-1.py'
    TEST_DIR = 'C:/alpha-machine-qc/test-online-windows-h1-oanda-sb3-1'
    TEST_EXE = 'C:/alpha-machine-qc/test-online-windows-h1-oanda-sb3-1/QuantConnect.Lean.Launcher.exe'
    MODELS_DIR = 'E:/alpha-machine-files/models/forex/oanda/hour/sb3-train'
    STATS_DIR = 'E:/alpha-machine-files/stats'
elif platform.system() == 'Linux':
    TEST_SCRIPT_DIR = '/home/ubuntu/alpha-machine/src/drl-pf-fx-sb3/test'
    TEST_SCRIPT = '/home/ubuntu/alpha-machine/src/drl-pf-fx-sb3/test/test-oanda-hour-online-sb3-1.py'
    TEST_DIR = '/home/ubuntu/alpha-machine-qc/test-online-ubuntu-h1-oanda-sb3-1'
    TEST_EXE = '/home/ubuntu/alpha-machine-qc/test-online-ubuntu-h1-oanda-sb3-1/QuantConnect.Lean.Launcher.dll'
    MODELS_DIR = '/home/ubuntu/alpha-machine-files/models/forex/oanda/hour/sb3-train'
    STATS_DIR = '/home/ubuntu/alpha-machine-files/stats'

exec(open("test-batch-qc-main.py").read())