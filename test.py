from pathlib import Path
from SzcoreEvaluation import MetricsStore
import logging
import numpy as np

n = 219
N = 10000
mask = np.array([[0]*n + [1]*n+(N-2*n-1)*[0]])
mask2 = np.random.randint(0,2,(1,N))
fs = 250
config = logging.getLogger("Config")
config.setLevel(logging.INFO)
# config.info()
config.model.save_dir = "TestNameTest"

metricsStoreTest = MetricsStore(config)
outDir = Path("./irregulars_neureka_codebase/predictions/")
metricsStoreTest.store_scores(outDir)
metricsStoreTest.store_metrics(outDir, outDir)
metricsStoreTest.evaluate_multiple_predictions(mask, mask2 == 1, ["Patient 1"])