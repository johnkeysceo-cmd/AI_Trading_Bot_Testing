from ray import tune

def tune_example():
    def train_fn(config):
        # Dummy function: replace with model training evaluation
        score = (1 - abs(config["threshold"] - 0.5)) * 1000
        tune.report(score=score)

    analysis = tune.run(train_fn, config={"threshold": tune.uniform(0.1, 1.0)})
    return analysis.get_best_config(metric="score", mode="max")
