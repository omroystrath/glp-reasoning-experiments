import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

def compute_ci(x, ci=0.95, n_resamples=10000):
    arr = np.array(x)
    res = bootstrap(
        (arr,),
        np.mean,
        confidence_level=ci,
        n_resamples=n_resamples,
        method='percentile',
    )
    return pd.Series({
        'mean': arr.mean(),
        'ci_low': res.confidence_interval.low,
        'ci_high': res.confidence_interval.high,
    })

def compile_results(out_file="runs/persona_vectors_results.csv", save_folder="persona_vectors", eval_dir="eval_persona", model="Llama-3.1-8B-Instruct"):
    results_list = []
    for file in glob.glob(f"{save_folder}/{eval_dir}/{model}/*"):
        method = os.path.basename(file).split("_")[0]
        trait = os.path.basename(file).split("_")[1]
        coef = float(os.path.basename(file).split("_")[-1].replace(".csv", "").replace("coef", ""))
        raw_results = pd.read_csv(file)
        results = pd.DataFrame()
        results["fluency_score"] = raw_results["coherence"]
        results["concept_score"] = raw_results[trait]
        results["trait"] = trait
        results["alpha"] = coef
        results["method"] = method
        results_list.append(results)
    df = pd.concat(results_list)
    summary = (
        df.groupby(['method', 'trait', 'alpha'], sort=True)
        .apply(lambda g: pd.DataFrame({
            'fluency': compute_ci(g['fluency_score']),
            'concept': compute_ci(g['concept_score']),
        }).T)
        .round(4)
    )
    summary.index.names = [*summary.index.names[:-1], 'metric']
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    summary.to_csv(out_file, index=True)

if __name__ == "__main__":
    compile_results()