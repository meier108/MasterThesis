import pandas as pd

from experiment_loop import run_tfbind8_experiment


def main():
    result = run_tfbind8_experiment(
        seed=42,
        transcription_factor="SIX6_REF_R1",
        target_split="test_c",
        num_rounds=30,
        mutants_per_round=32,
    )

    print("Training samples:", len(result["train_df"]))
    print("Split counts:")
    print(result["df"]["split"].value_counts())

    print("\nRQ1 metrics by split:")
    print(pd.DataFrame(result["metrics"].values()))

    print("\nTrajectory summary:")
    for key, value in result["trajectory"]["summary"].items():
        print(f"{key}: {value}")

    if not result["trajectory"]["trajectory"].empty:
        print("\nTrajectory improvements:")
        print(result["trajectory"]["trajectory"].head())

    return result

if __name__ == "__main__":
    main()
