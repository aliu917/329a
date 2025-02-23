import pandas as pd
import json

def print_df(df):
    for i, row in df.iterrows():
        print("Question: ", row["question"])
        print("Answer: ", row["answer_1"])
        print(f"Pred diff: {row['pred_1']}, {row['pred_2']}")
        print(f"Feedback: {row['feedback_1']}")
        print()

def run_acc(run_name, result_name="results"):
    file = f"runs/{run_name}/{result_name}.json"
    with open(file, 'r', encoding='utf-8') as f:
        x = json.load(f)
    df = pd.DataFrame(x)
    acc = df["correct"].sum() / len(df["correct"])
    return df, acc

def cmp_runs(run_name1, run_name2, result_name1="results", result_name2="results"):
    df1, acc1 = run_acc(run_name1, result_name1)
    df2, acc2 = run_acc(run_name2, result_name2)

    print(f"Run {run_name1} acc: {acc1}")
    print(f"Run {run_name2} acc: {acc2}")
    print()

    select_cols = ["question", "feedback_1", "pred_1", "pred_2", "answer_1"]

    df = pd.merge(df1, df2, on='question', suffixes=('_1', '_2'))
    improved_df = df[(~df['correct_1']) & (df['correct_1'] != df['correct_2'])][select_cols]
    print('-' * 50)
    print(f"Number of examples improved:", len(improved_df))
    print('-' * 50)
    print_df(improved_df)
    worse_df = df[(df['correct_1']) & (df['correct_1'] != df['correct_2'])][select_cols]

    print('-'*50)
    print(f"Number of examples worsened:", len(worse_df))
    print('-' * 50)

    wrong_df = df[(~df['correct_1']) & (df['correct_1'] == df['correct_2'])][select_cols]

    print('-' * 50)
    print(f"Number of examples both wrong:", len(wrong_df))
    print('-' * 50)

    print_df(wrong_df)
    return improved_df, wrong_df

if __name__ == '__main__':
    # print(run_acc("baseline"))
    diff = cmp_runs("baseline", "baseline", result_name1="base_results", result_name2="results")
