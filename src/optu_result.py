import sqlite3
import pandas as pd
import optuna
import numpy as np

# 最適化結果の確認
dbname = "optuna_study.db"
with sqlite3.connect(dbname) as conn:
    df = pd.read_sql("SELECT * FROM trial_params;", conn)
# 定義
study_name = 'example-study'
study = optuna.load_study(study_name=study_name, storage="sqlite:///./optuna_study.db")
#study = optuna.study.create_study(study_name=study_name,
#                                  storage='sqlite:///./optuna_study.db',
#                                  load_if_exists=True)
    

values = [each.value for each in study.trials]
values = list(filter(None, values))
trial_id = values.index(max(values)) + 1
print(trial_id)

best_values = [np.max(values[:k + 1]) for k in range(len(values))]
print(best_values)

print(df)
print(df[df['trial_id'] == trial_id])  # 最適だった時のパラメータの表示

