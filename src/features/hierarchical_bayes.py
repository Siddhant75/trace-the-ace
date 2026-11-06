"""
Hierarchical Empirical Bayes Difficulty Shrinkage Feature Extractor
Models student difficulty across 3 hierarchical levels:
1. Global Prior: mu_0 = E[y]
2. Grade Level Prior: mu_g = (n_g * y_bar_g + m1 * mu_0) / (n_g + m1)
3. Topic Level Prior: mu_gt = (n_gt * y_bar_gt + m2 * mu_g) / (n_gt + m2)
4. Learning Objective Posterior: mu_gto = (n_o * y_bar_o + m3 * mu_gt) / (n_o + m3)
"""
import numpy as np
import pandas as pd

def norm_str(s):
    return str(s).strip().lower()

class HierarchicalBayesEncoder:
    def __init__(self, m1=5.0, m2=10.0, m3=15.0):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.mu_0 = 0.7035
        self.grade_map = {}
        self.topic_map = {}
        self.obj_map = {}

    def fit(self, df):
        df_c = df.copy()
        df_c["norm_grade"] = df_c["grade_level"].apply(norm_str) if "grade_level" in df_c else "unknown"
        df_c["norm_topic"] = df_c["topic_id"].apply(norm_str) if "topic_id" in df_c else "unknown"
        df_c["norm_obj"] = df_c["learning_objective"].apply(norm_str)

        self.mu_0 = float(df_c["is_correct"].mean())

        # 1. Grade level
        for grade, grp in df_c.groupby("norm_grade"):
            n = len(grp)
            y_sum = grp["is_correct"].sum()
            self.grade_map[grade] = (y_sum + self.m1 * self.mu_0) / (n + self.m1)

        # 2. Topic level conditioned on grade
        for (grade, topic), grp in df_c.groupby(["norm_grade", "norm_topic"]):
            n = len(grp)
            y_sum = grp["is_correct"].sum()
            parent_prior = self.grade_map.get(grade, self.mu_0)
            self.topic_map[(grade, topic)] = (y_sum + self.m2 * parent_prior) / (n + self.m2)

        # 3. Objective level conditioned on (grade, topic)
        for (grade, topic, obj), grp in df_c.groupby(["norm_grade", "norm_topic", "norm_obj"]):
            n = len(grp)
            y_sum = grp["is_correct"].sum()
            parent_prior = self.topic_map.get((grade, topic), self.grade_map.get(grade, self.mu_0))
            self.obj_map[(grade, topic, obj)] = (y_sum + self.m3 * parent_prior) / (n + self.m3)

        return self

    def transform(self, df):
        df_c = df.copy()
        df_c["norm_grade"] = df_c["grade_level"].apply(norm_str) if "grade_level" in df_c else "unknown"
        df_c["norm_topic"] = df_c["topic_id"].apply(norm_str) if "topic_id" in df_c else "unknown"
        df_c["norm_obj"] = df_c["learning_objective"].apply(norm_str)

        enc_grade = []
        enc_topic = []
        enc_obj = []

        for _, row in df_c.iterrows():
            g = row["norm_grade"]
            t = row["norm_topic"]
            o = row["norm_obj"]

            g_val = self.grade_map.get(g, self.mu_0)
            t_val = self.topic_map.get((g, t), g_val)
            o_val = self.obj_map.get((g, t, o), t_val)

            enc_grade.append(g_val)
            enc_topic.append(t_val)
            enc_obj.append(o_val)

        res = pd.DataFrame(index=df.index)
        res["bayes_enc_grade"] = enc_grade
        res["bayes_enc_topic"] = enc_topic
        res["bayes_enc_obj"] = enc_obj
        res["bayes_difficulty_spread"] = np.array(enc_obj) - np.array(enc_grade)
        return res
