import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Collecting results


class Data:
    def __init__(self) -> None:

        self.u_auroc_20 = {"VI-NAF": np.array([78.19, 77.91, 77.68, 77.56, 77.81]),
                           "VI-Sylv": np.array([77.22, 77.52, 77.37, 77.5, 77.43]),
                           "VI-Normal": np.array([74.0, 74.6, 73.89, 74.44, 74.76]),
                           "Dropout": np.array([76.37, 76.41, 76.33])}
        self.u_aupr_20 = {"VI-NAF": np.array([33.08, 32.53, 32.41, 32.25, 32.04]),
                          "VI-Sylv": np.array([32.64, 32.08, 31.81, 31.96, 31.66]),
                          "VI-Normal": np.array([27.34, 28.19, 26.99, 27.44, 27.62]),
                          "Dropout": np.array([30.23, 30.37, 30.24])}
        self.u_fpr95_20 = {"VI-NAF": np.array([47.67, 48.05, 48.35, 48.56, 48.87]),
                           "VI-Sylv": np.array([48.72, 48.38, 48.92, 49.06, 48.3]),
                           "VI-Normal": np.array([55.30, 54.44, 55.62, 55., 54.34]),
                           "Dropout": np.array([51.24, 50.99, 51.27])}
        self.h_auroc_20 = {"VI-NAF": np.array([78.79, 78.43, 78.14, 78.13]),
                           "VI-Sylv": np.array([77.56, 77.84, 77.97, 77.98, 77.01]),
                           "VI-Normal": np.array([77.16, 77.37, 76.98, 77.06, 77.54]),
                           "Dropout": np.array([76.19, 76.22, 76.14])}
        self.h_aupr_20 = {"VI-NAF": np.array([35.1, 33.91, 33.9, 32.89, 33.08]),
                          "VI-Sylv": np.array([32.36, 32.33, 32.74, 32.55, 31.45]),
                          "VI-Normal": np.array([30.7, 30.39, 29.96, 30.29, 30.26]),
                          "Dropout": np.array([30.02, 30.17, 30.02])}
        self.h_fpr95_20 = {"VI-NAF": np.array([44.65, 45.32, 45.69, 45.75, 45.68]),
                           "VI-Sylv": np.array([46.92, 47.12, 46.26, 46.37, 48.4]),
                           "VI-Normal": np.array([47.4, 46.78, 47.21, 47.58, 46.91]),
                           "Dropout": np.array([51.37, 51.12, 51.39])}
        #########################################################
        self.u_auroc_10 = {"VI-NAF": np.array([77.73, 77.44, 77.38, 77.48, 77.77]),
                           "VI-Sylv": np.array([76.94, 77.73, 77.37, 77.54, 77.45]),
                           "VI-Normal": np.array([74.27, 73.99, 73.72, 73.91, 74.36]),
                           "Dropout": np.array([76.39, 76.37, 76.32])}
        self.u_aupr_10 = {"VI-NAF": np.array([32.17, 32.52, 31.96, 32.17, 32.56]),
                          "VI-Sylv": np.array([31.97, 32.24, 32.09, 31.88, 31.79]),
                          "VI-Normal": np.array([27.46, 27.81, 27.32, 26.88, 27.73]),
                          "Dropout": np.array([30.27, 30.52, 30.3])}
        self.u_fpr95_10 = {"VI-NAF": np.array([48.93, 49.13, 48.57, 48.79, 48.39]),
                           "VI-Sylv": np.array([48.78, 47.01, 49.17, 48.33, 49.06]),
                           "VI-Normal": np.array([54.79, 55.69, 56.11, 55.56, 55.1]),
                           "Dropout": np.array([51.13, 51.25, 51.33])}
        self.h_auroc_10 = {"VI-NAF": np.array([77.5, 77.5, 78.38, 77.94, 78.04]),
                           "VI-Sylv": np.array([76.79, 77.97, 77.31, 77.91, 77.41]),
                           "VI-Normal": np.array([76.99, 76.59, 77.04, 76.77, 76.54]),
                           "Dropout": np.array([76.22, 76.19, 76.13])}
        self.h_aupr_10 = {"VI-NAF": np.array([32.84, 33.46, 33.41, 32.83, 33.49]),
                          "VI-Sylv": np.array([31.84, 32., 32.64, 31.92, 32.05]),
                          "VI-Normal": np.array([30.38, 29.79, 29.95, 29.84, 30.07]),
                          "Dropout": np.array([30.11, 30.28, 30.08])}
        self.h_fpr95_10 = {"VI-NAF": np.array([47.67, 45.86, 45.85, 47.05, 47.15]),
                           "VI-Sylv": np.array([47.88, 46.78, 48.55, 46.99, 50.33]),
                           "VI-Normal": np.array([48.27, 48.25, 47.25, 47.86, 48.36]),
                           "Dropout": np.array([51.25, 51.41, 51.48])}
        #########################################################
        self.u_auroc_5 = {"VI-NAF": np.array([77.10, 77.7, 77.15, 77.13, 76.7, 77.29]),
                          "VI-Sylv": np.array([76.46, 76.65, 76.35, 76.7, 77.2]),
                          "VI-Normal": np.array([73.42, 73.53, 73.72, 74.36, 73.57]),
                          "Dropout": np.array([76.22, 76.38, 76.49])}
        self.u_aupr_5 = {"VI-NAF": np.array([31.15, 33.09, 31.63, 31.89, 31.7, 30.94]),
                         "VI-Sylv": np.array([31.08, 31.4, 31.04, 31.54, 31.05]),
                         "VI-Normal": np.array([28.37, 26.52, 27.26, 27.95, 27.42]),
                         "Dropout": np.array([30.21, 30.26, 30.47])}
        self.u_fpr95_5 = {"VI-NAF": np.array([49.05, 48.58, 49.52, 48.8, 50.08, 48.48]),
                          "VI-Sylv": np.array([49.68, 49.63, 49.65, 49.69, 49.87]),
                          "VI-Normal": np.array([56.4, 57.19, 55.73, 55.51, 56.]),
                          "Dropout": np.array([51.25, 51.28, 51.08])}
        self.h_auroc_5 = {"VI-NAF": np.array([77.37, 76.82, 76.36, 77.61, 78.86, 78.22]),
                          "VI-Sylv": np.array([76.99, 77.29, 76.38, 76.66, 76.56]),
                          "VI-Normal": np.array([76.88, 76.43, 76.02, 76.95, 75.71]),
                          "Dropout": np.array([76.05, 76.19, 76.3])}
        self.h_aupr_5 = {"VI-NAF": np.array([32.24, 32.07, 32.33, 33.5, 34.6, 31.97]),
                         "VI-Sylv": np.array([31.03, 31.16, 31.8, 30.65, 32.27]),
                         "VI-Normal": np.array([30.08, 29.09, 28.31, 29.27, 28.27]),
                         "Dropout": np.array([29.98, 30.02, 30.24])}
        self.h_fpr95_5 = {"VI-NAF": np.array([47.31, 49.68, 48.3, 47.92, 45.48, 46.01]),
                          "VI-Sylv": np.array([47.23, 48.26, 50.35, 50.35, 49.8]),
                          "VI-Normal": np.array([47.51, 48.58, 49.53, 49.74, 49.89]),
                          "Dropout": np.array([51.33, 51.48, 51.23])}


data = Data()
metric = "AUROC"
metric = "AUPR"
metric = "FPR95"
score = "u"
# score = "h"


# for metric in ("AUROC", "AUPR", "FPR95"):
#     for score in ("u", "h"):
#         d = {"# Samples": [], "Method": [], f"{metric} %": []}
#         for nsamples in (5, 10, 20):
#             for k, v in getattr(data, f"{score}_{metric.lower()}_{nsamples}").items():
#                 for v_ in v:
#                     d["# Samples"].append(nsamples)
#                     d["Method"].append(k)
#                     d[f"{metric} %"].append(v_)

#         df = pd.DataFrame(data=d)
#         # import ipdb; ipdb.set_trace()

#         # df_aupr_5 = pd.DataFrame({"# samples": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], "method": ["VI-NAF"]})
#         # df_aupr_10 = pd.DataFrame({"# samples": [1]})
#         # df_aupr_20 = pd.DataFrame({"# samples": [1]})
#         # df_fpr = pd.DataFrame({})

#         plt.figure()
#         sns.catplot(
#             data=df,
#             x="# Samples",
#             y=f"{metric} %",
#             hue="Method",
#             kind='point',
#             dodge=True,
#             errorbar='sd',
#             join=False,
#             capsize=0.05,
#             errwidth=0.5,
#             scale=0.2
#         )

#         plt.savefig(f"error_{score}_{metric.lower()}.png", dpi=300)


d = {"num_samples": [], "method": [],"score":[], "metric": [], "mean": [], "std": []}
for metric in ("AUROC", "AUPR", "FPR95"):
    for score in ("u", "h"):
        for nsamples in (5, 10, 20):
            for method in getattr(data, f"{score}_{metric.lower()}_{nsamples}").keys():
                d["num_samples"].append(nsamples)
                d["method"].append(method)
                d["metric"].append(metric)
                d["mean"].append(getattr(data, f"{score}_{metric.lower()}_{nsamples}")[method].mean())
                d["std"].append(getattr(data, f"{score}_{metric.lower()}_{nsamples}")[method].std())
                d["score"].append(score)
df = pd.DataFrame(data=d)
for score in ("u", "h"):
    for metric in ("AUROC", "AUPR", "FPR95"):
        plt.cla(); plt.clf()
        fig = plt.figure()
        for method in ('VI-NAF', 'VI-Sylv', 'VI-Normal', 'Dropout'):
            df_ = df[(df.method == method) & (df.score == score) & (df.metric == metric)]
            x = df_.loc[:, 'num_samples'].astype(str).ravel()
            y = df_.loc[:, 'mean'].ravel()
            y_hi = (df_.loc[:, 'mean'] + df_.loc[:, 'std']).values.ravel()
            y_lo = (df_.loc[:, 'mean'] - df_.loc[:, 'std']).values.ravel()
            plt.plot(x, y, label=method, linewidth=1.)
            plt.fill_between(x= x, y1=y_lo, y2=y_hi, alpha=0.2)
        plt.xlabel("num. samples")
        plt.ylabel(metric+" %")
        # plt.title(f"{metric} mean and standard deviation using score {score}")
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),          fancybox=True, shadow=True, ncol=4)
        # plt.legend(loc='best', bbox_to_anchor=(1.04, 0.5), ncol=1, borderaxespad=0)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        #   fancybox=False, shadow=False, ncol=4)
        plt.grid(visible=True, axis='y')
        plt.savefig(f"error_{score}_{metric.lower()}", dpi=300,bbox_inches='tight')

            
