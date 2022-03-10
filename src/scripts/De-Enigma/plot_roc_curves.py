import numpy as np
from src.utils.plot_utils import make_nice_line_plot

fpr_all_path = "/home/manu/PycharmProjects/code_repo/Manu_Code/results/EMBOA_VAD/evaluation_049/fpr.npy"
tpr_all_path = "/home/manu/PycharmProjects/code_repo/Manu_Code/results/EMBOA_VAD/evaluation_049/tpr.npy"
fpr_child_path = "/home/manu/PycharmProjects/code_repo/Manu_Code/results/EMBOA_VAD/evaluation_048/fpr.npy"
tpr_child_path = "/home/manu/PycharmProjects/code_repo/Manu_Code/results/EMBOA_VAD/evaluation_048/tpr.npy"

out_path = "roc_curves/ROC_Curve.jpg"

fpr_all = np.load(fpr_all_path)
fpr_child = np.load(fpr_child_path)

tpr_all = np.load(tpr_all_path)
tpr_child = np.load(tpr_child_path)

line_x = np.arange(0,1,0.01)
line_y = np.arange(0,1,0.01)

Xs = [fpr_all, fpr_child, line_x]
Ys = [tpr_all, tpr_child, line_y]
labels = ["General VAD", "Child VAD", ""]

title = "ROC-Curve"
x_axis = "FPR"
y_axis = "TPR"

#line_style = ["r", "b", "--"]
colors = ["r", "b", (0.1,0.1,0.1)]
line_styles = "-", "-", "--"

make_nice_line_plot(out_path, Xs, Ys, labels, title=title, font_size=16, x_axis=x_axis, y_axis=y_axis, colors=colors, line_styles=line_styles)