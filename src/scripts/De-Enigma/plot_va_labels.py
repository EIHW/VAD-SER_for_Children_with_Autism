import matplotlib.pyplot as plt
import pandas as pd
from src.utils.csv_utils import get_delimeter

processed_label = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/label_processed_0.05/B007_T01.csv"
original_label = "/home/manu/eihw/data_work/manuel/data/EMBOA/De-Enigma_Audio_Database/tier_5_arousal_gs/B007_T01_aligned.csv"

processed_df = pd.read_csv(processed_label, delimiter=get_delimeter(processed_label))
original_df = pd.read_csv(original_label, delimiter=get_delimeter(original_label))

time_processed = processed_df.iloc[:, 0].values
arousal_processed = processed_df.iloc[:, 1].values

time_original = original_df.iloc[:, 0].values
arousal_original = original_df.iloc[:, 1].values

plt.plot(time_original, arousal_original)
plt.savefig("images/arousal_original.png")
plt.clf()

plt.plot(time_processed, arousal_processed)
plt.savefig("images/arousal_processed.png")
plt.clf()


print()
