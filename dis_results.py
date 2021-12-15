import matplotlib.pyplot as plt


plt.plot([1, 2, 3], [52.22, 57.96, 58.06], color="black", label="ViT - No KD")
plt.legend()
plt.title("Hard Label Distillation")
plt.xlabel("Num. of Encoders")
plt.xticks([1, 2, 3])
plt.ylabel("Accuracy")
plt.ylim(0, 70)
plt.show()
