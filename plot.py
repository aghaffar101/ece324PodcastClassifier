import matplotlib.pyplot as plt

plt.title("Test Accuracy vs num classes in model")
plt.plot([2, 5, 10, 20, 30, 45], [0.998, 0.984, 0.965, 0.962, 0.957, 0.938], label="Train")
plt.xlabel("Number of Classes")
plt.ylabel("Test Accuracy")
plt.legend(loc='best')
plt.savefig("Train_Valid_Acc_LogRegression.pdf")
plt.show()
