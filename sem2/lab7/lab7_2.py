import numpy as np
import matplotlib.pyplot as plt

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
false_positive_probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
false_negative_probabilities = [0.2, 0.3, 0.4, 0.5, 0.6]

# Вычисление вероятности общей ошибки
total_error_probabilities = [(false_positive + false_negative) / 2 for false_positive, false_negative in zip(false_positive_probabilities, false_negative_probabilities)]

# Построение графика
plt.plot(thresholds, false_positive_probabilities, label='False Positive')
plt.plot(thresholds, false_negative_probabilities, label='False Negative')
plt.plot(thresholds, total_error_probabilities, label='Total Error')

plt.xlabel('Threshold')
plt.ylabel('Probability')
plt.title('Error Probabilities vs Threshold')
plt.legend()

plt.show()
