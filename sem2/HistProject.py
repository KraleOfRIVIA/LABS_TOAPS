import numpy as np
import matplotlib.pyplot as plt

# Данные для гистограммы
hist_data = [11.75, 14.88, 18.01, 21.14, 24.27, 27.4, 30.53, 33.68]  # Интервалы
bins = [0.0095, 0.0575, 0.0575, 0.0958, 0.0575, 0.0319, 0.0095]  # Высоты интервалов

# Данные для кривой
y_curve = np.array([0.012, 0.038, 0.054, 0.075, 0.089, 0.063, 0.054, 0.034, 0.007])
x_curve = np.array([13.315, 16.445, 16.89, 19.575, 22.705, 25.73, 25.895, 28.365, 32.105])

# Создание графика
fig, ax = plt.subplots()

# Построение гистограммы
ax.bar(np.arange(len(bins)), bins, width=0.8, alpha=0.7, align='center', label='Гистограмма')
# ax.hist(hist_data, bins=bins, alpha=0.7, align='mid', rwidth=0.8, label='Гистограмма')

# Построение кривой
ax.plot(x_curve, y_curve, color='red', linewidth=2, label='Теоретическая кривая')

# Добавление легенды и подписей осей
ax.legend()
ax.set_xlabel('Значение')
ax.set_ylabel('Высота интервалов')

# Отображение графика
plt.show()
