# Крок 1: Імпорт необхідних бібліотек
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Крок 2: Генерація симульованих даних
np.random.seed(0)  # Задаємо випадковий стан для відтворюваності
n_samples = 1000
n_features = 3

# Генерація числових даних
X_numeric = np.random.rand(n_samples, n_features)

# Визначення категорійних значень
work_types = ['employed', 'student', 'retired', 'unemployed']
residence_types = ['city', 'rural', 'suburb']
loan_purposes = ['auto loan', 'education', 'home purchase', 'medical expenses']

# Кодування категорійних даних у бінарний формат
work_type_encoded = pd.get_dummies(np.random.choice(work_types, size=n_samples), prefix='work_type')
residence_type_encoded = pd.get_dummies(np.random.choice(residence_types, size=n_samples), prefix='residence_type')
loan_purpose_encoded = pd.get_dummies(np.random.choice(loan_purposes, size=n_samples), prefix='loan_purpose')

# Об'єднання закодованих категорійних даних
X_categorical_encoded = np.column_stack((work_type_encoded, residence_type_encoded, loan_purpose_encoded))

# Об'єднання числових та закодованих категорійних даних
X = np.concatenate((X_numeric, X_categorical_encoded), axis=1)

# Генерація випадкових міток для цільової змінної
y = np.random.choice([0, 1], size=n_samples)

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Крок 3: Вибір та навчання моделі SVM
# Встановлення гіперпараметрів для решітчатого пошуку
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf', 'poly']}

# Створення моделі SVM
svm_model = SVC(random_state=41)

# Застосування решітчатого пошуку для знаходження оптимальних гіперпараметрів
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Вибір найкращої моделі після решітчатого пошуку
best_svm_model = grid_search.best_estimator_

# Навчання найкращої моделі (не обов'язково, оскільки модель вже навчена решітчатим пошуком)
best_svm_model.fit(X_train, y_train)

# Крок 4: Оцінка якості моделі
# Передбачення на тестовому наборі
y_pred = best_svm_model.predict(X_test)

# Обчислення метрик якості
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Виведення результатів оцінки
print("Best Model Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

# Крок 5: Візуалізація результатів передбачення
plt.figure(figsize=(8, 6))
plt.hist(y_pred, bins=[-0.5, 0.5, 1.5], alpha=0.5, color='green', edgecolor='black')
plt.xticks([0, 1], ['Rejected', 'Approved'])
plt.xlabel('Credit Approval')
plt.ylabel('Count')
plt.title('Distribution of Credit Approvals')
plt.show()
