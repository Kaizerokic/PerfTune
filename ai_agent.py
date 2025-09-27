import json
import numpy as np
import sqlite3
import re
import os
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import random

class SelfLearningNeuralNetwork:
    """Самообучающаяся нейросеть с reinforcement learning"""
    
    def __init__(self, input_size=8, hidden_size=16, output_size=6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Инициализация весов с Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Параметры обучения
        self.learning_rate = 0.01
        self.memory = deque(maxlen=1000)  # Experience replay buffer
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        
        # Трассировки для обратного распространения
        self.cache = {}
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, X):
        """Прямое распространение с сохранением промежуточных значений"""
        self.cache['X'] = X
        
        # Слой 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.cache['z1'] = self.z1
        self.cache['a1'] = self.a1
        
        # Слой 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        self.cache['z2'] = self.z2
        self.cache['a2'] = self.a2
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        """Обратное распространение ошибки"""
        m = X.shape[0]
        
        # Градиенты выходного слоя
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Градиенты скрытого слоя
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Обновление весов
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
        return np.mean(np.abs(dz2))
    
    def predict(self, state):
        """Предсказание на основе состояния системы"""
        try:
            # Нормализация входных данных
            state_normalized = self.normalize_state(state)
            X = np.array(state_normalized).reshape(1, -1)
            
            # Прямое распространение
            prediction = self.forward(X)
            return prediction[0]
        except Exception as e:
            # Возвращаем равномерное распределение в случае ошибки
            return np.ones(self.output_size) / self.output_size
    
    def normalize_state(self, state):
        """Нормализация состояния системы"""
        normalized = []
        
        # CPU usage (0-100) -> (0-1)
        normalized.append(state.get('sys_cpu', 0) / 100.0)
        
        # Memory usage (0-100) -> (0-1)
        normalized.append(state.get('sys_mem', 0) / 100.0)
        
        # Disk activity (0-200 MB/s) -> (0-1)
        disk_mbps = state.get('disk_Bps', 0) / (1024 * 1024)
        normalized.append(min(disk_mbps / 200.0, 1.0))
        
        # Network activity (0-100 MB/s) -> (0-1)
        net_mbps = state.get('net_Bps', 0) / (1024 * 1024)
        normalized.append(min(net_mbps / 100.0, 1.0))
        
        # Process CPU (0-100) -> (0-1)
        normalized.append(state.get('proc_cpu', 0) / 100.0)
        
        # Process Memory (0-100) -> (0-1)
        normalized.append(state.get('proc_mem', 0) / 100.0)
        
        # Process IO (нормализовано)
        proc_io = min(state.get('proc_io', 0) / (1024 * 1024), 10.0)  # До 10 MB
        normalized.append(proc_io / 10.0)
        
        # Время суток (0-1)
        hour = datetime.now().hour
        normalized.append(hour / 24.0)
        
        # Дополняем до нужного размера если необходимо
        while len(normalized) < self.input_size:
            normalized.append(0.0)
            
        return normalized[:self.input_size]
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняем опыт для обучения с подкреплением"""
        self.memory.append((state, action, reward, next_state, done))
    
    def experience_replay(self):
        """Обучение на накопленном опыте"""
        if len(self.memory) < self.batch_size:
            return 0
            
        # Выбираем случайную выборку из памяти
        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0
        
        for state, action_idx, reward, next_state, done in batch:
            # Текущее Q-значение
            current_q = self.predict(state)
            
            # Целевое Q-значение
            if done:
                target = reward
            else:
                next_q = self.predict(next_state)
                target = reward + self.gamma * np.max(next_q)
            
            # Обновляем Q-значение для выбранного действия
            target_q = current_q.copy()
            target_q[action_idx] = target
            
            # Обучаем сеть
            state_normalized = self.normalize_state(state)
            X = np.array(state_normalized).reshape(1, -1)
            loss = self.backward(X, target_q.reshape(1, -1), self.learning_rate)
            total_loss += loss
        
        return total_loss / self.batch_size
    
    def choose_action(self, state, epsilon=0.1):
        """Выбор действия с использованием epsilon-greedy стратегии"""
        if np.random.random() < epsilon:
            # Случайное действие (exploration)
            return random.randint(0, self.output_size - 1)
        else:
            # Жадное действие (exploitation)
            q_values = self.predict(state)
            return np.argmax(q_values)
    
    def save_model(self, filepath):
        """Сохранение обученной модели"""
        model_data = {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath):
        """Загрузка обученной модели"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.W1 = np.array(model_data['W1'])
            self.b1 = np.array(model_data['b1'])
            self.W2 = np.array(model_data['W2'])
            self.b2 = np.array(model_data['b2'])

class AdaptivePerformanceAI:
    """Самообучающийся ИИ-агент"""
    
    def __init__(self):
        self.nn = SelfLearningNeuralNetwork()
        self.history = deque(maxlen=500)
        self.learning_enabled = True
        self.episode_count = 0
        self.total_reward = 0
        self.setup_database()
        
        # Действия, которые может предпринимать ИИ
        self.actions = [
            "monitor_normal",      # 0 - Обычный мониторинг
            "suggest_optimization", # 1 - Предложить оптимизацию
            "alert_cpu_high",       # 2 - Предупредить о высокой загрузке CPU
            "alert_memory_high",    # 3 - Предупредить о высокой загрузке памяти
            "suggest_cleanup",      # 4 - Предложить очистку
            "predict_trend"         # 5 - Спрогнозировать тренд
        ]
        
        # Автоматическое создание базы знаний
        self.knowledge_base = self.create_adaptive_knowledge_base()
        self.model_file = 'models/self_learning_model.json'
        
        # Создаем папку для моделей
        os.makedirs('models', exist_ok=True)
        
        # Пытаемся загрузить предыдущую модель
        self.nn.load_model(self.model_file)
        
        print(f"🤖 ИИ инициализирован. Размер памяти: {len(self.nn.memory)}")
    
    def create_adaptive_knowledge_base(self):
        """Создание самообучающейся базы знаний"""
        return {
            "patterns": defaultdict(list),
            "solutions": defaultdict(list),
            "success_rates": defaultdict(float),
            "learning_stats": {
                "total_decisions": 0,
                "successful_decisions": 0,
                "learning_rate": 0.0
            }
        }
    
    def setup_database(self):
        """Настройка базы данных для обучения"""
        try:
            conn = sqlite3.connect('ai_learning.db')
            c = conn.cursor()
            
            # Таблица для хранения опыта обучения
            c.execute('''CREATE TABLE IF NOT EXISTS learning_experiences
                        (id INTEGER PRIMARY KEY, 
                         timestamp TEXT,
                         state TEXT,
                         action INTEGER,
                         reward REAL,
                         next_state TEXT,
                         success INTEGER)''')
            
            # Таблица для статистики обучения
            c.execute('''CREATE TABLE IF NOT EXISTS learning_stats
                        (id INTEGER PRIMARY KEY,
                         date TEXT,
                         episodes INTEGER,
                         avg_reward REAL,
                         success_rate REAL)''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка создания БД обучения: {e}")
    
    def calculate_reward(self, state, action, next_state):
        """Расчет награды для reinforcement learning"""
        reward = 0.0
        
        # Базовые награды за улучшение состояния
        cpu_improvement = state.get('sys_cpu', 0) - next_state.get('sys_cpu', 0)
        mem_improvement = state.get('sys_mem', 0) - next_state.get('sys_mem', 0)
        
        reward += cpu_improvement * 0.1  # Награда за снижение CPU
        reward += mem_improvement * 0.1  # Награда за снижение памяти
        
        # Награды за конкретные действия
        if action == 1:  # suggest_optimization
            if next_state.get('sys_cpu', 0) < state.get('sys_cpu', 0):
                reward += 2.0
        elif action == 2:  # alert_cpu_high
            if state.get('sys_cpu', 0) > 80:
                reward += 1.5
        elif action == 3:  # alert_memory_high
            if state.get('sys_mem', 0) > 80:
                reward += 1.5
        elif action == 4:  # suggest_cleanup
            if next_state.get('sys_mem', 0) < state.get('sys_mem', 0):
                reward += 1.0
        
        # Штраф за неправильные действия
        if action == 2 and state.get('sys_cpu', 0) < 30:
            reward -= 1.0
        if action == 3 and state.get('sys_mem', 0) < 30:
            reward -= 1.0
            
        return max(-5.0, min(5.0, reward))  # Ограничиваем награду
    
    def learn_from_experience(self, state, action, next_state):
        """Обучение на основе нового опыта"""
        if not self.learning_enabled:
            return
            
        # Расчет награды
        reward = self.calculate_reward(state, action, next_state)
        
        # Сохраняем опыт
        self.nn.remember(state, action, reward, next_state, False)
        
        # Обучение на накопленном опыте
        loss = self.nn.experience_replay()
        
        # Обновление статистики
        self.total_reward += reward
        self.episode_count += 1
        
        # Сохранение модели каждые 100 эпизодов
        if self.episode_count % 100 == 0:
            self.nn.save_model(self.model_file)
            self.save_learning_stats()
            
        # Обновление базы знаний
        self.update_knowledge_base(state, action, reward > 0)
        
        if self.episode_count % 50 == 0:
            print(f"📚 ИИ обучился на {self.episode_count} эпизодах. Средняя награда: {self.total_reward/self.episode_count:.2f}")
    
    def update_knowledge_base(self, state, action, was_successful):
        """Обновление базы знаний на основе опыта"""
        action_name = self.actions[action]
        
        # Анализ состояния системы
        state_features = []
        if state.get('sys_cpu', 0) > 70:
            state_features.append('high_cpu')
        if state.get('sys_mem', 0) > 70:
            state_features.append('high_memory')
        if state.get('disk_Bps', 0) > 50 * 1024 * 1024:
            state_features.append('high_disk')
            
        # Обновление паттернов
        for feature in state_features:
            self.knowledge_base['patterns'][feature].append(action_name)
            
        # Обновление успешности решений
        key = f"{action_name}_{'_'.join(state_features)}"
        if was_successful:
            self.knowledge_base['success_rates'][key] = min(1.0, 
                self.knowledge_base['success_rates'].get(key, 0) + 0.1)
        else:
            self.knowledge_base['success_rates'][key] = max(0.0,
                self.knowledge_base['success_rates'].get(key, 0) - 0.05)
        
        # Обновление статистики обучения
        self.knowledge_base['learning_stats']['total_decisions'] += 1
        if was_successful:
            self.knowledge_base['learning_stats']['successful_decisions'] += 1
            
        success_rate = (self.knowledge_base['learning_stats']['successful_decisions'] / 
                       max(1, self.knowledge_base['learning_stats']['total_decisions']))
        self.knowledge_base['learning_stats']['learning_rate'] = success_rate
    
    def analyze_and_act(self, system_data, process_data):
        """Анализ системы и принятие решений"""
        combined_data = {**system_data, **process_data}
        
        # Сохраняем в историю
        previous_state = self.history[-1] if self.history else combined_data
        self.history.append(combined_data)
        
        # Выбор действия с помощью нейросети
        action_idx = self.nn.choose_action(combined_data, 
                                         epsilon=max(0.01, 0.1 - self.episode_count * 0.001))
        action = self.actions[action_idx]
        
        # Обучение на основе предыдущего опыта
        if len(self.history) >= 2:
            self.learn_from_experience(previous_state, action_idx, combined_data)
        
        # Генерация ответа на основе выбранного действия
        response = self.generate_response(action, combined_data, previous_state)
        
        return {
            'action': action,
            'response': response,
            'confidence': float(np.max(self.nn.predict(combined_data))),
            'learning_stats': self.knowledge_base['learning_stats']
        }
    
    def generate_response(self, action, current_state, previous_state):
        """Генерация интеллектуального ответа"""
        responses = {
            "monitor_normal": self.get_normal_monitoring_response(current_state),
            "suggest_optimization": self.get_optimization_suggestions(current_state),
            "alert_cpu_high": self.get_cpu_alert_response(current_state),
            "alert_memory_high": self.get_memory_alert_response(current_state),
            "suggest_cleanup": self.get_cleanup_suggestions(current_state),
            "predict_trend": self.get_trend_prediction(current_state, previous_state)
        }
        
        return responses.get(action, "🤔 Анализирую ситуацию...")
    
    def get_normal_monitoring_response(self, state):
        """Ответ при нормальном мониторинге"""
        cpu = state.get('sys_cpu', 0)
        mem = state.get('sys_mem', 0)
        
        if cpu < 30 and mem < 50:
            return f"✅ Система работает отлично! CPU: {cpu:.1f}%, Память: {mem:.1f}%"
        else:
            return f"📊 Система стабильна. CPU: {cpu:.1f}%, Память: {mem:.1f}%"
    
    def get_optimization_suggestions(self, state):
        """Предложения по оптимизации"""
        suggestions = []
        cpu = state.get('sys_cpu', 0)
        mem = state.get('sys_mem', 0)
        
        if cpu > 60:
            suggestions.append("💡 Рассмотрите оптимизацию процессов с высоким CPU")
        if mem > 65:
            suggestions.append("💡 Проверьте использование памяти приложениями")
        if state.get('disk_Bps', 0) > 20 * 1024 * 1024:
            suggestions.append("💡 Высокая дисковая активность - возможно кэширование")
            
        if suggestions:
            return "🚀 **Рекомендации по оптимизации:**\n• " + "\n• ".join(suggestions)
        else:
            return "✅ Система хорошо оптимизирована. Продолжайте в том же духе!"
    
    def get_cpu_alert_response(self, state):
        """Предупреждение о высокой загрузке CPU"""
        cpu = state.get('sys_cpu', 0)
        if cpu > 80:
            return f"🚨 **ВНИМАНИЕ:** Высокая загрузка CPU ({cpu:.1f}%)\n💡 Рекомендуется проверить фоновые процессы"
        else:
            return f"⚠️ Загрузка CPU: {cpu:.1f}% - в пределах нормы"
    
    def get_memory_alert_response(self, state):
        """Предупреждение о высокой загрузке памяти"""
        mem = state.get('sys_mem', 0)
        if mem > 80:
            return f"🚨 **ВНИМАНИЕ:** Высокое использование памяти ({mem:.1f}%)\n💡 Проверьте наличие утечек памяти"
        else:
            return f"⚠️ Использование памяти: {mem:.1f}% - нормально"
    
    def get_cleanup_suggestions(self, state):
        """Предложения по очистке"""
        return "🧹 **Рекомендации по очистке:**\n• Очистите временные файлы\n• Закройте неиспользуемые приложения\n• Проверьте автозагрузку"
    
    def get_trend_prediction(self, current_state, previous_state):
        """Прогнозирование трендов"""
        if len(self.history) < 10:
            return "📈 Собираю данные для анализа трендов..."
        
        # Анализ изменений
        cpu_change = current_state.get('sys_cpu', 0) - previous_state.get('sys_cpu', 0)
        mem_change = current_state.get('sys_mem', 0) - previous_state.get('sys_mem', 0)
        
        trend = "стабильна"
        if cpu_change > 5:
            trend = "растет нагрузка на CPU"
        elif cpu_change < -5:
            trend = "снижается нагрузка на CPU"
        elif mem_change > 3:
            trend = "растет использование памяти"
            
        return f"🔮 **Прогноз:** Система {trend}. На основе анализа {len(self.history)} точек данных."
    
    def get_learning_report(self):
        """Отчет о процессе обучения"""
        stats = self.knowledge_base['learning_stats']
        success_rate = stats['learning_rate'] * 100
        
        report = f"""📚 **Отчет об обучении ИИ:**

• Всего решений принято: {stats['total_decisions']}
• Успешных решений: {stats['successful_decisions']}
• Уровень успеха: {success_rate:.1f}%
• Эпизодов обучения: {self.episode_count}
• Средняя награда: {self.total_reward/max(1, self.episode_count):.2f}

🤖 **Состояние ИИ:** {'Обучается активно' if success_rate > 60 else 'Нужно больше данных'}"""

        return report
    
    def save_learning_stats(self):
        """Сохранение статистики обучения"""
        try:
            conn = sqlite3.connect('ai_learning.db')
            c = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            avg_reward = self.total_reward / max(1, self.episode_count)
            success_rate = self.knowledge_base['learning_stats']['learning_rate']
            
            c.execute('''INSERT INTO learning_stats 
                        (date, episodes, avg_reward, success_rate) 
                        VALUES (?, ?, ?, ?)''',
                     (today, self.episode_count, avg_reward, success_rate))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка сохранения статистики: {e}")
    
    def interactive_learning(self, user_feedback, state, action):
        """Интерактивное обучение на основе feedback пользователя"""
        if user_feedback.lower() in ['хорошо', 'отлично', 'good', 'well']:
            reward = 2.0  # Положительная награда
            self.update_knowledge_base(state, action, True)
        elif user_feedback.lower() in ['плохо', 'неправильно', 'bad', 'wrong']:
            reward = -2.0  # Отрицательная награда
            self.update_knowledge_base(state, action, False)
        else:
            reward = 0.5  # Нейтральная награда за взаимодействие
            
        # Сохраняем опыт для обучения
        self.nn.remember(state, action, reward, state, True)
        self.nn.experience_replay()
        
        return f"📝 Спасибо за обратную связь! ИИ учится..."