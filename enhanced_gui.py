import sys
import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import psutil
import time
from datetime import datetime

# Импорт нашего ИИ-агента
from ai_agent import PerformanceAI

class AIChatWidget(QWidget):
    """Виджет чата с ИИ"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ai_agent = PerformanceAI()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок ИИ
        header = QLabel("🧠 PerfTune AI Assistant")
        header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #1DB954;
                padding: 10px;
                background: rgba(25, 25, 35, 0.9);
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)
        
        # Область сообщений ИИ
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("""
            QTextEdit {
                background: rgba(20, 20, 30, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: #E0E0E0;
                font-family: 'Segoe UI';
                font-size: 11px;
                padding: 10px;
            }
        """)
        self.chat_area.setMinimumHeight(300)
        layout.addWidget(self.chat_area)
        
        # Индикатор состояния ИИ
        self.ai_status = QLabel("🟢 ИИ активен и анализирует систему")
        self.ai_status.setStyleSheet("""
            QLabel {
                color: #1DB954;
                font-size: 10px;
                padding: 5px;
                background: rgba(29, 185, 84, 0.1);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.ai_status)
        
        # Кнопки быстрых команд
        self.setup_quick_commands(layout)
        
        # Поле ввода
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Задайте вопрос ИИ о производительности...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: rgba(40, 40, 50, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 8px;
                color: white;
                font-size: 11px;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        send_btn = QPushButton("📤")
        send_btn.setFixedWidth(40)
        send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(29, 185, 84, 220),
                    stop:1 rgba(20, 150, 70, 220));
                color: white;
                border-radius: 6px;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(39, 195, 94, 240);
            }
        """)
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(send_btn)
        
        layout.addLayout(input_layout)
        
        # Начальное сообщение
        self.add_ai_message("Привет! Я ваш ИИ-помощник по производительности. Я анализирую систему в реальном времени и могу дать рекомендации.")
    
    def setup_quick_commands(self, layout):
        """Быстрые команды для ИИ"""
        quick_commands = QWidget()
        quick_layout = QHBoxLayout(quick_commands)
        
        commands = [
            ("📊 Анализ системы", "проанализируй систему"),
            ("💾 Проверить память", "проверь использование памяти"),
            ("🚀 Оптимизация", "дай рекомендации по оптимизации"),
            ("📈 Прогноз", "какой прогноз по производительности?")
        ]
        
        for text, cmd in commands:
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(60, 60, 70, 0.8);
                    color: #B3B3B3;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                    padding: 6px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: rgba(80, 80, 90, 0.9);
                    color: white;
                }
            """)
            btn.clicked.connect(lambda checked, c=cmd: self.send_quick_command(c))
            quick_layout.addWidget(btn)
        
        layout.addWidget(quick_commands)
    
    def add_ai_message(self, message):
        """Добавление сообщения от ИИ"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f'<div style="margin: 5px; padding: 8px; border-radius: 8px; background: rgba(29, 185, 84, 0.1);">\
                        <span style="color: #1DB954; font-weight: bold;">🤖 ИИ [{timestamp}]:</span><br>{message}</div>'
        self.chat_area.append(formatted_msg)
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )
    
    def add_user_message(self, message):
        """Добавление сообщения пользователя"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f'<div style="margin: 5px; padding: 8px; border-radius: 8px; background: rgba(70, 70, 90, 0.3);">\
                        <span style="color: #B3B3B3; font-weight: bold;">👤 Вы [{timestamp}]:</span><br>{message}</div>'
        self.chat_area.append(formatted_msg)
    
    def send_message(self):
        """Отправка сообщения ИИ"""
        message = self.input_field.text().strip()
        if not message:
            return
        
        self.add_user_message(message)
        self.input_field.clear()
        
        # Обработка сообщения ИИ
        QTimer.singleShot(500, lambda: self.process_ai_response(message))
    
    def send_quick_command(self, command):
        """Отправка быстрой команды"""
        self.input_field.setText(command)
        self.send_message()
    
    def process_ai_response(self, message):
        """Обработка ответа ИИ"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['анализ', 'анализируй', 'проверь']):
            if any(word in message_lower for word in ['систем', 'систему']):
                response = self.ai_agent.get_system_analysis_report()
            elif any(word in message_lower for word in ['памят', 'memory']):
                response = self.ai_agent.get_memory_analysis()
            else:
                response = "🔍 Что именно вы хотите проанализировать? Систему, память, CPU или что-то другое?"
        
        elif any(word in message_lower for word in ['оптимизац', 'рекомендац', 'совет']):
            response = self.ai_agent.get_optimization_recommendations()
        
        elif any(word in message_lower for word in ['прогноз', 'предсказан', 'ожида']):
            response = self.ai_agent.get_performance_prediction()
        
        elif any(word in message_lower for word in ['привет', 'здравств', 'hello']):
            response = "Привет! Я готов помочь с анализом производительности вашей системы. 🚀"
        
        else:
            response = "🤔 Я специализируюсь на анализе производительности. Спросите меня об оптимизации, анализе системы или дайте рекомендации."
        
        self.add_ai_message(response)
    
    def update_ai_analysis(self, system_data, process_data):
        """Обновление анализа ИИ на основе новых данных"""
        try:
            # Генерация отчета ИИ
            report = self.ai_agent.generate_intelligent_report(system_data, process_data)
            
            # Автоматические оповещения
            if report['system_health']['severity'] == 'critical':
                self.add_ai_message("🚨 КРИТИЧЕСКОЕ СОСТОЯНИЕ СИСТЕМЫ! " + 
                                   report['system_health']['recommendations'][0])
            
            # Обновление статуса
            status_color = "🟢" if report['overall_score'] > 70 else "🟡" if report['overall_score'] > 50 else "🔴"
            self.ai_status.setText(f"{status_color} Оценка системы: {report['overall_score']}/100")
            
        except Exception as e:
            print(f"Ошибка обновления ИИ: {e}")

class AIAnalysisDashboard(QWidget):
    """Дашборд анализа ИИ"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("📊 AI Analysis Dashboard")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #1DB954;
                padding: 10px;
                background: rgba(25, 25, 35, 0.9);
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Метрики в реальном времени
        metrics_layout = QGridLayout()
        
        self.score_label = QLabel("Оценка: --/100")
        self.health_label = QLabel("Состояние: --")
        self.recommendation_label = QLabel("Рекомендация: --")
        
        for label in [self.score_label, self.health_label, self.recommendation_label]:
            label.setStyleSheet("""
                QLabel {
                    background: rgba(40, 40, 50, 0.8);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                    padding: 8px;
                    color: #E0E0E0;
                    font-size: 11px;
                    margin: 2px;
                }
            """)
        
        metrics_layout.addWidget(self.score_label, 0, 0)
        metrics_layout.addWidget(self.health_label, 0, 1)
        metrics_layout.addWidget(self.recommendation_label, 1, 0, 1, 2)
        
        layout.addLayout(metrics_layout)
        
        # График трендов (заглушка)
        trend_label = QLabel("📈 График трендов производительности (в разработке)")
        trend_label.setStyleSheet("""
            QLabel {
                background: rgba(30, 30, 40, 0.9);
                border: 1px dashed rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 20px;
                color: #888;
                font-size: 11px;
                margin-top: 10px;
            }
        """)
        trend_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(trend_label)
        
        # Инсайты ИИ
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setStyleSheet("""
            QTextEdit {
                background: rgba(20, 20, 30, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: #E0E0E0;
                font-size: 10px;
                padding: 10px;
            }
        """)
        self.insights_text.setMaximumHeight(120)
        layout.addWidget(QLabel("💡 Инсайты ИИ:"))
        layout.addWidget(self.insights_text)
    
    def update_dashboard(self, report):
        """Обновление дашборда"""
        try:
            # Обновление оценки
            self.score_label.setText(f"Оценка: {report['overall_score']}/100")
            
            # Обновление состояния
            health_text = f"Состояние: {report['system_health']['severity'].upper()}"
            color = "#1DB954" if report['system_health']['severity'] == 'normal' else "#FF6B6B"
            self.health_label.setText(health_text)
            self.health_label.setStyleSheet(f"color: {color};")
            
            # Обновление рекомендаций
            if report['system_health']['recommendations']:
                rec = report['system_health']['recommendations'][0]
                self.recommendation_label.setText(f"Рекомендация: {rec[:50]}...")
            
            # Обновление инсайтов
            insights = report.get('predictive_analysis', [])
            insights_text = "\n".join(insights) if insights else "Анализирую данные..."
            self.insights_text.setPlainText(insights_text)
            
        except Exception as e:
            print(f"Ошибка обновления дашборда: {e}")

# Интеграция с основным окном (дополнение к существующему коду)
class PerfTuneUltimateWithAI(PerfTuneUltimate):
    """Расширенная версия с ИИ"""
    
    def __init__(self):
        super().__init__()
        self.ai_agent = PerformanceAI()
        self.add_ai_tab()
        
    def add_ai_tab(self):
        """Добавление вкладки ИИ"""
        # Создание вкладки ИИ
        ai_tab = QWidget()
        ai_layout = QHBoxLayout(ai_tab)
        
        # Левая часть - чат с ИИ
        self.ai_chat = AIChatWidget()
        ai_layout.addWidget(self.ai_chat, 2)
        
        # Правая часть - дашборд анализа
        self.ai_dashboard = AIAnalysisDashboard()
        ai_layout.addWidget(self.ai_dashboard, 1)
        
        # Добавление вкладки в существующий интерфейс
        self.tabs.addTab(ai_tab, "🧠 AI Assistant")
    
    def on_sample(self, data):
        """Переопределение обработки данных с интеграцией ИИ"""
        super().on_sample(data)
        
        # Подготовка данных для ИИ
        system_data = {
            'sys_cpu': data['sys_cpu'],
            'sys_mem': data['sys_mem'], 
            'disk_MBps': data['disk_Bps'] / 1e6,
            'net_MBps': data['net_Bps'] / 1e6
        }
        
        process_data = {
            'proc_cpu': data['proc_cpu'],
            'proc_mem': data['proc_mem'],
            'proc_io': data['proc_io']
        }
        
        # Обновление ИИ-анализа
        try:
            report = self.ai_agent.generate_intelligent_report(system_data, process_data)
            self.ai_dashboard.update_dashboard(report)
            self.ai_chat.update_ai_analysis(system_data, process_data)
        except Exception as e:
            print(f"Ошибка ИИ-анализа: {e}")

# Обновленный main.py
def main():
    """Обновленная точка входа"""
    ensure_resources()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    app.setStyle('Fusion')
    
    # Использование улучшенной версии с ИИ
    win = PerfTuneUltimateWithAI()
    win.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()