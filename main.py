import sys
import os
import time
import math
import json
import platform
import shutil
import subprocess
import traceback
from collections import deque
from datetime import datetime

# Third-party imports
try:
    import psutil
except Exception as e:
    print("psutil required: pip install psutil")
    raise

try:
    import numpy as np
except Exception:
    print("numpy required: pip install numpy")
    raise

# Optional libs
try:
    import plotly.graph_objs as go
    import plotly.offline as po
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# GUI imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QGroupBox, QTextEdit, QListWidget, QListWidgetItem,
    QGridLayout, QFrame, QMessageBox, QFileDialog, QSizePolicy, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QBrush, QPixmap, QCursor
from PyQt6.QtWidgets import QGraphicsBlurEffect, QGraphicsDropShadowEffect, QGraphicsOpacityEffect

# Импорт нашего ИИ-агента
from ai_agent import AdaptivePerformanceAI

IS_ROOT = os.geteuid() == 0 if hasattr(os, "geteuid") else False
APP_VERSION = "4.0 AI POWERED"

# ----------------- Resources -----------------
RES_DIR = os.path.join(os.path.dirname(__file__), "resources")
ICON_FILES = {
    "cpu.svg": """<svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="7" y="7" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1.5" />
<path d="M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
</svg>""",
    "ram.svg": """<svg width="64" height="64" viewBox="0 0 24 24" fill="none">
<rect x="3" y="7" width="18" height="10" rx="2" stroke="currentColor" stroke-width="1.5"/>
<rect x="7" y="10" width="3" height="4" fill="currentColor"/>
<rect x="14" y="10" width="3" height="4" fill="currentColor"/>
</svg>""",
    "disk.svg": """<svg width="64" height="64" viewBox="0 0 24 24" fill="none">
<ellipse cx="12" cy="8" rx="7" ry="3" stroke="currentColor" stroke-width="1.5"/>
<path d="M5 8v6c0 1.657 3.134 3 7 3s7-1.343 7-3V8" stroke="currentColor" stroke-width="1.5"/>
<circle cx="12" cy="11" r="1" fill="currentColor"/>
</svg>""",
    "net.svg": """<svg width="64" height="64" viewBox="0 0 24 24" fill="none">
<path d="M3 12h6M15 12h6M9 6l3 6 3-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>""",
    "logo.svg": """<svg width="128" height="32" viewBox="0 0 240 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="4" y="4" width="232" height="56" rx="10" fill="#0f1720"/>
<text x="28" y="40" fill="#1DB954" font-family="Segoe UI" font-size="22" font-weight="700">PerfTune AI</text>
</svg>"""
}

def ensure_resources():
    try:
        os.makedirs(RES_DIR, exist_ok=True)
        for name, svg in ICON_FILES.items():
            path = os.path.join(RES_DIR, name)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(svg)
    except Exception as e:
        print(f"Error creating resources: {e}")

# ----------------- Helpers -----------------
def safe_write_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f"Error writing JSON to {path}: {e}")

def safe_read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON from {path}: {e}")
        return None

# ----------------- EMASmoother -----------------
class EMASmoother:
    def __init__(self, alpha=0.2, initial=0.0):
        self.alpha = float(alpha)
        self.value = float(initial)
        self.initialized = False

    def update(self, sample):
        try:
            s = float(sample)
        except Exception:
            s = 0.0
        if not self.initialized:
            self.value = s
            self.initialized = True
        else:
            self.value = self.alpha * s + (1 - self.alpha) * self.value
        return self.value

# ----------------- Sampler Thread -----------------
class Sampler(QThread):
    sample = pyqtSignal(dict)
    log = pyqtSignal(str)

    def __init__(self, pid=None, interval=0.18):
        super().__init__()
        self._running = True
        self.interval = max(0.05, float(interval))
        self.pid = pid
        self.prev_net = None
        self.prev_disk = None

    def run(self):
        while self._running:
            ts = time.time()
            try:
                data = self.collect_sample_data(ts)
                self.sample.emit(data)
            except Exception as e:
                self.log.emit(f"Sampler exception: {str(e)}\n{traceback.format_exc()}")
                self.sample.emit(self.get_fallback_data(ts))
            
            slept = 0.0
            while self._running and slept < self.interval:
                time.sleep(0.01)
                slept += 0.01

    def collect_sample_data(self, ts):
        data = {'timestamp': ts}
        
        if self.pid and psutil.pid_exists(self.pid):
            try:
                p = psutil.Process(self.pid)
                data['proc_cpu'] = p.cpu_percent(interval=None)
                data['proc_mem'] = p.memory_percent()
                try:
                    io = p.io_counters()
                    data['proc_io'] = (io.read_bytes + io.write_bytes)
                except Exception:
                    data['proc_io'] = 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                data.update({'proc_cpu': 0.0, 'proc_mem': 0.0, 'proc_io': 0})
        else:
            data.update({'proc_cpu': 0.0, 'proc_mem': 0.0, 'proc_io': 0})

        data['sys_cpu'] = psutil.cpu_percent(interval=None)
        data['sys_mem'] = psutil.virtual_memory().percent

        net = psutil.net_io_counters()
        total_net = net.bytes_sent + net.bytes_recv
        if self.prev_net is not None:
            data['net_Bps'] = (total_net - self.prev_net) / max(0.01, self.interval)
        else:
            data['net_Bps'] = 0.0
        self.prev_net = total_net

        disk = psutil.disk_io_counters()
        total_disk = (disk.read_bytes + disk.write_bytes) if disk else 0
        if self.prev_disk is not None:
            data['disk_Bps'] = (total_disk - self.prev_disk) / max(0.01, self.interval)
        else:
            data['disk_Bps'] = 0.0
        self.prev_disk = total_disk

        return data

    def get_fallback_data(self, ts):
        return {
            'timestamp': ts, 'proc_cpu': 0.0, 'proc_mem': 0.0, 'proc_io': 0,
            'sys_cpu': 0.0, 'sys_mem': 0.0, 'net_Bps': 0.0, 'disk_Bps': 0.0
        }

    def stop(self):
        self._running = False
        self.wait(2000)

# ----------------- SmoothGauge -----------------
class SmoothGauge(QLabel):
    def __init__(self, title, max_value=100.0, unit="", color="#1DB954"):
        super().__init__()
        self.title = title
        self.max_value = float(max_value) if max_value else 100.0
        self.unit = unit
        self.color = QColor(color)
        self._display = 0.0
        self._target = 0.0
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.shadow.setOffset(3, 3)
        self.setGraphicsEffect(self.shadow)

    def set_target(self, v):
        try:
            self._target = max(0.0, min(float(v), self.max_value))
        except Exception:
            self._target = 0.0

    def tick(self, dt=0.016):
        tau = 0.15
        alpha = 1 - math.exp(-dt / tau) if tau > 0 else 1.0
        self._display += alpha * (self._target - self._display)
        self.update()

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        side = min(r.width(), r.height())
        cx, cy = r.center().x(), r.center().y()
        rad = side // 2 - 10

        bg_gradient = QLinearGradient(0, 0, self.width(), self.height())
        bg_gradient.setColorAt(0.0, QColor(40, 40, 40, 180))
        bg_gradient.setColorAt(1.0, QColor(20, 20, 20, 180))
        painter.setBrush(QBrush(bg_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - rad, cy - rad, rad * 2, rad * 2)

        progress = (self._display / self.max_value) if self.max_value else 0.0
        start_angle = 90 * 16
        span = int(-progress * 360 * 16)
        
        progress_gradient = QLinearGradient(0, 0, self.width(), self.height())
        progress_gradient.setColorAt(0.0, self.color.lighter(120))
        progress_gradient.setColorAt(0.7, self.color)
        progress_gradient.setColorAt(1.0, self.color.darker(120))
        
        pen = QPen(QBrush(progress_gradient), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(cx - rad, cy - rad, rad * 2, rad * 2, start_angle, span)

        painter.setPen(QColor(240, 240, 240))
        painter.setFont(QFont("Segoe UI", max(10, side // 12), QFont.Weight.Bold))
        txt = f"{self._display:.1f}{self.unit}"
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)

        painter.setFont(QFont("Segoe UI", max(7, side // 20)))
        painter.setPen(QColor(170, 170, 170))
        painter.drawText(0, r.height() - 20, r.width(), 20, Qt.AlignmentFlag.AlignCenter, self.title)

# ----------------- QuickAnalyzer -----------------
class QuickAnalyzer(QThread):
    finished = pyqtSignal(dict)
    log = pyqtSignal(str)

    def __init__(self, pid, duration=15):
        super().__init__()
        self.pid = pid
        self.duration = max(2, int(duration))

    def run(self):
        try:
            cpu_samples = []
            mem_samples = []
            io_samples = []
            timestamps = []
            
            t0 = time.time()
            sample_count = 0
            max_samples = self.duration * 4
            
            while time.time() - t0 < self.duration and sample_count < max_samples:
                if not psutil.pid_exists(self.pid):
                    break
                    
                try:
                    p = psutil.Process(self.pid)
                    cpu_samples.append(p.cpu_percent(interval=0.1))
                    mem_samples.append(p.memory_percent())
                    try:
                        io = p.io_counters()
                        io_samples.append(io.read_bytes + io.write_bytes)
                    except Exception:
                        io_samples.append(0)
                    timestamps.append(time.time())
                    sample_count += 1
                    time.sleep(0.15)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.log.emit(f"QuickAnalyzer: access error: {e}")
                    break
                except Exception as e:
                    self.log.emit(f"QuickAnalyzer error: {e}")
                    break

            res = self.analyze_samples(cpu_samples, mem_samples, io_samples, timestamps)
            self.finished.emit(res)
        except Exception as e:
            self.log.emit(f"QuickAnalyzer fatal: {e}")
            self.finished.emit({'error': str(e), 'recs': ["Analysis failed"]})

    def analyze_samples(self, cpu_samples, mem_samples, io_samples, timestamps):
        if not cpu_samples:
            return {'avg_cpu': 0.0, 'avg_mem': 0.0, 'disk_trend_Bps': 0.0, 'recs': ["No data collected"]}

        avg_cpu = float(np.mean(cpu_samples))
        avg_mem = float(np.mean(mem_samples))
        max_cpu = float(np.max(cpu_samples))
        max_mem = float(np.max(mem_samples))
        
        cpu_trend = float(np.polyfit(range(len(cpu_samples)), cpu_samples, 1)[0]) if len(cpu_samples) > 1 else 0
        mem_trend = float(np.polyfit(range(len(mem_samples)), mem_samples, 1)[0]) if len(mem_samples) > 1 else 0
        
        disk_trend_Bps = 0.0
        if len(io_samples) > 1:
            total_duration = timestamps[-1] - timestamps[0] if timestamps else self.duration
            disk_trend_Bps = (io_samples[-1] - io_samples[0]) / max(0.1, total_duration)

        recs = []
        if avg_cpu > 85 or max_cpu > 95:
            recs.append("🚨 High CPU usage: Profile hot code paths and consider optimization")
        elif cpu_trend > 0.5:
            recs.append("📈 CPU usage trending upward: Monitor for potential issues")
        if avg_mem > 80 or max_mem > 90:
            recs.append("🚨 High memory usage: Check for memory leaks")
        elif mem_trend > 0.3:
            recs.append("📈 Memory usage trending upward: Potential leak detected")
        if disk_trend_Bps > 10 * 1024 * 1024:
            recs.append("💾 High disk I/O: Consider optimizing file operations")
        if not recs:
            recs.append("✅ No critical issues detected in quick scan")

        return {
            'avg_cpu': avg_cpu,
            'avg_mem': avg_mem,
            'max_cpu': max_cpu,
            'max_mem': max_mem,
            'cpu_trend': cpu_trend,
            'mem_trend': mem_trend,
            'disk_trend_Bps': disk_trend_Bps,
            'recs': recs
        }

# ----------------- AI Chat Widget -----------------
class AIChatWidget(QWidget):
    def __init__(self, parent=None, ai_agent=None):
        super().__init__(parent)
        self.ai_agent = ai_agent
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
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f'<div style="margin: 5px; padding: 8px; border-radius: 8px; background: rgba(29, 185, 84, 0.1);">\
                        <span style="color: #1DB954; font-weight: bold;">🤖 ИИ [{timestamp}]:</span><br>{message}</div>'
        self.chat_area.append(formatted_msg)
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )
    
    def add_user_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f'<div style="margin: 5px; padding: 8px; border-radius: 8px; background: rgba(70, 70, 90, 0.3);">\
                        <span style="color: #B3B3B3; font-weight: bold;">👤 Вы [{timestamp}]:</span><br>{message}</div>'
        self.chat_area.append(formatted_msg)
    
    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return
        
        self.add_user_message(message)
        self.input_field.clear()
        
        QTimer.singleShot(500, lambda: self.process_ai_response(message))
    
    def send_quick_command(self, command):
        self.input_field.setText(command)
        self.send_message()
    
    def process_ai_response(self, message):
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['анализ', 'анализируй', 'проверь']):
            if any(word in message_lower for word in ['систем', 'систему']):
                response = "🔍 Анализирую общее состояние системы...\n\n📊 **Текущие метрики:**\n- Загрузка CPU: анализирую\n- Использование памяти: проверяю\n- Дисковые операции: мониторю\n- Сетевая активность: оцениваю\n\n💡 **Рекомендации будут готовы через несколько секунд непрерывного мониторинга.**"
            elif any(word in message_lower for word in ['памят', 'memory']):
                response = "💾 Анализирую использование памяти...\n\n**Что проверяю:**\n- Общее потребление RAM\n- Потенциальные утечки памяти\n- Эффективность использования\n- Swapping активность\n\n📈 Данные собираются в реальном времени."
            else:
                response = "🔍 Что именно вы хотите проанализировать? Систему, память, CPU или что-то другое?"
        
        elif any(word in message_lower for word in ['оптимизац', 'рекомендац', 'совет']):
            response = "🚀 **Рекомендации по оптимизации:**\n\n1. **Для CPU:** ограничьте фоновые процессы\n2. **Для памяти:** закройте неиспользуемые приложения\n3. **Для диска:** используйте SSD для лучшей производительности\n4. **Общее:** регулярно обновляйте систему и драйверы"
        
        elif any(word in message_lower for word in ['прогноз', 'предсказан', 'ожида']):
            response = "📈 **Прогноз производительности:**\n\nНа основе текущих трендов система должна оставаться стабильной. Рекомендую продолжить мониторинг для более точных предсказаний."
        
        elif any(word in message_lower for word in ['привет', 'здравств', 'hello']):
            response = "Привет! Я ваш ИИ-помощник по производительности! 🚀\n\nЯ могу:\n- 📊 Анализировать систему в реальном времени\n- 💾 Давать рекомендации по оптимизации\n- 📈 Предсказывать тренды производительности\n- 🚀 Помогать с настройкой системы\n\nЧем могу помочь?"
        
        else:
            response = "🤔 Я специализируюсь на анализе производительности. Спросите меня об:\n- 📊 Анализе системы\n- 💾 Оптимизации памяти\n- 🚀 Советах по настройке\n- 📈 Прогнозах производительности"
        
        self.add_ai_message(response)
    
    def update_ai_analysis(self, system_data, process_data):
        try:
            if system_data.get('sys_cpu', 0) > 80:
                self.add_ai_message("⚠️ **ВНИМАНИЕ:** Высокая загрузка CPU! Рекомендую проверить процессы.")
            if system_data.get('sys_mem', 0) > 75:
                self.add_ai_message("⚠️ **ВНИМАНИЕ:** Высокое использование памяти! Возможна оптимизация.")
                
            status_color = "🟢" if system_data.get('sys_cpu', 0) < 70 else "🟡" if system_data.get('sys_cpu', 0) < 85 else "🔴"
            self.ai_status.setText(f"{status_color} ИИ анализирует: CPU {system_data.get('sys_cpu', 0):.1f}%")
            
        except Exception as e:
            print(f"Ошибка обновления ИИ: {e}")
    def add_learning_features(self):
        """Добавление элементов для интерактивного обучения"""
        learning_layout = QHBoxLayout()
        
        # Кнопка отчета об обучении
        self.btn_learning_report = QPushButton("📚 Отчет об обучении")
        self.btn_learning_report.setStyleSheet("""
            QPushButton {
                background: rgba(70, 70, 90, 0.8);
                color: #B3B3B3;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                background: rgba(90, 90, 110, 0.9);
            }
        """)
        self.btn_learning_report.clicked.connect(self.show_learning_report)
        
        # Кнопки обратной связи
        self.btn_feedback_good = QPushButton("👍")
        self.btn_feedback_bad = QPushButton("👎")
        
        for btn in [self.btn_feedback_good, self.btn_feedback_bad]:
            btn.setFixedWidth(40)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(60, 60, 70, 0.8);
                    border-radius: 4px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: rgba(80, 80, 90, 0.9);
                }
            """)
        
        self.btn_feedback_good.clicked.connect(lambda: self.give_feedback("хорошо"))
        self.btn_feedback_bad.clicked.connect(lambda: self.give_feedback("плохо"))
        
        learning_layout.addWidget(self.btn_learning_report)
        learning_layout.addWidget(self.btn_feedback_good)
        learning_layout.addWidget(self.btn_feedback_bad)
        learning_layout.addStretch()
        
        # Добавляем перед полем ввода
        self.layout().insertLayout(self.layout().count() - 2, learning_layout)
    
    def give_feedback(self, feedback):
        """Обратная связь для обучения ИИ"""
        if self.ai_agent and hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            response = self.ai_agent.interactive_learning(
                feedback, self.last_state, self.last_action)
            self.add_ai_message(response)
        else:
            self.add_ai_message("🤔 Нет данных для обучения. Продолжайте использовать систему.")
    
    def show_learning_report(self):
        """Показать отчет об обучении"""
        if self.ai_agent:
            report = self.ai_agent.get_learning_report()
            self.add_ai_message(report)
        else:
            self.add_ai_message("🤔 ИИ-агент недоступен.")
    
    def update_ai_analysis(self, system_data, process_data):
        """Обновление анализа с сохранением состояния для обучения"""
        try:
            if not self.ai_agent:
                return
                
            combined_data = {**system_data, **process_data}
            
            # Анализ и действие ИИ
            ai_result = self.ai_agent.analyze_and_act(system_data, process_data)
            
            # Сохраняем для обратной связи
            self.last_state = combined_data
            self.last_action = self.ai_agent.actions.index(ai_result['action'])
            
            # Автоматические оповещения для критических состояний
            if system_data.get('sys_cpu', 0) > 85 or system_data.get('sys_mem', 0) > 85:
                self.add_ai_message(ai_result['response'])
                
            # Обновление статуса с информацией об обучении
            stats = ai_result['learning_stats']
            success_rate = stats['learning_rate'] * 100
            
            status_text = f"🟢 ИИ обучается | Успех: {success_rate:.1f}% | Решения: {stats['total_decisions']}"
            self.ai_status.setText(status_text)
            
        except Exception as e:
            print(f"Ошибка обновления ИИ: {e}")

# ----------------- AI Analysis Dashboard -----------------
class AIAnalysisDashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
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
        
        trend_label = QLabel("📈 График трендов производительности\n(активируется после 10 секунд мониторинга)")
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
    
    def update_dashboard(self, system_data, process_data):
        try:
            # Расчет оценки системы
            cpu_score = max(0, 100 - system_data.get('sys_cpu', 0))
            mem_score = max(0, 100 - system_data.get('sys_mem', 0))
            overall_score = int((cpu_score + mem_score) / 2)
            
            self.score_label.setText(f"Оценка: {overall_score}/100")
            
            # Определение состояния
            if overall_score > 80:
                health_status = "ОТЛИЧНО"
                color = "#1DB954"
            elif overall_score > 60:
                health_status = "ХОРОШО" 
                color = "#FFA500"
            else:
                health_status = "ВНИМАНИЕ"
                color = "#FF6B6B"
                
            self.health_label.setText(f"Состояние: {health_status}")
            self.health_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            # Рекомендации
            recommendations = []
            if system_data.get('sys_cpu', 0) > 70:
                recommendations.append("Оптимизировать CPU использование")
            if system_data.get('sys_mem', 0) > 70:
                recommendations.append("Освободить память")
            if system_data.get('disk_Bps', 0) > 50 * 1024 * 1024:
                recommendations.append("Проверить дисковую активность")
                
            rec_text = " | ".join(recommendations) if recommendations else "Система в норме"
            self.recommendation_label.setText(f"Рекомендация: {rec_text}")
            
            # Инсайты
            insights = []
            if system_data.get('sys_cpu', 0) < 30 and process_data.get('proc_cpu', 0) > 50:
                insights.append("🎯 Процесс использует много CPU при низкой системной нагрузке")
            if system_data.get('sys_mem', 0) > 80:
                insights.append("💾 Высокое использование памяти - возможна оптимизация")
                
            insights_text = "\n".join(insights) if insights else "✅ Система работает стабильно"
            self.insights_text.setPlainText(insights_text)
            
        except Exception as e:
            print(f"Ошибка обновления дашборда: {e}")

# ----------------- Enhanced Main Window with AI -----------------
class PerfTuneUltimate(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_resources()
        self.setWindowTitle(f"PerfTune AI — {APP_VERSION}")
        self.resize(1400, 900)
        self.setFont(QFont("Segoe UI", 9))

        self.target_pid = None
        self.sampler = None
        self.last = None
        self.ai_agent = AdaptivePerformanceAI()

        self.smoothers = {
            'proc_cpu': EMASmoother(0.2),
            'proc_mem': EMASmoother(0.15),
            'disk_MBps': EMASmoother(0.25),
            'net_MBps': EMASmoother(0.25),
            'sys_cpu': EMASmoother(0.15),
            'sys_mem': EMASmoother(0.15)
        }

        self.history = deque(maxlen=1500)
        self.baseline_path = os.path.join(RES_DIR, "baseline.json")
        self.rollback_path = os.path.join(RES_DIR, "rollback.json")

        self.init_ui()
        self.setup_timers()
        self.start_sampler()
        self.log("PerfTune AI Powered initialized")

    def get_system_info(self):
        try:
            system = platform.system()
            release = platform.release()
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total // (1024**3)
            load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            return f"{system} {release} | Cores: {cpu_count} | Memory: {memory_gb}GB | Load: {load:.2f}"
        except Exception as e:
            self.log(f"Error getting system info: {e}")
            return "Unknown system"

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(15)
        root.setContentsMargins(15, 15, 15, 15)

        left = QWidget()
        l_layout = QVBoxLayout(left)
        l_layout.setSpacing(10)

        self.sys_info = QLabel(self.get_system_info())
        self.sys_info.setStyleSheet("color: #B3B3B3; font-size: 11px;")
        l_layout.addWidget(self.sys_info)

        proc_group = QGroupBox("Process Selection")
        proc_group.setStyleSheet("""
            QGroupBox {
                color: #1DB954;
                font-weight: bold;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        pg_layout = QVBoxLayout(proc_group)
        h = QHBoxLayout()
        self.pid_input = QLineEdit()
        self.pid_input.setPlaceholderText("PID or process name...")
        self.pid_input.setStyleSheet("""
            QLineEdit {
                background: rgba(40, 40, 50, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 6px;
                color: white;
            }
        """)
        self.pid_input.returnPressed.connect(self.on_pid_search)
        h.addWidget(self.pid_input)
        
        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedWidth(40)
        refresh_btn.setStyleSheet(self.get_button_style(light=True))
        refresh_btn.clicked.connect(self.refresh_proc_list)
        h.addWidget(refresh_btn)
        pg_layout.addLayout(h)

        self.proc_list = QListWidget()
        self.proc_list.setFixedHeight(200)
        self.proc_list.setStyleSheet("""
            QListWidget {
                background: rgba(30, 30, 40, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                color: white;
            }
            QListWidget::item {
                padding: 2px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }
            QListWidget::item:selected {
                background: rgba(29, 185, 84, 0.3);
            }
        """)
        self.proc_list.itemDoubleClicked.connect(self.on_proc_selected)
        pg_layout.addWidget(self.proc_list)
        l_layout.addWidget(proc_group)

        # Tabs including AI tab
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                background: rgba(30, 30, 40, 0.8);
            }
            QTabBar::tab {
                background: rgba(40, 40, 50, 0.8);
                color: #B3B3B3;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: rgba(29, 185, 84, 0.3);
                color: white;
            }
        """)
        
        tab_rec = QTextEdit()
        tab_rec.setReadOnly(True)
        tab_log = QTextEdit()
        tab_log.setReadOnly(True)
        
        text_style = """
            QTextEdit {
                background: rgba(25, 25, 35, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                color: #E0E0E0;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """
        tab_rec.setStyleSheet(text_style)
        tab_log.setStyleSheet(text_style)
        
        self.rec_tab = tab_rec
        self.log_tab = tab_log
        self.tabs.addTab(tab_rec, "Recommendations")
        self.tabs.addTab(tab_log, "Log")
        
        # Add AI tab
        ai_tab = self.create_ai_tab()
        self.tabs.addTab(ai_tab, "🧠 AI Assistant")
        
        l_layout.addWidget(self.tabs)

        btns_layout = QGridLayout()
        btns_layout.setSpacing(8)

        self.btn_analyze = QPushButton("🚀 Analyze")
        self.btn_profile = QPushButton("📡 Profile")
        self.btn_baseline = QPushButton("📌 Baseline")
        self.btn_compare = QPushButton("📈 Compare")
        self.btn_optimize = QPushButton("🛠 Optimize")
        self.btn_rollback = QPushButton("↩ Rollback")
        
        buttons = [self.btn_analyze, self.btn_profile, self.btn_baseline, 
                  self.btn_compare, self.btn_optimize, self.btn_rollback]
        
        for btn in buttons:
            btn.setStyleSheet(self.get_button_style())
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.setup_button_animation(btn)

        btns_layout.addWidget(self.btn_analyze, 0, 0)
        btns_layout.addWidget(self.btn_profile, 0, 1)
        btns_layout.addWidget(self.btn_baseline, 1, 0)
        btns_layout.addWidget(self.btn_compare, 1, 1)
        btns_layout.addWidget(self.btn_optimize, 2, 0)
        btns_layout.addWidget(self.btn_rollback, 2, 1)
        
        self.btn_analyze.clicked.connect(self.action_analyze)
        self.btn_profile.clicked.connect(self.action_profile)
        self.btn_baseline.clicked.connect(self.action_save_baseline)
        self.btn_compare.clicked.connect(self.action_compare)
        self.btn_optimize.clicked.connect(self.action_optimize)
        self.btn_rollback.clicked.connect(self.action_rollback)
        
        l_layout.addLayout(btns_layout)

        lower = QHBoxLayout()
        self.btn_screenshot = QPushButton("💾 Screenshot")
        self.btn_export = QPushButton("🌐 Export HTML")
        
        for btn in (self.btn_screenshot, self.btn_export):
            btn.setStyleSheet(self.get_button_style(light=True))
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.setup_button_animation(btn)
            
        self.btn_screenshot.clicked.connect(self.action_screenshot)
        self.btn_export.clicked.connect(self.action_export_html)
        
        lower.addWidget(self.btn_screenshot)
        lower.addWidget(self.btn_export)
        l_layout.addLayout(lower)

        if not IS_ROOT:
            root_notice = QLabel("🔒 Root access required for system optimization")
            root_notice.setStyleSheet("color: #FF6B6B; font-size: 10px; padding: 5px;")
            l_layout.addWidget(root_notice)
            self.btn_optimize.setEnabled(False)
            self.btn_rollback.setEnabled(False)
        else:
            root_notice = QLabel("🔓 Root access available")
            root_notice.setStyleSheet("color: #1DB954; font-size: 10px; padding: 5px;")
            l_layout.addWidget(root_notice)

        l_layout.addStretch()
        root.addWidget(left)

        right = QFrame()
        right.setStyleSheet("QFrame { background: transparent; }")
        r_layout = QVBoxLayout(right)
        r_layout.setSpacing(15)

        bg = QFrame(right)
        bg.setStyleSheet("""
            QFrame { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 rgba(18, 18, 24, 220), 
                    stop:1 rgba(28, 28, 36, 220)); 
                border-radius: 16px; 
            }
        """)
        bg.setGeometry(0, 0, 950, 750)
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(12)
        bg.setGraphicsEffect(blur)

        card = QFrame()
        card.setStyleSheet("""
            QFrame { 
                background: rgba(25, 25, 35, 220); 
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(25)
        card_shadow.setColor(QColor(0, 0, 0, 140))
        card_shadow.setOffset(8, 8)
        card.setGraphicsEffect(card_shadow)
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(15, 15, 15, 15)
        card_layout.setSpacing(12)

        header_row = QHBoxLayout()
        self.sel_label = QLabel("Target: system overview")
        self.sel_label.setStyleSheet("font-weight: 700; color: #E6EEF3; font-size: 14px;")
        header_row.addWidget(self.sel_label)
        header_row.addStretch()
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #B3B3B3; font-size: 12px;")
        header_row.addWidget(self.fps_label)
        card_layout.addLayout(header_row)

        grid = QGridLayout()
        grid.setSpacing(15)
        
        self.g_cpu = SmoothGauge("CPU Usage", 100.0, "%", "#1DB954")
        self.g_mem = SmoothGauge("Memory", 100.0, "%", "#1DB954")
        self.g_disk = SmoothGauge("Disk I/O", 200.0, "MB/s", "#1DB954")
        self.g_net = SmoothGauge("Network", 100.0, "MB/s", "#1DB954")
        self.g_sys_cpu = SmoothGauge("System CPU", 100.0, "%", "#1DB954")
        self.g_sys_mem = SmoothGauge("System RAM", 100.0, "%", "#1DB954")
        
        grid.addWidget(self.g_cpu, 0, 0)
        grid.addWidget(self.g_mem, 0, 1)
        grid.addWidget(self.g_disk, 0, 2)
        grid.addWidget(self.g_net, 1, 0)
        grid.addWidget(self.g_sys_cpu, 1, 1)
        grid.addWidget(self.g_sys_mem, 1, 2)
        
        card_layout.addLayout(grid)

        status_row = QHBoxLayout()
        self.status_label = QLabel("🟢 System monitoring active")
        self.status_label.setStyleSheet("color: #1DB954; font-size: 11px;")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: #888; font-size: 10px;")
        status_row.addWidget(self.time_label)
        
        card_layout.addLayout(status_row)
        r_layout.addWidget(card)
        root.addWidget(right)

        self.card = card
        self.bg = bg
        self.rec_box = tab_rec
        self.log_box = tab_log

        QTimer.singleShot(100, self.refresh_proc_list)
        self.update_time()

    def create_ai_tab(self):
        """Создание вкладки ИИ"""
        ai_tab = QWidget()
        ai_layout = QHBoxLayout(ai_tab)
        
        # Левая часть - чат с ИИ
        self.ai_chat = AIChatWidget(ai_agent=self.ai_agent)
        ai_layout.addWidget(self.ai_chat, 2)
        
        # Правая часть - дашборд анализа
        self.ai_dashboard = AIAnalysisDashboard()
        ai_layout.addWidget(self.ai_dashboard, 1)
        
        return ai_tab

    def get_button_style(self, light=False):
        if light:
            return """
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 rgba(50, 50, 60, 200), 
                    stop:1 rgba(70, 70, 80, 200)); 
                color: #E6EEF3; 
                border-radius: 8px; 
                padding: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                font-weight: 500;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(60, 60, 70, 220),
                    stop:1 rgba(80, 80, 90, 220));
            }
            QPushButton:pressed { 
                background: rgba(29, 185, 84, 0.3);
            }
            """
        return """
        QPushButton { 
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(29, 185, 84, 220),
                stop:1 rgba(20, 150, 70, 220));
            color: white; 
            border-radius: 8px; 
            padding: 10px;
            border: none;
            font-weight: 600;
            font-size: 11px;
        }
        QPushButton:hover { 
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(39, 195, 94, 240),
                stop:1 rgba(30, 160, 80, 240));
        }
        QPushButton:pressed { 
            background: rgba(29, 185, 84, 200);
        }
        """

    def setup_button_animation(self, button):
        effect = QGraphicsOpacityEffect(button)
        button.setGraphicsEffect(effect)
        button._animation = QPropertyAnimation(effect, b"opacity")
        button._animation.setDuration(200)
        button._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        def enter_event(e):
            button._animation.setStartValue(1.0)
            button._animation.setEndValue(0.9)
            button._animation.start()
            QPushButton.enterEvent(button, e)

        def leave_event(e):
            button._animation.setStartValue(0.9)
            button._animation.setEndValue(1.0)
            button._animation.start()
            QPushButton.leaveEvent(button, e)

        button.enterEvent = enter_event
        button.leaveEvent = leave_event

    def update_time(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(f"⌚ {current_time}")
        QTimer.singleShot(1000, self.update_time)

    def action_optimize(self):
        if not IS_ROOT:
            QMessageBox.warning(self, "Optimize", "Root access required for system optimization")
            return
            
        try:
            optimizations = {
                "vm.swappiness": "10",
                "vm.dirty_ratio": "15",
                "vm.dirty_background_ratio": "5",
                "vm.vfs_cache_pressure": "50",
                "net.core.netdev_max_backlog": "30000",
                "net.core.somaxconn": "1024"
            }
            
            current_values = {}
            for param in optimizations.keys():
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", param],
                        capture_output=True, text=True, check=True
                    )
                    current_values[param] = result.stdout.strip()
                except subprocess.CalledProcessError:
                    current_values[param] = "Not available"
                    self.log(f"Could not read current value for {param}")
            
            safe_write_json(self.rollback_path, {
                'timestamp': datetime.now().isoformat(),
                'values': current_values
            })
            
            applied = []
            for param, value in optimizations.items():
                try:
                    subprocess.run(
                        ["sysctl", "-w", f"{param}={value}"],
                        check=True, capture_output=True
                    )
                    applied.append(param)
                    self.log(f"Optimized {param} = {value}")
                except subprocess.CalledProcessError as e:
                    self.log(f"Failed to set {param}: {e}")
            
            if applied:
                self.rec(f"✅ Applied {len(applied)} optimizations")
                QMessageBox.information(self, "Optimization", 
                                      f"Successfully applied {len(applied)} system optimizations")
            else:
                self.rec("❌ No optimizations could be applied")
                QMessageBox.warning(self, "Optimization", "Failed to apply any optimizations")
                
        except Exception as e:
            error_msg = f"Optimization error: {str(e)}"
            self.log(error_msg)
            QMessageBox.critical(self, "Optimization Error", error_msg)

    def setup_timers(self):
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)

        self.gauge_timer = QTimer()
        self.gauge_timer.timeout.connect(self.update_gauges)
        self.gauge_timer.start(16)  # ~60 FPS

    def update_fps(self):
        if self.last:
            fps = 1.0 / max(0.001, time.time() - self.last)
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.last = time.time()

    def update_gauges(self):
        for gauge in (self.g_cpu, self.g_mem, self.g_disk, self.g_net, self.g_sys_cpu, self.g_sys_mem):
            gauge.tick()

    def start_sampler(self):
        if self.sampler and self.sampler.isRunning():
            self.sampler.stop()
        self.sampler = Sampler(self.target_pid)
        self.sampler.sample.connect(self.on_sample)
        self.sampler.log.connect(self.log)
        self.sampler.start()

    def on_sample(self, data):
        self.history.append(data)
        self.smoothers['proc_cpu'].update(data['proc_cpu'])
        self.smoothers['proc_mem'].update(data['proc_mem'])
        self.smoothers['disk_MBps'].update(data['disk_Bps'] / 1e6)
        self.smoothers['net_MBps'].update(data['net_Bps'] / 1e6)
        self.smoothers['sys_cpu'].update(data['sys_cpu'])
        self.smoothers['sys_mem'].update(data['sys_mem'])

        self.g_cpu.set_target(self.smoothers['proc_cpu'].value)
        self.g_mem.set_target(self.smoothers['proc_mem'].value)
        self.g_disk.set_target(self.smoothers['disk_MBps'].value)
        self.g_net.set_target(self.smoothers['net_MBps'].value)
        self.g_sys_cpu.set_target(self.smoothers['sys_cpu'].value)
        self.g_sys_mem.set_target(self.smoothers['sys_mem'].value)

        # Обновление ИИ-анализа
        system_data = {
            'sys_cpu': data['sys_cpu'],
            'sys_mem': data['sys_mem'], 
            'disk_Bps': data['disk_Bps'],
            'net_Bps': data['net_Bps']
        }
        
        process_data = {
            'proc_cpu': data['proc_cpu'],
            'proc_mem': data['proc_mem'],
            'proc_io': data['proc_io']
        }
        
        self.ai_dashboard.update_dashboard(system_data, process_data)
        self.ai_chat.update_ai_analysis(system_data, process_data)

    def refresh_proc_list(self):
        self.proc_list.clear()
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            processes.sort(key=lambda x: x['memory_percent'] or 0, reverse=True)
            
            for proc in processes[:25]:
                if proc['name']:
                    memory = proc['memory_percent'] or 0
                    cpu = proc['cpu_percent'] or 0
                    item_text = f"{proc['pid']:>6} | {proc['name'][:20]:<20} | CPU:{cpu:5.1f}% RAM:{memory:5.1f}%"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, proc['pid'])
                    self.proc_list.addItem(item)
        except Exception as e:
            self.log(f"Error refreshing process list: {e}")

    def on_pid_search(self):
        text = self.pid_input.text().strip()
        try:
            pid = int(text)
            if psutil.pid_exists(pid):
                p = psutil.Process(pid)
                self.target_pid = pid
                self.sel_label.setText(f"Target: {p.name()} (PID: {pid})")
                self.start_sampler()
                self.log(f"Selected process: {p.name()} (PID: {pid})")
            else:
                QMessageBox.warning(self, "Error", "Process not found")
        except ValueError:
            for proc in psutil.process_iter(['pid', 'name']):
                if text.lower() in proc.info['name'].lower():
                    self.target_pid = proc.info['pid']
                    self.sel_label.setText(f"Target: {proc.info['name']} (PID: {proc.info['pid']})")
                    self.start_sampler()
                    self.log(f"Selected process: {proc.info['name']} (PID: {proc.info['pid']})")
                    return
            QMessageBox.warning(self, "Error", "Process not found")

    def on_proc_selected(self, item):
        pid = item.data(Qt.ItemDataRole.UserRole)
        try:
            p = psutil.Process(pid)
            self.target_pid = pid
            self.sel_label.setText(f"Target: {p.name()} (PID: {pid})")
            self.start_sampler()
            self.log(f"Selected process: {p.name()} (PID: {pid})")
        except psutil.NoSuchProcess:
            self.log("Error: Process not found")

    def log(self, msg):
        self.log_box.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        scrollbar = self.log_box.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def rec(self, msg):
        self.rec_box.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        scrollbar = self.rec_box.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def action_analyze(self):
        if not self.target_pid:
            QMessageBox.warning(self, "Analyze", "Select a process first")
            return
        self.analyzer = QuickAnalyzer(self.target_pid, duration=15)
        self.analyzer.finished.connect(self.on_analyze_finished)
        self.analyzer.log.connect(self.log)
        self.analyzer.start()
        self.log("Started analysis...")

    def on_analyze_finished(self, result):
        if 'error' in result:
            self.rec(f"Analysis failed: {result['error']}")
            return
        self.rec("\n".join(result['recs']))
        self.log(f"Analysis complete: CPU {result['avg_cpu']:.1f}%, Mem {result['avg_mem']:.1f}%")

    def action_profile(self):
        self.log("Profiling not implemented yet")

    def action_save_baseline(self):
        if not self.history:
            self.log("No data to save as baseline")
            return
        safe_write_json(self.baseline_path, list(self.history))
        self.log("Baseline saved")

    def action_compare(self):
        baseline = safe_read_json(self.baseline_path)
        if not baseline or not self.history:
            self.log("No baseline or current data to compare")
            return
        self.log("Comparison not fully implemented yet")

    def action_screenshot(self):
        try:
            screenshot = self.grab()
            filename, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png)")
            if filename:
                screenshot.save(filename, "PNG")
                self.log(f"Screenshot saved to {filename}")
        except Exception as e:
            self.log(f"Screenshot error: {e}")

    def action_export_html(self):
        if not PLOTLY_AVAILABLE:
            self.log("Plotly not installed, cannot export HTML")
            return
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Export HTML", "", "HTML Files (*.html)")
            if filename:
                self.log(f"Export HTML not fully implemented, saved placeholder to {filename}")
        except Exception as e:
            self.log(f"Export HTML error: {e}")

    def action_rollback(self):
        if not IS_ROOT:
            QMessageBox.warning(self, "Rollback", "Root access required for rollback")
            return
        rollback = safe_read_json(self.rollback_path)
        if not rollback:
            self.log("No rollback data available")
            return
        try:
            for param, value in rollback['values'].items():
                if value != "Not available":
                    subprocess.run(["sysctl", "-w", f"{param}={value}"], check=True)
                    self.log(f"Restored {param} = {value}")
            self.rec("Rollback completed")
        except Exception as e:
            self.log(f"Rollback error: {e}")

    def closeEvent(self, event):
        if self.sampler and self.sampler.isRunning():
            self.sampler.stop()
        event.accept()

# ----------------- Entrypoint -----------------
def main():
    ensure_resources()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    app.setStyle('Fusion')
    
    win = PerfTuneUltimate()
    win.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}\n{traceback.format_exc()}")