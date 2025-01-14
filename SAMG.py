import sys
import cv2
import numpy as np
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

# Импорт из segment-anything
from segment_anything import SamPredictor, sam_model_registry


def create_checkerboard(width, height, tile_size=20):
    """
    Создаёт шахматное поле (height, width, 3).
    """
    board = np.zeros((height, width, 3), dtype=np.uint8)

    # Оттенки серого (или любые другие на ваше усмотрение)
    color1 = (180, 180, 180)
    color2 = (220, 220, 220)

    for y in range(height):
        for x in range(width):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                board[y, x] = color1
            else:
                board[y, x] = color2

    return board


def smooth_mask(mask, ksize=5):
    """
    Сглаживание маски морфологической операцией (close).
    Можно дополнительно добавить размытие по границе.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    # Превращаем обратно в bool
    final_mask = (closed > 127)
    return final_mask


class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Editor - SAM")
        self.setGeometry(100, 100, 1200, 700)

        # Стили для тёмно-серого интерфейса
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D2D;
            }
            QLabel {
                background-color: #3C3C3C;
                border: 1px solid #666666;
            }
            QPushButton {
                background-color: #4D4D4D;
                color: #FFFFFF;
                border: 1px solid #888888;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #5D5D5D;
            }
        """)

        # ------------ Две области просмотра -------------
        self.label_left = QLabel(self)
        self.label_left.setAlignment(Qt.AlignCenter)
        self.label_left.setGeometry(30, 50, 550, 500)

        self.label_right = QLabel(self)
        self.label_right.setAlignment(Qt.AlignCenter)
        self.label_right.setGeometry(620, 50, 550, 500)

        # ------------- Кнопки ----------------
        self.load_button = QPushButton("Load Image", self)
        self.load_button.setGeometry(30, 600, 100, 40)
        self.load_button.clicked.connect(self.load_image)

        self.save_button = QPushButton("Save PNG", self)
        self.save_button.setGeometry(140, 600, 100, 40)
        self.save_button.clicked.connect(self.save_image)

        # ------------ Данные ---------------
        self.image = None             # Исходная картинка (RGB)
        self.overlay_image = None     # Для левого окна
        self.result_image = None      # Для правого окна (вырезка)

        # Инициализация SAM
        self.model = None
        self.predictor = None

        # Список принятых масок (каждый элемент — np.array(bool))
        self.masks = []
        # Текущая «hover»-маска
        self.temp_hover_mask = None

        # Стек для undo
        self.history = []

        # Для drag (если пользователь зажал Ctrl + ЛКМ и двигает)
        self.dragging = False
        self.last_drag_point = None   # (x_img, y_img)
        self.drag_distance_threshold = 30.0  # Каждые 30 пикселей делаем новую сегментацию

        # Чтобы не пересегментировать на каждый пиксель при hover
        self.last_hover_point = None

    # ------------------ Загрузка/сохранение --------------------
    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Images (*.png *.jpg *.jpeg)",
            options=options
        )
        if file_name:
            try:
                temp_img = cv2.imread(file_name)
                if temp_img is None:
                    raise ValueError("OpenCV вернул None (файл не найден или неверный формат).")

                temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                self.image = temp_img

                # Инициализируем SAM (vit_h или другую). Для скорости можно взять vit_l или vit_b.
                # Но vit_h даёт наиболее точные результаты.
                if self.model is None:
                    MODEL_TYPE = "vit_h"
                    MODEL_PATH = "sam_vit_h.pth"  # Укажите свой путь к модели
                    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
                    self.predictor = SamPredictor(sam)
                    self.model = sam
                # Передаём картинку в SAM (создаётся embedding)
                # Это самая «дорогая» операция, делается 1 раз на картинку.
                self.predictor.set_image(self.image)

                # Очищаем маски, историю и т.д.
                self.masks.clear()
                self.history.clear()
                self.temp_hover_mask = None
                self.last_hover_point = None
                self.dragging = False
                self.last_drag_point = None

                self.update_visuals()

            except Exception as e:
                QMessageBox.warning(self, "Ошибка загрузки", f"Не удалось загрузить изображение:\n{e}")

    def save_image(self):
        """Сохраняем результат (правое окно) в PNG."""
        if self.result_image is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;All Files (*)"
        )
        if file_name:
            cv2.imwrite(file_name, cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR))

    # ------------------ Сегментация --------------------
    def segment_object(self, x, y):
        """
        Сегментация SAM по одной точке (x, y) в координатах self.image.
        Возвращаем сглаженную маску (bool).
        """
        if self.predictor is None:
            return np.zeros(self.image.shape[:2], dtype=bool)
        # Применяем point prompt
        input_points = np.array([[x, y]])
        input_labels = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        raw_mask = masks[0]
        sm_mask = smooth_mask(raw_mask, ksize=5)
        return sm_mask

    def apply_masks(self):
        """Объединяем все утверждённые маски логическим ИЛИ."""
        if not self.masks:
            return np.zeros(self.image.shape[:2], dtype=bool)
        combined = np.zeros_like(self.masks[0], dtype=bool)
        for m in self.masks:
            combined |= m
        return combined

    # ------------------ Undo / History --------------------
    def push_history(self):
        """Сохраняем копию self.masks в стек, чтобы можно было откатить (undo)."""
        copy_of_masks = [m.copy() for m in self.masks]
        self.history.append(copy_of_masks)

    def pop_history(self):
        """Восстанавливаем последнее сохранённое состояние масок."""
        if self.history:
            self.masks = self.history.pop()
        else:
            self.masks.clear()

    # ------------------ Отрисовка --------------------
    def update_visuals(self):
        """
        Перерисовываем оба окна:
        - Левое: исходное + принятые маски + hover-маска
        - Правое: итог (объединённая маска) на шахматном фоне
        """
        if self.image is None:
            self.label_left.clear()
            self.label_right.clear()
            return

        # Собираем объединённую маску
        full_mask = self.apply_masks()

        # Добавляем hover (только на левый экран)
        preview_mask = full_mask.copy()
        if self.temp_hover_mask is not None:
            preview_mask |= self.temp_hover_mask

        # Рисуем overlay
        overlay_img = self.image.copy()
        overlay = np.zeros_like(overlay_img, dtype=np.uint8)
        # Тёмно-красный цвет маски
        overlay[preview_mask] = [200, 50, 50]
        alpha = 0.5
        overlay_img = cv2.addWeighted(overlay_img, 1 - alpha, overlay, alpha, 0)

        # Правое окно: вырезаем объекты на шахматном фоне
        h, w = self.image.shape[:2]
        checker = create_checkerboard(w, h, 20)
        result = checker.copy()
        result[full_mask] = self.image[full_mask]

        self.overlay_image = overlay_img
        self.result_image = result

        # Выводим в QLabel
        self.display_image(self.label_left, overlay_img)
        self.display_image(self.label_right, result)

    def display_image(self, label, img):
        """Показ numpy-изображения (RGB) в QLabel."""
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # ------------------ Мышь --------------------
    def mousePressEvent(self, event):
        """Обработка нажатия кнопок мыши."""
        if self.image is None:
            return

        # Проверяем, попали ли мы в левый QLabel
        if not self.is_in_label(self.label_left, event.pos()):
            return

        x_img, y_img = self.to_image_coords(self.label_left, event.pos())

        # ЛКМ
        if event.button() == Qt.LeftButton:
            # Проверим, зажат ли Ctrl
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                # Начинаем «drag»-режим для серийного выделения
                self.dragging = True
                self.last_drag_point = (x_img, y_img)
                # push_history один раз, чтобы потом можно было сделать Undo
                self.push_history()

                # Добавляем первую маску прямо сейчас
                new_mask = self.segment_object(x_img, y_img)
                if new_mask.any():
                    self.masks.append(new_mask)
                self.temp_hover_mask = None
                self.last_hover_point = None
                self.update_visuals()
            else:
                # Без Ctrl: сброс всех, делаем одну маску
                self.push_history()
                self.masks.clear()
                new_mask = self.segment_object(x_img, y_img)
                if new_mask.any():
                    self.masks.append(new_mask)
                self.temp_hover_mask = None
                self.last_hover_point = None
                self.update_visuals()

        # ПКМ
        elif event.button() == Qt.RightButton:
            # Проверим, зажат ли Ctrl
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                # Удалить конкретную маску, на которую кликнули.
                self.push_history()
                self.remove_mask_at_point(x_img, y_img)
                self.temp_hover_mask = None
                self.last_hover_point = None
                self.update_visuals()
            else:
                # Сброс всех масок
                self.push_history()
                self.masks.clear()
                self.temp_hover_mask = None
                self.last_hover_point = None
                self.update_visuals()

    def mouseMoveEvent(self, event):
        """Hover-подсветка или drag при Ctrl+ЛКМ."""
        if self.image is None:
            return

        # Если drag (CTRL+ЛКМ зажаты), значит добавляем маски «по пути».
        if self.dragging:
            if not (event.buttons() & Qt.LeftButton):
                # Если пользователь отпустил ЛКМ
                self.dragging = False
                self.last_drag_point = None
                return

            if not self.is_in_label(self.label_left, event.pos()):
                # Если вышли за пределы левого QLabel
                return

            x_img, y_img = self.to_image_coords(self.label_left, event.pos())
            # Проверим дистанцию от последней точки
            if self.last_drag_point is not None:
                dx = x_img - self.last_drag_point[0]
                dy = y_img - self.last_drag_point[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist >= self.drag_distance_threshold:
                    # Добавляем новую маску
                    new_mask = self.segment_object(x_img, y_img)
                    if new_mask.any():
                        self.masks.append(new_mask)
                    self.last_drag_point = (x_img, y_img)
                    self.update_visuals()

        else:
            # Если просто двигаем мышь (hover), показываем temp_hover_mask
            if not self.is_in_label(self.label_left, event.pos()):
                # Убрали курсор из левой зоны - убираем hover
                if self.temp_hover_mask is not None:
                    self.temp_hover_mask = None
                    self.last_hover_point = None
                    self.update_visuals()
                return

            x_img, y_img = self.to_image_coords(self.label_left, event.pos())
            if (self.last_hover_point == (x_img, y_img)):
                return

            # Вычисляем hover-маску
            hover_mask = self.segment_object(x_img, y_img)
            self.temp_hover_mask = hover_mask
            self.last_hover_point = (x_img, y_img)
            self.update_visuals()

    def mouseReleaseEvent(self, event):
        """Если отпустили ЛКМ после drag-а, завершаем drag."""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.last_drag_point = None

    # ------------------ Клавиатура --------------------
    def keyPressEvent(self, event):
        # Ctrl + Z = Undo
        if (event.key() == Qt.Key_Z) and (event.modifiers() & Qt.ControlModifier):
            self.pop_history()
            self.temp_hover_mask = None
            self.last_hover_point = None
            self.dragging = False
            self.last_drag_point = None
            self.update_visuals()
        else:
            super().keyPressEvent(event)

    # ------------------ Утилиты --------------------
    def remove_mask_at_point(self, x, y):
        """
        Удаляем «верхнюю» маску (последнюю в списке), которая включает точку (x, y).
        Перебираем маски с конца, удаляем первую найденную.
        """
        for i in reversed(range(len(self.masks))):
            if self.masks[i][y, x]:
                self.masks.pop(i)
                break

    def is_in_label(self, label, global_pos):
        """Проверяем, попадает ли точка global_pos (в системе координат окна) в label."""
        rel = label.mapFromGlobal(self.mapToGlobal(global_pos))
        return (0 <= rel.x() < label.width() and 0 <= rel.y() < label.height())

    def to_image_coords(self, label, global_pos):
        """
        Преобразуем координаты окна -> координаты внутри label -> (x,y) на картинке self.image.
        """
        rel = label.mapFromGlobal(self.mapToGlobal(global_pos))
        img_h, img_w = self.image.shape[:2]
        x_ratio = img_w / label.width()
        y_ratio = img_h / label.height()

        x_img = int(rel.x() * x_ratio)
        y_img = int(rel.y() * y_ratio)

        x_img = max(0, min(x_img, img_w - 1))
        y_img = max(0, min(y_img, img_h - 1))
        return x_img, y_img


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ImageEditor()
    editor.show()
    sys.exit(app.exec_())
