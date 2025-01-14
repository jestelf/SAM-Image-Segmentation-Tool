### **SAM Image Segmentation Tool**

---

### Краткое описание репозитория:

**SAM Image Segmentation Tool** — это инструмент для интерактивной сегментации изображений на основе **Segment Anything Model (SAM)**. Приложение позволяет пользователям быстро выделять объекты на изображениях, сохраняя результаты в виде прозрачных PNG-файлов.

---

# SAM Image Segmentation Tool

**SAM Image Segmentation Tool** — интерактивное приложение для выделения объектов на изображениях с использованием модели **Segment Anything Model (SAM)**. Оно позволяет легко выделять объекты, используя точечные подсказки (prompting) и операции drag-and-drop.

## Возможности
- Быстрая сегментация объектов по точкам на изображении.
- Поддержка drag-and-drop для серийного выделения объектов.
- Сохранение результатов в формате PNG с прозрачностью.
- Интерфейс с тёмной темой.

## Установка и запуск

### Требования:
- Python 3.9+
- Установленные зависимости из `requirements.txt`
- Модель SAM (`sam_vit_h.pth`, `sam_vit_l.pth` или `sam_vit_b.pth`)

### Шаги установки:

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/jestelf/SAM-Image-Segmentation-Tool.git
   cd SAM-Image-Segmentation-Tool
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Скачайте модель SAM:
   - Вы можете выбрать модель из [официального репозитория SAM](https://github.com/facebookresearch/segment-anything).
   - Поместите скачанный файл в корень проекта. Например, `sam_vit_h.pth`.

4. Запустите приложение:
   ```bash
   python main.py
   ```

## Сборка в `.exe`

Если вы хотите собрать приложение в единый `.exe` файл, выполните следующие шаги:

1. Установите `auto-py-to-exe`:
   ```bash
   pip install auto-py-to-exe
   ```

2. Запустите GUI для сборки:
   ```bash
   auto-py-to-exe
   ```

3. Укажите:
   - Файл `main.py` как главный скрипт.
   - Дополнительно включите модель `sam_vit_h.pth` через `--add-data`.

4. Сконвертируйте файл и проверьте `.exe` в папке `dist`.

## Использование

### Интерфейс
- **Левое окно**: исходное изображение с выделением объектов.
- **Правое окно**: итоговый результат с прозрачным фоном.
- Кнопки:
  - `Load Image` — загрузка изображения.
  - `Save PNG` — сохранение результата.

### Горячие клавиши
- **Ctrl+ЛКМ**: добавление объектов (drag для серийного выделения).
- **Ctrl+ПКМ**: удаление объектов.
- **Ctrl+Z**: отмена последнего действия.

### Пример структуры папок
```bash
SAM-Image-Segmentation-Tool/
├── main.py               # Главный скрипт
├── sam_vit_h.pth         # Модель Segment Anything (HUGE)
├── requirements.txt      # Зависимости проекта
├── README.md             # Инструкция
```

## Установка модели SAM

1. Перейдите на страницу [Segment Anything Model](https://github.com/facebookresearch/segment-anything).
2. Выберите одну из моделей:
   - `sam_vit_h.pth` (большая, точная)
   - `sam_vit_l.pth` (средняя)
   - `sam_vit_b.pth` (компактная)
3. Скачайте модель и поместите в корень папки проекта.

## Зависимости

Все необходимые библиотеки указаны в `requirements.txt`. Для установки выполните:

```bash
pip install -r requirements.txt
```

Список ключевых библиотек:
- `PyQt5` — для GUI.
- `cv2` — для обработки изображений.
- `numpy` — для работы с массивами.
- `segment-anything` — библиотека Segment Anything Model.

## Лицензия

Все права защищены. Лицензионная информация доступна в файле [LICENSE](./LICENSE).
