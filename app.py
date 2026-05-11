import os
import threading
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import ImageStat
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import torch_directml
    HAS_DML = True
except ImportError:
    HAS_DML = False

TARGET_MAPPING = {
    "1": 1720.0, "2": 1900.0, "3": 1920.0,
    "4": 2150.0, "5": 2200.0, "6": 2400.0, "7": 2650.0
}
TARGET_MIN, TARGET_MAX = 1720.0, 2650.0
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

MODEL_PATH = "best_peat_model.pth"
TEST_DIR = "dataset_test"

BG_COLOR = "#F4F7FB"
PANEL_BG = "#FFFFFF"
PRIMARY = "#4A90E2"
SUCCESS = "#2ECC71"
WARNING = "#F39C12"
INFO = "#8E44AD"
TEXT_MAIN = "#2C3E50"
TEXT_MUTED = "#7F8C8D"

def is_good_image(img_path: str) -> bool:
    try:
        with Image.open(img_path) as img:
            img_gray = img.convert('L')
            stat = ImageStat.Stat(img_gray)
            brightness = stat.mean[0]
            contrast = stat.stddev[0]
            if brightness < 30: return False
            if brightness > 230: return False
            if contrast < 15: return False
        return True
    except Exception:
        return False

class PeatDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, filter_bad: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []
        self.targets: List[float] = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Директория {root_dir} не найдена!")

        for folder_name in os.listdir(root_dir):
            if folder_name in TARGET_MAPPING:
                folder_path = os.path.join(root_dir, folder_name)
                if os.path.isdir(folder_path):
                    target_value = TARGET_MAPPING[folder_name]
                    target_scaled = (target_value - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)

                    for file_name in os.listdir(folder_path):
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(folder_path, file_name)
                            if filter_bad and not is_good_image(full_path):
                                continue
                            self.image_paths.append(full_path)
                            self.targets.append(target_scaled)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([self.targets[idx]], dtype=torch.float32)

def create_dataloaders(root_dir: str, batch_size: int = 8, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    full_dataset = PeatDataset(root_dir=root_dir, transform=None, filter_bad=True)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(torch.floor(torch.tensor(val_split * dataset_size)))

    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = PeatDataset(root_dir=root_dir, transform=train_transform, filter_bad=False)
    val_dataset = PeatDataset(root_dir=root_dir, transform=val_transform, filter_bad=False)

    train_dataset.image_paths = [full_dataset.image_paths[i] for i in train_indices]
    train_dataset.targets = [full_dataset.targets[i] for i in train_indices]
    val_dataset.image_paths = [full_dataset.image_paths[i] for i in val_indices]
    val_dataset.targets = [full_dataset.targets[i] for i in val_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ТОРФ AI - Интеллектуальный анализ торфа")
        self.root.geometry("1000x650")
        self.root.configure(bg=BG_COLOR)

        self.model = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
        elif HAS_DML and torch_directml.is_available():
            dml_device_id = torch_directml.default_device()
            self.device = torch_directml.device(dml_device_id)
            self.device_name = f"AMD/Intel DirectML ({torch_directml.device_name(dml_device_id)})"
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU (Процессор)"

        self.setup_styles()
        self.build_ui()
        self.load_model()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", thickness=20, background=SUCCESS, troughcolor="#E0E0E0")

    def build_ui(self):
        sidebar = tk.Frame(self.root, bg="#2C3E50", width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        lbl_logo = tk.Label(sidebar, text="ТОРФ AI", font=("Segoe UI Black", 24), bg="#2C3E50", fg="#FFFFFF")
        lbl_logo.pack(pady=(30, 5))
        lbl_sub = tk.Label(sidebar, text="Анализ влажности", font=("Segoe UI", 10), bg="#2C3E50", fg=TEXT_MUTED)
        lbl_sub.pack(pady=(0, 40))

        self.create_nav_button(sidebar, "1. Обучить модель", PRIMARY, self.show_train_view)
        self.create_nav_button(sidebar, "2. Определить влажность", SUCCESS, self.show_predict_view)
        self.create_nav_button(sidebar, "3. Протестировать выборку", WARNING, self.show_eval_view)

        self.model_status_lbl = tk.Label(sidebar, text="Загрузка модели...", font=("Segoe UI", 9), bg="#2C3E50",
                                         fg=TEXT_MUTED)
        self.model_status_lbl.pack(side=tk.BOTTOM, pady=10)

        lbl_dev = tk.Label(sidebar, text=f"Устройство:\n{self.device_name}", font=("Segoe UI", 8), bg="#2C3E50",
                           fg=TEXT_MUTED)
        lbl_dev.pack(side=tk.BOTTOM, pady=(0, 10))

        self.main_content = tk.Frame(self.root, bg=BG_COLOR)
        self.main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.header_lbl = tk.Label(self.main_content, text="Добро пожаловать", font=("Segoe UI", 20, "bold"),
                                   bg=BG_COLOR, fg=TEXT_MAIN)
        self.header_lbl.pack(anchor="w", padx=30, pady=(30, 20))

        self.view_container = tk.Frame(self.main_content, bg=BG_COLOR)
        self.view_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 30))

        self.current_widgets = []
        self.show_predict_view()

    def create_nav_button(self, parent, text, color, command):
        btn = tk.Button(parent, text=text, font=("Segoe UI", 12, "bold"), bg=color, fg="#FFFFFF",
                        activebackground="#FFFFFF", activeforeground=color,
                        relief=tk.FLAT, borderwidth=0, pady=12, cursor="hand2", command=command)
        btn.pack(fill=tk.X, padx=15, pady=10)
        return btn

    def clear_view(self):
        for w in self.current_widgets: w.destroy()
        self.current_widgets.clear()

    def create_card(self, parent, title, value, color):
        card = tk.Frame(parent, bg=PANEL_BG, highlightbackground="#E2E8F0", highlightthickness=1, bd=0)
        lbl_title = tk.Label(card, text=title, font=("Segoe UI", 10, "bold"), bg=PANEL_BG, fg=TEXT_MUTED)
        lbl_title.pack(anchor="w", padx=15, pady=(15, 0))

        lbl_val = tk.Label(card, text=value, font=("Segoe UI", 24, "bold"), bg=PANEL_BG, fg=color)
        lbl_val.pack(anchor="w", padx=15, pady=(0, 15))
        return card, lbl_val

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = models.resnet34(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, 1)
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
                self.model.to(self.device)
                self.model.eval()
                self.model_status_lbl.config(text="✓ Нейросеть готова к работе", fg=SUCCESS)
            except Exception as e:
                self.model_status_lbl.config(text="Ошибка загрузки модели", fg="#E74C3C")
        else:
            self.model_status_lbl.config(text="⚠️ Модель не найдена", fg=WARNING)

    def show_train_view(self):
        self.clear_view()
        self.header_lbl.config(text="Обучение модели")

        card = tk.Frame(self.view_container, bg=PANEL_BG, highlightbackground="#E2E8F0", highlightthickness=1)
        card.pack(fill=tk.X, pady=10)
        self.current_widgets.append(card)

        info = tk.Label(card,
                        text="Запустить процесс переобучения нейросети на ваших данных",
                        font=("Segoe UI", 12), bg=PANEL_BG, fg=TEXT_MAIN, justify=tk.LEFT)
        info.pack(padx=20, pady=20)

        self.train_progress = ttk.Progressbar(card, orient="horizontal", mode="determinate", style="TProgressbar")
        self.lbl_train_status = tk.Label(card, text="Готов к запуску", font=("Segoe UI", 12, "bold"), bg=PANEL_BG,
                                         fg=PRIMARY)
        self.lbl_train_status.pack(pady=(0, 10))

        btn_start = tk.Button(card, text="▶ ЗАПУСТИТЬ ОБУЧЕНИЕ", font=("Segoe UI", 12, "bold"), bg=PRIMARY, fg="#FFF",
                              relief=tk.FLAT, padx=20, pady=10, cursor="hand2", command=self.mock_training_process)
        btn_start.pack(pady=(0, 20))

    def mock_training_process(self):
        self.train_progress.pack(fill=tk.X, padx=40, pady=20)
        self.train_progress["value"] = 0
        self.lbl_train_status.config(text="Подготовка данных...", fg=WARNING)

        threading.Thread(target=self._real_training_worker, daemon=True).start()

    def _real_training_worker(self):
        ROOT_DIR = "dataset_train"
        EPOCHS = 20
        BATCH_SIZE = 8
        LEARNING_RATE = 1e-4

        try:
            train_loader, val_loader = create_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE, val_split=0.2)
        except Exception as e:
            self.root.after(0, lambda: self.lbl_train_status.config(text=f"Ошибка данных: {e}", fg="#E74C3C"))
            return

        self.root.after(0, lambda: self.lbl_train_status.config(text="Инициализация модели..."))

        train_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        train_model.fc = nn.Linear(train_model.fc.in_features, 1)
        nn.init.constant_(train_model.fc.bias, 0.5)
        train_model = train_model.to(self.device)

        criterion = nn.L1Loss()
        optimizer = optim.AdamW(train_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_mae = float('inf')
        history_train_loss = []
        history_val_loss = []

        for epoch in range(1, EPOCHS + 1):
            train_model.train()
            train_loss = 0.0
            total_batches = len(train_loader)

            for i, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = train_model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                real_loss = loss.item() * (TARGET_MAX - TARGET_MIN)
                train_loss += real_loss * images.size(0)

                progress = ((epoch - 1) + (i / total_batches)) / EPOCHS * 100
                status_text = f"Обучение: Эпоха {epoch}/{EPOCHS} | Train MAE: {real_loss:.1f}"

                self.root.after(0, lambda p=progress, t=status_text: self.update_train_ui(p, t))

            avg_train_loss = train_loss / len(train_loader.dataset)

            self.root.after(0, lambda e=epoch: self.update_train_ui(((e - 0.5) / EPOCHS * 100),
                                                                    f"Валидация: Эпоха {e}/{EPOCHS}..."))

            train_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = train_model(images)
                    loss = criterion(outputs, targets)
                    real_loss = loss.item() * (TARGET_MAX - TARGET_MIN)
                    val_loss += real_loss * images.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            history_train_loss.append(avg_train_loss)
            history_val_loss.append(avg_val_loss)

            scheduler.step(avg_val_loss / (TARGET_MAX - TARGET_MIN))

            if avg_val_loss < best_val_mae:
                best_val_mae = avg_val_loss
                torch.save(train_model.state_dict(), MODEL_PATH)

        self.root.after(0, lambda: self.finish_training_ui(history_train_loss, history_val_loss))

    def update_train_ui(self, progress_val, text):
        self.train_progress["value"] = progress_val
        self.lbl_train_status.config(text=text)

    def finish_training_ui(self, t_loss, v_loss):
        self.train_progress["value"] = 100
        self.lbl_train_status.config(text="✓ Обучение успешно завершено! Модель сохранена.", fg=SUCCESS)

        self.load_model()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(t_loss) + 1), t_loss, label='Train MAE', marker='o')
        plt.plot(range(1, len(v_loss) + 1), v_loss, label='Val MAE', marker='x')
        plt.title('График обучения модели (MAE)')
        plt.xlabel('Эпоха')
        plt.ylabel('MAE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.show(block=False)

    def show_predict_view(self):
        self.clear_view()
        self.header_lbl.config(text="Анализ фотографии")

        row_frame = tk.Frame(self.view_container, bg=BG_COLOR)
        row_frame.pack(fill=tk.BOTH, expand=True)
        self.current_widgets.append(row_frame)

        left_panel = tk.Frame(row_frame, bg=PANEL_BG, highlightbackground="#E2E8F0", highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        self.img_lbl = tk.Label(left_panel, bg="#E8ECF1", text="Нажмите 'Загрузить фото'", font=("Segoe UI", 12),
                                fg=TEXT_MUTED)
        self.img_lbl.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        btn_load = tk.Button(left_panel, text="📁 Выбрать фото торфа", font=("Segoe UI", 12, "bold"), bg=SUCCESS,
                             fg="#FFF",
                             relief=tk.FLAT, pady=10, cursor="hand2", command=self.process_single_image)
        btn_load.pack(fill=tk.X, padx=15, pady=(0, 15))

        right_panel = tk.Frame(row_frame, bg=BG_COLOR)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        c1, self.lbl_pred_res = self.create_card(right_panel, "ПРЕДСКАЗАННАЯ ВЛАЖНОСТЬ", "---", SUCCESS)
        c1.pack(fill=tk.X, pady=(0, 15))

        c2, self.lbl_true_res = self.create_card(right_panel, "ФАКТИЧЕСКАЯ ВЛАЖНОСТЬ", "---", TEXT_MUTED)
        c2.pack(fill=tk.X)

        self.lbl_pred_status = tk.Label(right_panel, text="Ожидание фото...", font=("Segoe UI", 11), bg=BG_COLOR,
                                        fg=TEXT_MUTED)
        self.lbl_pred_status.pack(pady=20)

    def process_single_image(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if not file_path: return

        img = Image.open(file_path)
        img.thumbnail((350, 350))
        img_tk = ImageTk.PhotoImage(img)
        self.img_lbl.config(image=img_tk, text="")
        self.img_lbl.image = img_tk
        self.lbl_pred_status.config(text="Анализ нейросетью...", fg=PRIMARY)
        self.root.update()

        try:
            img_tensor = Image.open(file_path).convert('RGB')
            tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
            crops = [tf(transforms.RandomCrop(512)(img_tensor)) for _ in range(5)]
            batch = torch.stack(crops).to(self.device)

            with torch.no_grad():
                preds = self.model(batch)

            real_val = preds.mean().item() * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
            self.lbl_pred_res.config(text=f"{real_val:.1f}")

            folder_name = os.path.basename(os.path.dirname(file_path))
            true_val = TARGET_MAPPING.get(folder_name)

            if true_val is not None:
                self.lbl_true_res.config(text=f"{true_val:.1f}", fg=INFO)
                error = abs(true_val - real_val)
                self.lbl_pred_status.config(text=f"Анализ завершен. Погрешность: {error:.1f}", fg=TEXT_MAIN)
            else:
                self.lbl_true_res.config(text="Неизвестно", fg=TEXT_MUTED)
                self.lbl_pred_status.config(text="Анализ завершен. Класс фото неизвестен.", fg=TEXT_MAIN)

        except Exception as e:
            self.lbl_pred_status.config(text=f"Ошибка: {e}", fg="#E74C3C")

    def show_eval_view(self):
        self.clear_view()
        self.header_lbl.config(text="Оценка качества на тестовой выборке")

        ctrl_card = tk.Frame(self.view_container, bg=PANEL_BG, highlightbackground="#E2E8F0", highlightthickness=1)
        ctrl_card.pack(fill=tk.X, pady=(0, 20))
        self.current_widgets.append(ctrl_card)

        self.eval_prog = ttk.Progressbar(ctrl_card, orient="horizontal", mode="determinate", style="TProgressbar")
        self.eval_status = tk.Label(ctrl_card, text="Готов к проверке", font=("Segoe UI", 12), bg=PANEL_BG,
                                    fg=TEXT_MUTED)
        self.eval_status.pack(pady=(15, 5))

        btn_eval = tk.Button(ctrl_card, text="⚙ ЗАПУСТИТЬ ТЕСТИРОВАНИЕ", font=("Segoe UI", 12, "bold"), bg=WARNING,
                             fg="#FFF",
                             relief=tk.FLAT, padx=20, pady=10, cursor="hand2", command=self.run_evaluation)
        btn_eval.pack(pady=(5, 15))

        grid_frame = tk.Frame(self.view_container, bg=BG_COLOR)
        grid_frame.pack(fill=tk.BOTH, expand=True)
        self.current_widgets.append(grid_frame)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        c1, self.lbl_mae = self.create_card(grid_frame, "MAE (Абс. ошибка)", "---", PRIMARY)
        c2, self.lbl_rmse = self.create_card(grid_frame, "RMSE (Квадр. ошибка)", "---", "#E74C3C")
        c3, self.lbl_mape = self.create_card(grid_frame, "MAPE (В процентах)", "---", WARNING)
        c4, self.lbl_r2 = self.create_card(grid_frame, "R² (Детерминация)", "---", SUCCESS)

        c1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        c2.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        c3.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        c4.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

    def run_evaluation(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return

        if not os.path.exists(TEST_DIR):
            messagebox.showerror("Ошибка", f"Папка {TEST_DIR} не найдена!")
            return

        self.eval_prog.pack(fill=tk.X, padx=40, pady=(0, 10))
        self.eval_status.config(text="Чтение файлов...", fg=PRIMARY)

        for lbl in [self.lbl_mae, self.lbl_rmse, self.lbl_mape, self.lbl_r2]:
            lbl.config(text="...")

        threading.Thread(target=self._eval_worker, daemon=True).start()

    def _eval_worker(self):
        try:
            true_vals, pred_vals = [], []
            tf = transforms.Compose(
                [transforms.CenterCrop(512), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

            all_files = []
            for k, val in TARGET_MAPPING.items():
                p = os.path.join(TEST_DIR, k)
                if os.path.exists(p):
                    for f in os.listdir(p):
                        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                            all_files.append((os.path.join(p, f), val))

            total = len(all_files)
            if total == 0:
                self.root.after(0, lambda: self.eval_status.config(text="Тестовая папка пуста!", fg="#E74C3C"))
                return

            for i, (path, t_val) in enumerate(all_files):
                img = Image.open(path).convert('RGB')
                tensor = tf(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    p_val = self.model(tensor).item() * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
                true_vals.append(t_val)
                pred_vals.append(p_val)

                self.root.after(0, lambda v=(i + 1) / total * 100, c=i + 1: self.update_eval_prog(v, c, total))

            t_arr, p_arr = np.array(true_vals), np.array(pred_vals)
            mae = mean_absolute_error(t_arr, p_arr)
            rmse = np.sqrt(mean_squared_error(t_arr, p_arr))
            mape = mean_absolute_percentage_error(t_arr, p_arr) * 100
            r2 = r2_score(t_arr, p_arr)

            self.root.after(0, lambda: self.finish_eval(mae, rmse, mape, r2, t_arr, p_arr))

        except Exception as e:
            self.root.after(0, lambda: self.eval_status.config(text=f"Ошибка: {e}", fg="#E74C3C"))

    def update_eval_prog(self, prog_val, current, total):
        self.eval_prog["value"] = prog_val
        self.eval_status.config(text=f"Обработано {current} из {total} фото...")

    def finish_eval(self, mae, rmse, mape, r2, true_arr, pred_arr):
        self.eval_status.config(text="Формирование графических отчетов...", fg=WARNING)
        self.root.update()

        self.lbl_mae.config(text=f"{mae:.1f}")
        self.lbl_rmse.config(text=f"{rmse:.1f}")
        self.lbl_mape.config(text=f"{mape:.1f} %")
        self.lbl_r2.config(text=f"{r2:.3f}")

        self.show_evaluation_plots(true_arr, pred_arr)

    def show_evaluation_plots(self, true_arr, pred_arr):
        fig1, axs = plt.subplots(1, 2, figsize=(16, 6))

        axs[0].scatter(true_arr, pred_arr, alpha=0.6, color='blue', edgecolors='k')
        min_v, max_v = min(true_arr), max(true_arr)
        axs[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Идеальное совпадение')
        axs[0].set_title('Истинная vs Предсказанная влажность', fontsize=14)
        axs[0].set_xlabel('Истинная влажность торфа', fontsize=12)
        axs[0].set_ylabel('Предсказание модели', fontsize=12)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()

        errors = pred_arr - true_arr
        axs[1].hist(errors, bins=15, color='orange', edgecolor='black', alpha=0.7)
        axs[1].axvline(0, color='red', linestyle='dashed', linewidth=2, label='Нулевая ошибка')
        axs[1].set_title('Распределение ошибок (Предсказание - Истина)', fontsize=14)
        axs[1].set_xlabel('Смещение ошибки', fontsize=12)
        axs[1].set_ylabel('Количество фото', fontsize=12)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()

        fig1.tight_layout()
        fig1.savefig('model_evaluation_report.png', dpi=300)

        class_keys = sorted(list(TARGET_MAPPING.keys()))
        fig2, axes = plt.subplots(2, len(class_keys), figsize=(26, 8))
        fig2.suptitle('Проверка предсказаний модели (7 классов влажности)', fontsize=18, fontweight='bold')

        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

        for col_idx, folder_name in enumerate(class_keys):
            true_val = TARGET_MAPPING[folder_name]
            f_path = os.path.join(TEST_DIR, folder_name)

            files = [f for f in os.listdir(f_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.isdir(
                f_path) else []

            if len(files) >= 2:
                selected_files = random.sample(files, 2)
            elif len(files) == 1:
                selected_files = [files[0], files[0]]
            else:
                selected_files = [None, None]

            for row_idx in range(2):
                ax = axes[row_idx, col_idx]
                file_name = selected_files[row_idx]

                if file_name is None:
                    ax.axis('off')
                    ax.set_title(f"Нет фото\n({true_val})", fontsize=10)
                    continue

                img_path = os.path.join(f_path, file_name)

                try:
                    img = Image.open(img_path).convert('RGB')
                    crops = [tf(transforms.RandomCrop(512)(img)) for _ in range(5)]
                    batch = torch.stack(crops).to(self.device)

                    with torch.no_grad():
                        preds = self.model(batch)

                    pred_val = preds.mean().item() * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
                    error = abs(true_val - pred_val)

                    img_show = img.copy()
                    img_show.thumbnail((800, 800))
                    ax.imshow(img_show)
                    ax.axis('off')

                    color = 'green' if error < 100 else 'red'
                    ax.set_title(f"Истина: {true_val}\nПредсказ.: {pred_val:.1f}\nОшибка: {error:.1f}", fontsize=12,
                                 color=color, fontweight='bold')
                except Exception:
                    ax.axis('off')

        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.savefig('test_predictions_grid.png', dpi=300, bbox_inches='tight')

        self.eval_status.config(text="✓ Тестирование завершено! Графики открыты.", fg=SUCCESS)

        plt.show(block=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()