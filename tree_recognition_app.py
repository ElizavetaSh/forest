import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import time
import onnxruntime as ort
import clip
# import streamlit as st
# import paddle
# from paddleseg.models import BiSeNetV2, UNet, DeepLabV3P
# from paddleseg.transforms import Compose, Normalize, Resize
# from paddleseg.cvlibs import Config

# from appp.inferene_seg import BinaryTreeSegmentator, create_binary_deeplabv3_mobilenet
from clip_inference import run_clip, species_dict,species, options, options_dict

class TreeRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание пород деревьев")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Инициализация модели
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.pressor = self.load_model()
        self.class_names = ['Дуб', 'Береза', 'Сосна', 'Ель', 'Клен']
        
        # Переменные
        self.original_image = None
        self.processed_image = None
        self.mask = None
        
        self.setup_ui()
        
    def load_model(self):
        """Загрузка или создание модели"""
        # model_path = "binary_deeplabv3_mobilenet_best.pth"
        # model = create_binary_deeplabv3_mobilenet()  # или create_binary_unet()
        model_clip, preprocess = clip.load("ViT-L/14@336px", device=self.device)
       
        # Загружаем веса
        # checkpoint = torch.load(model_path, map_location=self.device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model.to(self.device)
        # model.eval()
        return  model_clip, preprocess
    
    def setup_ui(self):
        """Создание интерфейса"""
        # Заголовок
        title_label = tk.Label(self.root, text="Распознавание пород деревьев", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Фрейм для кнопок
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Кнопки управления
        self.load_btn = ttk.Button(button_frame, text="Загрузить изображение", 
                                  command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = ttk.Button(button_frame, text="Обработать", 
                                     command=self.process_image, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Сохранить результат", 
                                  command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Фрейм для изображений
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Оригинальное изображение
        original_frame = ttk.LabelFrame(image_frame, text="Оригинальное изображение")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg='white', width=400, height=400)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Обработанное изображение
        processed_frame = ttk.LabelFrame(image_frame, text="Результат с маской")
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_canvas = tk.Canvas(processed_frame, bg='white', width=400, height=400)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Фрейм для информации
        info_frame = ttk.LabelFrame(main_frame, text="Информация о распознавании")
        info_frame.pack(fill=tk.X, pady=10)
        
        self.info_text = tk.Text(info_frame, height=6, width=80)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image, self.original_canvas)
                self.process_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                self.update_info("Изображение загружено успешно")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")
    
    def display_image(self, image, canvas):
        """Отображение изображения на canvas"""
        # Получаем размеры canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 400
        
        # Масштабируем изображение
        image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        
        # Конвертируем для tkinter
        photo = ImageTk.PhotoImage(image)
        
        # Обновляем canvas
        canvas.delete("all")
        canvas.image = photo
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
    
    def process_image(self):
        """Обработка изображения нейронной сетью"""
        if self.original_image is None:
            return
        original_size = self.original_image.size
        self.progress.start()
        self.update_info("Начинается обработка...")
        
        try:
            # Преобразуем изображение для модели
            transform = transforms.Compose([
                transforms.Resize((768, 768)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(self.original_image).unsqueeze(0).to(self.device)
            session = ort.InferenceSession("segformer_b0_cityscapes_1024x512_160k_model.onnx")
            
            # Обработка нейронной сетью
            with torch.no_grad():
                input_name = session.get_inputs()[0].name
    
                # Выполняем инференс
                output = session.run(None, {input_name: image_tensor.cpu().numpy()})
                output = output[0]
                prediction = np.argmax(output, axis=1)
                if np.max(prediction[0])>0:
                    sp_tok = clip.tokenize(species).to(self.device)
                    opt_tok = clip.tokenize(options_dict).to(self.device)

                    pred_species = run_clip(self.clip_model, self.pressor, self.original_image, sp_tok, species, species_dict)
                    pred_options = run_clip(self.clip_model, self.pressor, self.original_image, opt_tok, options, options_dict)
                    # Создаем цветную маску

                    resize_pred = cv2.resize(prediction[0].astype(np.uint8), 
                                            original_size,  # (width, height)
                                            interpolation=cv2.INTER_NEAREST)
                    self.mask = self.create_colored_mask(resize_pred)
        
                    # Создаем результат с наложением маски
                    result_image = self.overlay_mask_on_image(self.original_image, self.mask)
                    self.processed_image = result_image

                    self.display_image(result_image, self.processed_canvas)

                    # Обновляем информацию
                    self.update_detection_info(prediction, pred_species, pred_options)
                    self.save_btn.config(state=tk.NORMAL)
                else:

                    self.update_detection_info(None, None, None)
                    self.save_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обработке: {str(e)}")
        finally:
            self.progress.stop()
    
    def create_colored_mask(self, prediction):
        """Создание цветной маски"""
        # Цвета для разных классов
        colors = {
            0: [0, 0, 0],    # Дуб - красный
            1: [0, 255, 0],    # Береза - зеленый
    
        }
        
        mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            mask[prediction == class_id] = color
        
        return Image.fromarray(mask)
    
    def overlay_mask_on_image(self, original_image, mask):
        """Наложение маски на оригинальное изображение"""
        # Приводим к одинаковому размеру
        # original_resized = original_image.resize((256, 256), Image.Resampling.LANCZOS)
        # mask_resized = mask.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Конвертируем в numpy arrays
        original_np = np.array(original_image)
        mask_np = np.array(mask)         
        # Смешиваем изображения
        alpha = 0.6  # Прозрачность маски
        blended = cv2.addWeighted(original_np, 1 - alpha, mask_np, alpha, 0)
        
        return Image.fromarray(blended)
    
    def update_info(self, message):
        """Обновление информационного текста"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"{message}\n")
    
    def update_detection_info(self, prediction, pred_spesies, pred_options):

        """Обновление информации о распознавании"""
        if pred_spesies:
           
            info_text = "Результаты распознавания:\n\n"
            info_text += f"{pred_spesies}\n"
            if isinstance(pred_options, list):
                for i in pred_options:
                    info_text += f"Присудствуют такие болезни {i} \n"
            else:
                info_text += f"Присудствуют такие болезни {pred_options}\n"
            
            # info_text += f"\nВсего обработано пикселей: {total_pixels}"
            # info_text += f"\nОбнаружено классов: {len(unique_classes)}"
        else:
            info_text = "Ничего не найдено"
        self.update_info(info_text)
        
    
    def save_result(self):
        """Сохранение результата"""
        if self.processed_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.processed_image.save(file_path)
                messagebox.showinfo("Успех", "Результат сохранен успешно!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить: {str(e)}")

def main():
    try:
        root = tk.Tk()
        app = TreeRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Ошибка при запуске приложения: {e}")
        input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()