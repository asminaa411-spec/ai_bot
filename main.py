import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

def classify_image(image_path):
    """
    Функция принимает путь к картинке и возвращает предсказание нейросети.
    """
    print("Загрузка модели... (это может занять время при первом запуске)")

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    print("Модель готова. Обрабатываем изображение...")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Ошибка открытия файла: {e}"

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()

    return f"На изображении: {label} (Уверенность: {confidence:.2%})"


def main():
    image_file = input("Введите имя файла  (например, cat.jpg): ")
    
    if image_file:
        result = classify_image(image_file)
        print(result)
    else:
        print("Файл не указан.")


if __name__ == "__main__":
    main()
