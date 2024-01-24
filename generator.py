import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Определение класса Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Определение класса Classifier
class Classifier(nn.Module):
    def __init__(self, img_channels):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

def imageGenerator(classifier_threshold=0.5):
    # Загрузка сохраненных параметров модели генератора
    generator_checkpoint = torch.load('model.pth')
    img_channels = 3
    latent_dim = 100

    # Создание и загрузка генератора
    generator = Generator(latent_dim, img_channels)
    generator.load_state_dict(generator_checkpoint['generator_state_dict'])
    generator.eval()

    # Создание и загрузка классификатора
    classifier = Classifier(img_channels)
    classifier_checkpoint = torch.load('classifier.pth')
    classifier.load_state_dict(classifier_checkpoint['classifier_state_dict'])
    classifier.eval()

    # Генерация изображения
    with torch.no_grad():
        # Сэмплирование случайного вектора из скрытого пространства
        z = torch.randn(1, latent_dim)
        
        # Генерация изображения
        generated_image = generator(z)

        # Предсказание успешности генерации с помощью классификатора
        classifier_input = generated_image.clone().detach()
        classifier_input.requires_grad = False
        classifier_output = classifier(classifier_input)

        # Принятие решения на основе классификатора
        if classifier_output.item() < classifier_threshold:
            # Генерация неудачна, перегенерируем изображение
            return imageGenerator(classifier_threshold=classifier_threshold)

        generated_image_pil = transforms.ToPILImage()(generated_image.squeeze())

        # Обрезка изображения до размера 64x32
        cropped_image = generated_image_pil.crop((0, 0, 64, 32))

        mask = Image.open('static/mask.png').convert("RGBA")
        result = Image.alpha_composite(cropped_image.convert("RGBA"), mask)

        return result
