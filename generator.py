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
    
def imageGenerator():
    # Загрузка сохраненных параметров модели
    checkpoint = torch.load('model.pth')
    img_channels = 3
    latent_dim = 100

    # Создание и загрузка генератора
    generator = Generator(latent_dim, img_channels)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    # Генерация изображения
    with torch.no_grad():
        # Сэмплирование случайного вектора из скрытого пространства
        z = torch.randn(1, latent_dim)
        
        # Генерация изображения
        generated_image = generator(z)

        generated_image_pil = transforms.ToPILImage()(generated_image.squeeze())

        # Обрезка изображения до размера 64x32
        cropped_image = generated_image_pil.crop((0, 0, 64, 32))

        mask = Image.open('static/mask.png').convert("RGBA")
        #alpha = mask.split()[3]
        result = Image.alpha_composite(cropped_image.convert("RGBA"), mask)

        return result
