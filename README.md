# Minecraft Skin Generator

![Minecraft Skin Generator](<demo/3.JPG>)

## О проекте

Minecraft Skin Generator - это проект, посвященный генерации уникальных скинов для персонажей в игре Minecraft. Модель обучена с использованием библиотеки PyTorch на датасете, содержащем 100 тысяч изображений скинов размером 64 на 64 пикселя.

## Визуальный интерфейс

Для удобства пользователей был разработан визуальный интерфейс, который позволяет создавать персонализированные скины. Модифицированный скрипт для отображения 3D модели скина на фронтенде основан на коде, предоставленном [djazz](https://djazz.se/apps/MinecraftSkin/), и использует Three.js. 

![Example](<demo/1.jpg>)

## Технические детали

Для обучения генеративно-состязательной сети (GAN) использовался датасет из 100 тысяч изображений скинов. Данные были разделены на классы в зависимости от доминантного цвета, что улучшило качество обучения. 

Процесс обучения включает в себя генерацию случайных векторов, обработку через генератор и оценку подлинности с использованием дискриминатора.

![Education](<demo/2.JPG>)

## Запуск локально

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/Grandher/Minecraft-skin-generator_python
    ```

2. Установите необходимые зависимости:

    ```bash
    pip install -r requirements.txt
    ```

3. Запустите приложение:

    ```bash
    python server.py
    ```

Теперь вы можете открыть браузер и перейти по адресу [http://localhost:5000](http://localhost:5000), чтобы начать генерацию уникальных скинов для Minecraft.
