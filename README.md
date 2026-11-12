# Machine-Images-Detection

Detection of generated and partly generated images.

![](assets/architecture.svg)

## Данные

```
data/
  real/<источник>/       реальные картинки: laion, pexels, docci и свои папки
  fake/<генератор>/      сгенерированные; имя папки — это имя генератора
  renders/               тексты, интерфейсы, графики и логотипы, рисуются кодом
  sid_validation/        внешний тест SID_Set, рядом sid_validation.csv
  openfake_train.csv     списки скачанного из OpenFake с промптами;
  openfake_test.csv      пока файл на месте, сплит считается скачанным
```

Порядок:

1. `python scripts/download_data.py --config configs/train.yaml` качает OpenFake и SID_Set сразу в эти папки.
   Можно прерывать и запускать снова. Служебные папки загрузки он удаляет сам, в конце печатает число
   картинок по генераторам.
2. Свои реальные картинки: положить папку в `data/real/` или указать её в `data.real_dirs`. Подпапки читаются.
3. Свой генератор: папка `data/fake/<имя>/`. Чтобы у него был свой эксперт, добавить имя в `model.families`.
4. `python scripts/run_train.py --config configs/train.yaml`, затем `python scripts/run_test.py --config configs/train.yaml`.

Каждая картинка попадает в обучение (85 %), валидацию (5 %) или тест (10 %) по хешу «папка/файл»:
разбиение не меняется между запусками и при добавлении картинок.
