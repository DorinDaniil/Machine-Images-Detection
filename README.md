# Machine-Images-Detection

Detection of generated and partly generated images.

![](assets/architecture.svg)

## Модель

Обычный классификатор на DINOv3 ConvNeXt-Tiny: softmax по классам «реальная» и семействам генераторов
(OpenAI, Google, Flux, SDXL и так далее, `model.families` в конфиге). Картинка сгенерирована, если
`1 − p(реальная) > 0.5`; семейство — самый вероятный из остальных классов. Порога рядом с моделью нет:
после обучения он записывается в смещение класса «реальная».

```python
model = Detector.load('runs/base/model.pt')
p = model(image).softmax(1)   # image [B, 3, 512, 512] от 0 до 1; p[:, 0] — «реальная», дальше model.classes
```

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

Каждая картинка попадает в обучение (85 %), валидацию (5 %) или тест (10 %) по хешу «папка/файл»:
разбиение не меняется между запусками и при добавлении картинок.

## Порядок

1. `python scripts/download_data.py --config configs/train.yaml` качает OpenFake и SID_Set сразу в эти папки.
   Можно прерывать и запускать снова. В конце печатает число картинок по генераторам.
2. Свои реальные картинки: положить папку в `data/real/` или указать её в `data.real_dirs`. Подпапки читаются.
3. Свой генератор: папка `data/fake/<имя>/`. Имя, добавленное в `model.families`, попадает в класс семейства;
   без этого генератор становится отдельным классом.
4. `python scripts/run_train.py --config configs/train.yaml` учит модель и пишет запуск в `runs/base`:
   `model.pt`, `metrics.csv`, `config.yaml`. Обучение остановилось:
   `python scripts/run_train.py --config runs/base/config.yaml --resume` продолжает с последнего сохранения
   (после каждой эпохи и каждые `train.save_every` шагов).
5. `python scripts/run_test.py --config runs/base/config.yaml` пишет `report.md` в папку запуска.

## Варианты обучения

Вариант — тот же конфиг с другими значениями и своей папкой:

```
python scripts/run_train.py --config configs/train.yaml --set train.out=runs/balance train.balance=0.5
python scripts/run_train.py --config configs/train.yaml --set train.out=runs/contrast train.contrast=0.5
python scripts/run_test.py --config runs/contrast/config.yaml
```

| Ключ | Что делает |
|---|---|
| `train.balance` | Картинка генератора с n картинками берётся в эпоху с весом n^(−balance): редкие генераторы модель видит чаще. 0 — каждая картинка по одному разу. |
| `train.contrast` | Вес InfoNCE на векторах картинок: один класс стягивается, разные расталкиваются. Сравнение идёт с векторами последних шагов, поэтому хватает и маленького батча. 0 — выключен. |

После каждого `run_test.py` все проверенные запуски сводятся в `runs/compare.md`: столбец на запуск,
строки — итоги, FPR по источникам реальных и TPR по генераторам.
