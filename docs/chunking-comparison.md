# Сравнение стратегий chunking

Отчёт сгенерирован командой `index-mcp compare` по базе индекса; все числа — из прогона.

- Корпус: 1 файлов `.docx`, 397365 символов.
- Модель эмбеддингов: `nomic-embed-text`, размерность 768.
- Вопросов: 14.
- Параметры `fixed`: `{"chunker":{"chunk_size":1200,"overlap":200},"doc_prefix":"search_document: ","num_ctx":2048}`.
- Параметры `structure`: `{"chunker":{"max_section":1500,"min_section":200},"doc_prefix":"search_document: ","num_ctx":2048}`.

## Метрики

| Метрика | `fixed` | `structure` |
|---|---|---|
| Чанков | 432 | 367 |
| Длина, символов: min / median / p95 / max | 565 / 1135 / 1197 / 1200 | 247 / 1207 / 1466 / 1497 |
| Пересекают границу раздела | 31.0% (134) | 0.3% (1) |
| Обрезаны посреди предложения | 6.0% (26) | 0.0% (0) |
| Время эмбеддинга | 77.4 с (179 мс на чанк) | 70.4 с (192 мс на чанк) |
| hit@1 | 0.36 (5/14) | 0.57 (8/14) |
| hit@5 | 0.86 (12/14) | 0.93 (13/14) |
| MRR | 0.584 | 0.707 |
| Ожидаемого раздела нет среди чанков | 0 из 14 | 0 из 14 |

## Ранг первого попадания по вопросам

| # | Вопрос | Ожидаемый раздел | `fixed` | `structure` |
|---|---|---|---|---|
| 1 | В каком порядке Android убивает процессы при нехватке памяти? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 1. Платформа Android и приложение > 1.1 Устройство платформы > 1.1.2 Процессы и приоритеты: foreground/visible/service/background/empty, low memory killer | 1 | 16 |
| 2 | Чем onSaveInstanceState отличается от ViewModel и какой лимит у Bundle? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 2. Activity > 2.1 Жизненный цикл > 2.1.3 onSaveInstanceState/onRestoreInstanceState vs ViewModel: лимиты Bundle | 3 | 1 |
| 3 | Как работает back stack и что делает taskAffinity? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 2. Activity > 2.2 Запуск и навигация > 2.2.2 Task и back stack, флаги Intent, taskAffinity | 2 | 1 |
| 4 | Какие типы foreground service нужно указывать начиная с Android 14? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 5. Service и фоновая работа > 5.1 Service > 5.1.3 Foreground service: уведомление, типы FGS (Android 14+), ограничения запуска из фона (Android 12+) | 1 | 1 |
| 5 | Что такое Doze mode и App Standby Buckets? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 5. Service и фоновая работа > 5.3 Ограничения фонового выполнения > 5.3.1 Doze mode, App Standby Buckets, background execution limits (Android 8+), exact alarms (SCHEDULE_EXACT_ALARM) | 2 | 1 |
| 6 | Чем volatile отличается от synchronized и Atomic-классов? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 7. Классическая многопоточность > 7.1 Основы > 7.1.2 Thread, Runnable, синхронизация (synchronized, volatile, Atomic), happens-before кратко | 1 | 1 |
| 7 | Почему отмена корутин кооперативная и как работает NonCancellable? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 8. Kotlin Coroutines > 8.3 Структурированная конкурентность > 8.3.2 Отмена: CancellationException, isActive/ensureActive/yield, try-finally, NonCancellable | 6 | 4 |
| 8 | Чем postValue отличается от setValue в LiveData? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 9. Flow и LiveData > 9.2 LiveData > 9.2.1 LiveData vs StateFlow: lifecycle-awareness, postValue vs setValue, MediatorLiveData | 2 | 2 |
| 9 | Какие стандартные компоненты и скоупы есть в Hilt? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 12. Dependency Injection > 12.2 Dagger 2 и Hilt > 12.2.2 Hilt: @HiltAndroidApp, @AndroidEntryPoint, @HiltViewModel, стандартные компоненты и скоупы, assisted injection | 3 | 3 |
| 10 | Почему DataStore лучше SharedPreferences и чем apply отличается от commit? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 14. Хранение данных > 14.2 Прочие способы > 14.2.1 SharedPreferences vs DataStore (Preferences/Proto): синхронность, apply() vs commit(), почему DataStore | 1 | 1 |
| 11 | Чем remember отличается от rememberSaveable и что такое state hoisting? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 16. Jetpack Compose > 16.1 Основы мышления > 16.1.2 State: mutableStateOf, remember vs rememberSaveable, state hoisting | 1 | 1 |
| 12 | Какие фазы проходит кадр в Jetpack Compose? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 16. Jetpack Compose > 16.2 Побочные эффекты и жизненный цикл > 16.2.2 Фазы Compose (composition → layout → drawing), что читать в какой фазе, snapshot system кратко | 69 | 4 |
| 13 | Как LeakCanary находит утечки памяти? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 19. Память, производительность и стабильность > 19.1 Память > 19.1.2 LeakCanary: принцип работы, типичные сценарии утечек (фрагменты, listeners, non-static inner classes) | 2 | 1 |
| 14 | Чем lateinit отличается от by lazy и == от ===? | Конспект для подготовки к собеседованию: Middle Android Developer (Kotlin) > 20. Kotlin для Android-собеседования > 20.1 Ядро языка > 20.1.5 equals/hashCode, == vs ===, lazy initialization, lateinit vs by lazy | 3 | 2 |

«—» — чанка с ожидаемым разделом нет во всей выдаче.

## Вывод

По качеству поиска на этом корпусе лучше `structure`: MRR 0.584 у `fixed`, 0.707 у `structure`; hit@1 0.36 у `fixed`, 0.57 у `structure`; hit@5 0.86 у `fixed`, 0.93 у `structure`. Границу раздела пересекают 31.0% у `fixed`, 0.3% у `structure`, посреди предложения обрезаны 6.0% у `fixed`, 0.0% у `structure`. Чанков 432 у `fixed`, 367 у `structure`, медианная длина 1135 у `fixed`, 1207 у `structure` символов, максимальная 1200 у `fixed`, 1497 у `structure`. Абсолютные значения ограничены моделью `nomic-embed-text`, обученной в основном на английском, при русских конспектах; сравнение при этом честное, потому что модель, префиксы и вопросы у стратегий общие.

## Толкование

Первый прогон на прежних умолчаниях (`num_ctx` 8192, `max_section` 3000) упал на эмбеддинге `structure` с `the input length exceeds the context length`: у `nomic-embed-text` в Ollama 0.33.3 контекст 2048 токенов, а цифры выше — с новыми умолчаниями 2048 и 1500. `structure` ставит ожидаемый раздел выше, чем `fixed`, в 7 вопросах из 14, ниже — в одном, в остальных 6 ранг одинаковый. Самый большой разрыв — вопрос 12 про фазы Compose: ранг 69 у `fixed` и 4 у `structure`; обратный случай — вопрос 1 про приоритеты процессов: ранг 1 у `fixed` и 16 у `structure`. Границу раздела у `structure` пересекает 1 чанк из 367 (0.3%), у `fixed` — 134 из 432 (31.0%). Выборка мала: 14 вопросов по одному конспекту, и разница hit@1 (8 против 5) — это 3 вопроса.
