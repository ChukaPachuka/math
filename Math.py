import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import kruskal

# 1. Подключение к БД и загрузка данных
db_params = {
    "host": "95.163.241.236",
    "port": 5432,
    "dbname": "apteka",
    "user": "student",
    "password": "qweasd963"
}

def load_data(query):
    conn = psycopg2.connect(**db_params)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

sales = load_data("SELECT * FROM sales;")
bonuscheques = load_data("SELECT * FROM bonuscheques;")
employee = load_data("SELECT * FROM employee;")
shops = load_data("SELECT * FROM shops;")

# 2. Анализ продаж по дням
sales["total_sales"] = sales["dr_croz"] * sales["dr_kol"]
sales_by_date = sales.groupby("dr_dat")["total_sales"].sum().reset_index() # группируем продажи по датам

### Визуализация
plt.figure(figsize=(12, 5))
plt.plot(sales_by_date["dr_dat"], sales_by_date["total_sales"], marker='o')
plt.xlabel("Дата", fontsize=8)
plt.ylabel("Объем продаж", fontsize=8)
plt.title("Динамика продаж по дням", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# 3. Средний чек
sales_clean = sales.dropna(subset=["dr_croz", "dr_kol", "dr_sdisc"]) # убираем пропущенные значения и возможные ошибки
sales_clean = sales_clean[(sales_clean["dr_croz"] > 0) & (sales_clean["dr_kol"] > 0)]

sales_clean["position_total"] = (sales_clean["dr_croz"] - sales_clean["dr_sdisc"]) * sales_clean["dr_kol"] # считаем сумму чека с учетом скидок
avg_check = sales_clean.groupby(["dr_apt", "dr_nchk", "dr_dat"])["position_total"].sum().reset_index(name="check_total") # группируем продажи по чекам (аптека, номер чека, дата), считаем сумму каждого чека
avg_check_by_date = avg_check.groupby("dr_dat")["check_total"].mean() # средний чек по дням

### Визуализация
plt.figure(figsize=(12, 5))
plt.plot(avg_check_by_date.index, avg_check_by_date.values, marker='o', color='r')
plt.xlabel("Дата", fontsize=8)
plt.ylabel("Средний чек", fontsize=8)
plt.title("Средний чек по дням", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# 4. Влияние скидок на продажи
sales["dr_sdisc"] = sales["dr_sdisc"].fillna(0) # pаполняем пропущенные значения скидок нулями
sales["discount_pct"] = (sales["dr_sdisc"] / sales["dr_croz"]) * 100 # размер скидки в процентах
sales["discount_pct"] = sales["discount_pct"].clip(0, 100) # проерка, что скидка не может быть больше 100%

bins = [0, 5, 10, 20, 50, 100]  # # группируем скидки по диапазонам
labels = ["0-5%", "5-10%", "10-20%", "20-50%", "50%+"]
sales["discount_range"] = pd.cut(sales["discount_pct"], bins=bins, labels=labels, right=False)
sales["total_sales"] = (sales["dr_croz"] - sales["dr_sdisc"]) * sales["dr_kol"] # добавляем столбец с общей суммой продаж
discount_sales = sales.groupby("discount_range")["total_sales"].sum() # группируем сумму продаж по диапазонам скидок

### Визуализация
plt.figure(figsize=(8, 5))
discount_sales.plot(kind="bar", color="g")
plt.xlabel("Размер скидки)", fontsize=8)
plt.ylabel("Объем продаж", fontsize=8)
plt.title("Объем продаж товаров со скидкой", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# 5. Топ-10 товаров по количеству продаж
sales_filtered = sales[~sales["dr_ndrugs"].isin(["ПАКЕТ", "Карта LOYALITY 25Р", "Карта LOYALITY 0,01Р"])] # убираем "Пакет" и "Бонусная карта" из топа товаров)))
top_products = sales_filtered["dr_ndrugs"].value_counts().head(10) # топ-10 товаров по количеству продаж
short_labels = top_products.index.to_series().apply(lambda x: x[:30] + "…" if len(x) > 30 else x) # обрезаем длинные названия (до 30 символов, а то не умещаются на графике)

### Визуализация
plt.figure(figsize=(10, 5))
top_products.plot(kind="bar", color="b")
plt.xlabel("Товар", fontsize=8)
plt.ylabel("Количество продаж", fontsize=8)
plt.title("Топ-10 продаваемых товаров", fontsize=10)
plt.xticks(range(len(short_labels)), short_labels, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# 6. Влияние бонусов на сумму чека
bonuscheques["has_bonus"] = bonuscheques["bonus_spent"] > 0 # группируем покупки по использованию (списанию) бонусов
bonus_sales = bonuscheques.groupby("has_bonus")["summ"].sum()

# Визуализация
plt.figure(figsize=(6, 4))
bonus_sales.plot(kind="bar", color=["gray", "orange"])
plt.xticks(ticks=[0, 1], labels=["Без бонусов", "С бонусами"], rotation=0, fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Тип покупки", fontsize=8)
plt.ylabel("Объем продаж", fontsize=8)
plt.title("Продажи со использованием бонусов и без", fontsize=10)
plt.tight_layout()
plt.show()

# 7. Топ аптек по продажам
sales_shops = sales.merge(shops, left_on="dr_apt", right_on="id", how="left") # соединяем продажи с аптеками
sales_shops["total_sales"] = sales_shops["dr_croz"] * sales_shops["dr_kol"] # добавляем столбец с суммой продаж
shop_sales = sales_shops.groupby("name")["total_sales"].sum().sort_values(ascending=False).head(10) # группируем по названию аптеки и суммируем продажи

### Визуализация
plt.figure(figsize=(10, 5))
shop_sales.plot(kind="bar", color="purple")
plt.xlabel("Аптека", fontsize=8)
plt.ylabel("Объем продаж", fontsize=8)
plt.title("Топ аптек по продажам", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# 8. Гистрограмма продаж по чекам
sales["position_total"] = (sales["dr_croz"] - sales["dr_sdisc"]) * sales["dr_kol"] # добавляем сумму по каждой товарной позиции
sales_check_total = sales.groupby(["dr_apt", "dr_nchk", "dr_dat"])["position_total"].sum().reset_index() # группируем по номеру чека, аптеке и дате (чтобы получить сумму всего чека)
sales_check_total.rename(columns={"position_total": "total_check_amount"}, inplace=True) # переименовываем колонку для удобства
sales_clean = sales_check_total["total_check_amount"].dropna() # убираем пропущенные и отрицательные значения
sales_clean = sales_clean[sales_clean > 0]
low, high = np.percentile(sales_clean, [1, 99]) # определяем границы оси X (убираем выбросы, 1-й и 99-й процентили)

### Визуализация
plt.figure(figsize=(10, 5))
sns.histplot(sales_clean, bins=50, kde=True, color="blue")
plt.xlim(low, high) # ограничиваем ось X, чтобы убрать пустые области
plt.xlabel("Сумма чека", fontsize=8)
plt.ylabel("Частота", fontsize=8)
plt.title("Распределение сумм чеков", fontsize=10)
plt.show()

# 9. Основные метрики (среднее, медиана, стандартное отклонение)
stats = {
    "Среднее": sales["total_sales"].mean(),
    "Медиана": sales["total_sales"].median(),
    "Мода": sales["total_sales"].mode()[0],
    "Стандартное отклонение": sales["total_sales"].std(),
    "Коэффициент вариации": sales["total_sales"].std() / sales["total_sales"].mean(),
} # вычисляем основные статистические показатели

for key, value in stats.items():
    print(f"{key}: {value:,.2f}") # выводим статистику

### Визуализация
sales["total_sales"] = (sales["dr_croz"] - sales["dr_sdisc"]) * sales["dr_kol"] # добавляем сумму каждого чека (учитываем скидки)
sales_clean = sales["total_sales"].dropna() # убираем пропущенные и отрицательные значения
sales_clean = sales_clean[sales_clean > 0]

mean_value = sales_clean.mean() # вычисляем основные статистические показатели
median_value = sales_clean.median()
mode_value = sales_clean.mode()[0]  # наиболее часто встречающееся значение
std_dev = sales_clean.std()
coefficient_variation = (std_dev / mean_value) * 100  # CV в процентах

### Визуализация
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

sns.boxplot(x=sales_clean, ax=ax[0], color="lightblue") # боксплот
ax[0].set_title("Распределение суммы продаж")
ax[0].set_xlabel("Сумма продаж")

sns.histplot(sales_clean, bins=50, kde=True, color="blue", ax=ax[1]) # гистограмма

ax[1].axvline(mean_value, color="r", linestyle="--", label=f"Среднее: {mean_value:,.0f}") # вертикальные линии
ax[1].axvline(median_value, color="g", linestyle="--", label=f"Медиана: {median_value:,.0f}")
ax[1].axvline(mode_value, color="purple", linestyle="--", label=f"Мода: {mode_value:,.0f}")
ax[1].axvline(mean_value + std_dev, color="orange", linestyle="--", label=f"Стандартное отклонение (+1σ)")
ax[1].axvline(mean_value - std_dev, color="orange", linestyle="--")

ax[1].text(
    0.05, 0.9, 
    f"Коэффициент вариации (CV): {coefficient_variation:.2f}%", 
    transform=ax[1].transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.8)
) # текст с коэффициентом вариации (CV)

ax[1].set_title("Гистограмма суммы продаж")
ax[1].set_xlabel("Сумма продаж")
ax[1].set_ylabel("Частота")
ax[1].legend()

plt.tight_layout()
plt.show()

# 10. Корреляции между скидками, ценами и продажами
corr_df = sales[["dr_sdisc", "dr_croz", "dr_kol", "total_sales"]] # выбираем ключевые переменные
corr_matrix = corr_df.corr() # вычисляем корреляционную матрицу

### Визуализация
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Корреляция продажи, цены и скидки", fontsize=10)
plt.show()

# 11. Асимметрия и эксцесс (форма распределения)
asymmetry = skew(sales["total_sales"]) # вычисляем асимметрию и эксцесс
excess_kurtosis = kurtosis(sales["total_sales"])

print(f"Асимметрия: {asymmetry:.2f}")
print(f"Эксцесс (избыточная острота распределения): {excess_kurtosis:.2f}")

### Визуализация
sales_clean = sales["total_sales"].dropna() # убираем пропущенные и отрицательные значения
sales_clean = sales_clean[sales_clean > 0]
asymmetry = skew(sales_clean) # вычисляем асимметрию и эксцесс
excess_kurtosis = kurtosis(sales_clean)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

sns.boxplot(x=sales_clean, ax=ax[0], color="lightblue") # боксплот (для смещения)
ax[0].set_title("Асимметрия и эксцесс распределения")

sns.histplot(sales_clean, bins=50, kde=True, color="blue", ax=ax[1]) # гистограмма + KDE (для формы распределения)

ax[1].text(
    0.05, 0.9, 
    f"Асимметрия: {asymmetry:.2f}\nЭксцесс: {excess_kurtosis:.2f}", 
    transform=ax[1].transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.8)
) # текст с коэффициентами асимметрии и эксцесса

ax[1].set_title("Гистограмма суммы продаж с асимметрией и эксцессом")
ax[1].set_xlabel("Сумма продаж")
ax[1].set_ylabel("Частота")

plt.tight_layout()
plt.show()

# 12. Формулировка гипотез
## Гипотеза 1. Средний чек покупок с бонусами выше, чем без бонусов (t-тест)
## Гипотеза 2. Чем больше скидка, тем выше объем продаж (корреляция между скидкой и объемом продаж)
## Гипотеза 3. Медианная сумма продаж отличается в разных аптеках (ANOVA / критерий Крускала-Уоллиса)
## Гипотеза 4. Дорогие товары покупают реже, чем дешевые (корреляция между ценой и количеством продаж)
## Гипотеза 5. Чеки с онлайн-заказами (интернет-заказ) в среднем выше, чем в обычных магазинах (t-тест)
## Гипотеза 6. Скидки сильнее влияют на дорогие товары, чем на дешевые (коэффициент эластичности)

# 13. Проверка гипотез
## Гипотеза 1
bonus_purchases = bonuscheques[bonuscheques["bonus_spent"] > 0]["summ_with_disc"] # разделяем данные на две группы: со списанием бонусов и без
no_bonus_purchases = bonuscheques[bonuscheques["bonus_spent"] == 0]["summ_with_disc"]

t_stat, p_value = ttest_ind(bonus_purchases, no_bonus_purchases, equal_var=False) # проверяем средние чеки
print("\n📊 Гипотеза 1: Средний чек покупок с бонусами выше, чем без бонусов")
print(f"T-статистика: {t_stat:.2f}")
print(f"P-значение: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Гипотеза подтверждена: средний чек с использованием бонусов статистически выше")
else:
    print("❌ Гипотеза НЕ подтверждена: разница в среднем чеке статистически незначима")

bonus_df = pd.DataFrame({
    "Сумма чека": list(bonus_purchases) + list(no_bonus_purchases),
    "Группа": ["С бонусами"] * len(bonus_purchases) + ["Без бонусов"] * len(no_bonus_purchases)
}) # создаем dataframe для визуализации

### Визуализация
plt.figure(figsize=(10, 5))
sns.boxplot(x="Группа", y="Сумма чека", data=bonus_df, palette=["orange", "gray"]) # boxplot
plt.title("Сравнение среднего чека: с бонусами vs без бонусов")
plt.show()

### 
plt.figure(figsize=(10, 5))
sns.violinplot(x="Группа", y="Сумма чека", data=bonus_df, palette=["orange", "gray"]) # violin plot
plt.title("Распределение сумм чеков: с бонусами vs без бонусов")
plt.show()

## Гипотеза 2
sales_clean = sales[(sales["discount_pct"] >= 0) & (sales["discount_pct"] <= 100)]
corr_coef, p_value = pearsonr(sales_clean["discount_pct"], sales_clean["dr_kol"]) # корреляция

print("\n📊 Гипотеза 2: Чем больше скидка, тем выше объем продаж")
print(f"Коэффициент корреляции: {corr_coef:.2f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("✅ Гипотеза подтверждена: скидки действительно увеличивают продажи")
else:
    print("n❌ Гипотеза НЕ подтверждена: скидки не оказывают значимого влияния на продажи")

### Визуализация
plt.figure(figsize=(10, 5))
sns.regplot(x="discount_pct", y="dr_kol", data=sales_clean, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.xlabel("Скидка, %", fontsize=8)
plt.ylabel("Количество продаж", fontsize=8)
plt.title("Взаимосвязь скидки и количества продаж", fontsize=10)

plt.text(2, sales_clean["dr_kol"].max()*0.9, f"Корреляция: {corr_coef:.2f}", fontsize=12, bbox=dict(facecolor="white", alpha=0.8))
plt.show()

# Гипотеза 3
shop_sales = sales_shops.groupby("name")["total_sales"].apply(list)
stat, p_normality = kruskal(*shop_sales) # критерий Крускала-Уоллиса
print("\n📊 Гипотеза 3: Медианная сумма продаж отличается в разных аптеках")
print(f"Kruskal-Wallis статистика: {stat:.2f}")
print(f"p-value: {p_normality:.4f}")
if p_normality < 0.05:
    print("✅ Гипотеза подтверждена: медианные суммы продаж в разных аптеках статистически отличаются")
else:
    print("❌ Гипотеза НЕ подтверждена: различия в медианных суммах продаж незначительны")

shop_sales_df = sales_shops[["name", "total_sales"]].copy() # dataframe

### Визуализация
plt.figure(figsize=(12, 6))
sns.boxplot(x="name", y="total_sales", data=shop_sales_df)
plt.xticks(fontsize=8)
plt.xlabel("Аптека", fontsize=8)
plt.ylabel("Сумма продаж", fontsize=8)
plt.title("Распределение сумм продаж по аптекам", fontsize=10)
plt.show()

# Гипотеза 4
sales_clean = sales[(sales["dr_croz"] > 0) & (sales["dr_kol"] > 0)]
corr_coef, p_value = pearsonr(sales_clean["dr_croz"], sales_clean["dr_kol"])

print("\n📊 Гипотеза 4: Дорогие товары покупают реже, чем дешевые")
print(f"Корреляция цены и количества продаж: {corr_coef:.2f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("✅ Гипотеза подтверждена: дорогие товары покупают реже")
else:
    print("❌ Гипотеза НЕ подтверждена: цена не влияет на количество продаж")

### Визуализация
plt.figure(figsize=(10, 5))
sns.regplot(x="dr_croz", y="dr_kol", data=sales_clean, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.xlabel("Цена товара (руб.)", fontsize=8)
plt.ylabel("Количество продаж", fontsize=8)
plt.title("Зависимость цены и количества продаж", fontsize=10)

plt.text(sales_clean["dr_croz"].min(), sales_clean["dr_kol"].max()*0.9, f"Корреляция: {corr_coef:.2f}", fontsize=12, bbox=dict(facecolor="white", alpha=0.8))
plt.show()

median_price = sales_clean["dr_croz"].median() # группируем товары на "дорогие" и "дешевые" по медианной цене
sales_clean["price_category"] = np.where(sales_clean["dr_croz"] >= median_price, "Дорогие", "Дешевые")

### Визуализация(разброс количества продаж по категориям цены)
plt.figure(figsize=(8, 5))
sns.boxplot(x="price_category", y="dr_kol", data=sales_clean, palette=["red", "blue"])
plt.xlabel("Категория товара", fontsize=8)
plt.ylabel("Количество продаж", fontsize=8)
plt.title("Распределение количества продаж для дорогих и дешевых товаров", fontsize=10)
plt.show()

# Гипотеза 5
sales_clean = sales.dropna(subset=["total_sales", "dr_vzak"])
sales_clean = sales_clean[sales_clean["total_sales"] > 0]

online_sales = sales_clean[sales_clean["dr_vzak"] == 2]["total_sales"] # разделяем данные на обычные покупки и интернет-заказы
offline_sales = sales_clean[sales_clean["dr_vzak"] == 1]["total_sales"]

t_stat, p_value = ttest_ind(online_sales, offline_sales, equal_var=False)
print("\n📊 Гипотеза 5: Чеки онлайн-заказов (интернет-заказы) в среднем выше, чем оффлайн")
print(f"T-статистика: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Гипотеза подтверждена: средний чек интернет-заказов статистически выше")
else:
    print("❌ Гипотеза НЕ подтверждена: разница между чеками незначительна")

sales_clean["order_type"] = sales_clean["dr_vzak"].map({1: "Обычный заказ", 2: "Интернет-заказ"})

### Визуализация
plt.figure(figsize=(8, 5))
sns.boxplot(x="order_type", y="total_sales", data=sales_clean, palette=["gray", "orange"])
plt.xlabel("Тип заказа", fontsize=8)
plt.ylabel("Сумма чека", fontsize=8)
plt.title("Распределение сумм чеков для обычных и интернет-заказов", fontsize=10)
plt.show()

### Визуализация
plt.figure(figsize=(8, 5))
sns.violinplot(x="order_type", y="total_sales", data=sales_clean, palette=["gray", "orange"])
plt.xlabel("Тип заказа", fontsize=8)
plt.ylabel("Сумма чека", fontsize=8)
plt.title("Плотность распределения сумм чеков", fontsize=10)
plt.show()