import time
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # مكتبة للرسم البياني

# الثوابت
GA_POPSIZE = 2048  # عدد السكان (الكروموسومات)
GA_MAXITER = 1000  # الحد الأقصى لعدد الأجيال
GA_ELITRATE = 0.10  # نسبة الأفراد الأفضل الذين نحتفظ بهم بدون تغيير
GA_MUTATIONRATE = 0.25  # معدل الطفرة
GA_TARGET = "Hello world!"  # الجملة الهدف

# تمثيل الفرد
class GAIndividual:
    def __init__(self, string_val=None):
        self.string = string_val or self.random_string()  # توليد جملة عشوائية إن لم تُعطى واحدة
        self.fitness = 0  # قيمة fitness الأولية

    @staticmethod
    def random_string():
        return ''.join(random.choice(string.printable[:95]) for _ in range(len(GA_TARGET)))  # توليد سلسلة بنفس طول الهدف

# إنشاء السكان
def init_population():
    return [GAIndividual() for _ in range(GA_POPSIZE)]  # إنشاء سكان بعدد GA_POPSIZE

# حساب fitness
def calc_fitness(population):
    for individual in population:
        individual.fitness = sum(abs(ord(individual.string[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))  # كلما قلت الفروق زادت الجودة

# ترتيب حسب الأفضلية
def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)  # الترتيب تصاعديًا حسب fitness

# نسخ الأفضل (Elitism)
def elitism(population, buffer, esize):
    for i in range(esize):
        buffer[i].string = population[i].string  # نسخ الجملة
        buffer[i].fitness = population[i].fitness  # نسخ قيمة fitness

# طفرة (Mutation)
def mutate(individual):
    ipos = random.randint(0, len(GA_TARGET) - 1)  # اختيار موقع عشوائي
    mutated = list(individual.string)
    mutated[ipos] = random.choice(string.printable[:95])  # تغيير الحرف العشوائي
    individual.string = ''.join(mutated)  # إعادة بناء الجملة بعد الطفرة

# التزاوج (Crossover)
def mate(population, buffer):
    esize = int(GA_POPSIZE * GA_ELITRATE)  # عدد الأفراد المحفوظين (elitism)
    elitism(population, buffer, esize)  # نسخ الأفراد الأفضل

    for i in range(esize, GA_POPSIZE):
        i1 = random.randint(0, GA_POPSIZE // 2 - 1)  # اختيار أب
        i2 = random.randint(0, GA_POPSIZE // 2 - 1)  # اختيار أم
        spos = random.randint(0, len(GA_TARGET) - 1)  # نقطة التقاطع
        child_str = population[i1].string[:spos] + population[i2].string[spos:]  # بناء الطفل من الأب والأم
        child = GAIndividual(child_str)

        if random.random() < GA_MUTATIONRATE:  # احتمال إجراء طفرة
            mutate(child)

        buffer[i] = child  # حفظ الطفل في الجيل الجديد

# تشغيل الخوارزمية وتسجيل الاحصائيات والقيم

def genetic_algorithm():
    population = init_population()  # إنشاء الجيل الأول
    buffer = init_population()  # تحضير جيل جديد فارغ
    stats = []  # لتخزين ملخص الإحصائيات لكل جيل
    all_fitness_per_generation = []  # لتخزين fitness لكل الأفراد في كل جيل (لـ boxplot)

    for generation in range(GA_MAXITER):
        start_time = time.time()  # بداية توقيت الجيل

        calc_fitness(population)  # حساب fitness لجميع الأفراد
        sort_by_fitness(population)  # ترتيب السكان

        fitness_values = [ind.fitness for ind in population]  # جمع جميع قيم fitness
        all_fitness_per_generation.append(fitness_values.copy())  # تخزينهم للرسم لاحقًا

        best = fitness_values[0]  # أفضل fitness
        worst = fitness_values[-1]  # أسوأ fitness
        avg = np.mean(fitness_values)  # المتوسط
        std = np.std(fitness_values)  # الانحراف المعياري
        elapsed = time.time() - start_time  # الوقت المستغرق للجيلة

        stats.append({  # تخزين الإحصائيات
            "Generation": generation,
            "Best Fitness": best,
            "Worst Fitness": worst,
            "Average Fitness": avg,
            "Std Dev": std,
            "Elapsed Time (s)": elapsed
        })

        print(f"Gen {generation}: Best={best}, Avg={avg:.2f}, Worst={worst}, Std={std:.2f}")  # طباعة الإحصائيات
        print(f"Best string: {population[0].string}")  # طباعة الجملة الأفضل

        if best == 0:  # إذا وصلنا للحل المثالي نوقف
            break

        mate(population, buffer)  # إنشاء جيل جديد
        population, buffer = buffer, population  # تبديل السكان

    return pd.DataFrame(stats), all_fitness_per_generation  # إرجاع الإحصائيات وقيم fitness الكاملة

# تشغيل الخوارزمية
stats_df, fitness_data = genetic_algorithm()  # stats_df فيها إحصائيات، fitness_data فيها القيم الكاملة لكل جيل

# رسم الجراف للبند 3a
plt.figure(figsize=(12, 6))
plt.plot(stats_df['Generation'], stats_df['Best Fitness'], label='Best Fitness', linewidth=2)  # رسم أفضل fitness
plt.plot(stats_df['Generation'], stats_df['Average Fitness'], label='Average Fitness', linestyle='--')  # رسم المتوسط
plt.plot(stats_df['Generation'], stats_df['Worst Fitness'], label='Worst Fitness', linestyle=':')  # رسم الأسوأ
plt.title('Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# رسم boxplot للبند 3b
plt.figure(figsize=(14, 6))
plt.boxplot(fitness_data, vert=True, patch_artist=True, showfliers=True)  # رسم Boxplot لكل جيل بناء على قيم fitness
plt.title("Boxplot of Fitness per Generation")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()
