import cmath
import csv
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate


class Scheme:
    def __init__(self) -> None:
        self.nodes = []
        self.lines = []
        self.order = []
        self.load_losses = {}

    """
    Выполняет обратный шаг расчета потоков мощности в узлах электрической сети.
    """
    def backward(self):
        for n in self.nodes:
            n._S = complex(n.P, n.Q)
            for l in n._inc:
                if l.KTR == 0:
                    n._S += complex(l.G, l.B)/2000000*abs(n.V) * abs(n.V)
                elif l.ip == n.ny:
                    n._S += complex(l.G, l.B)/1000000*abs(n.V) * abs(n.V)
        for l in reversed(self.order):
            Z = complex(l.R, l.X)
            if l.KTR != 0:
                Z *= l.KTR*l.KTR
            S = abs(l._iq._S)
            U = abs(l._iq.V)
            dS = Z*((S*S)/(U*U))  # нагрузочные потери в линии
            self.load_losses[l.iq] = dS
            l._S = l._iq._S+dS
            l._ip._S += l._S

    """
    Выполняет прямой шаг расчета напряжений в электрической сети.

    Данный метод проходит по упорядоченному списку линий,
        рассчитывая новое значение напряжения для конечных узлов каждой линии.
    """
    def forward(self):
        maxdU = 0
        for l in self.order:
            U2 = l._ip.V-(
                (complex(l.R, l.X)*l._S.conjugate())/l._ip.V.conjugate())
            if l.KTR != 0:
                U2 *= l.KTR

            maxdU = max(abs(U2-l._iq.V), maxdU)
            l._iq.V = U2
        return maxdU

    """
    Выполняет двухступенчатый процесс расчета состояния электрической сети.

    Параметры:
        maxIters (int): Максимальное количество итераций процесса коррекции.
        maxDU (float): Максимально допустимое отклонение напряжения между итерациями; если изменение напряжения менее этого значения - расчет считается завершенным.

    Возвращает:
        True, если расчет успешно завершен по достижению нужной точности.
        False, если в графе обнаружены циклы.
    """
    def two_steps(self, maxIters, maxDU):
        self.order.clear()

        for l in self.lines:
            l._marker = False

        for n in self.nodes:
            n._marker = False
            n.V = complex(n.unom, 0)

        stack = []
        for n in self.nodes:
            if n.type is False:
                stack.append(n)
                break

        while len(stack) > 0:
            n = stack.pop()
            if n._marker is True:
                print("Сеть содержит замкнутые контура. Расчет невозможен.")
                return False
            n._marker = True

            for l in n._inc:
                if l._marker is True:
                    continue
                l._marker = True

                if l.iq == n.ny:
                    l.ip, l.iq = l.iq, l.ip
                    l._ip, l._iq = l._iq, l._ip

                self.order.append(l)
                stack.append(l._iq)

        for it in range(maxIters):
            self.backward()
            dU = self.forward()
            print(f"Итерация - {it}, Поправка по напряжению: {dU}")

            if dU < maxDU:
                for n in self.nodes:
                    if n.type is False:
                        n.P = -n._S.real
                        n.Q = -n._S.imag
                print("Расчет режима завершен")
                return True

        return False

    """
    Загружает данные о узлах и линиях из CSV-файлов и создает соответствующие объекты Node и Line.

    Для каждой строки в CSV-файлах создается новый объект Node или Line,
    который добавляется в список узлов или линий объекта.

    Параметры:
        nodes (str): Путь к CSV-файлу с информацией о узлах.
        Ожидается, что файл содержит следующие колонки:
                    - ID узла (int)
                    - Название узла (str)
                    - Тип узла (bool)
                    - Номинальное напряжение (float)
                    - Активная мощность (float)
                    - Реактивная мощность (float)

        lines (str): Путь к CSV-файлу с информацией о линиях.
        Ожидается, что файл содержит следующие колонки:
                    - ID начального узла линии (int)
                    - ID конечного узла линии (int)
                    - Сопротивление линии (float)
                    - Реактивное сопротивление линии (float)
                    - Проводимость линии (float)
                    - Вместимость линии (float)
                    - Коэффициент трансформации линии (float)
    """
    def load_scheme(self, nodes, lines):
        nodes_data = pd.read_csv(nodes, sep=";", encoding="utf-8")
        lines_data = pd.read_csv(lines, sep=";", encoding="utf-8")
        for index, row in nodes_data.iterrows():
            node = Node()
            node.ny = int(row[0])
            node.name = row[1]
            node.type = bool(row[2])
            node.unom = float(row[3])
            node.P = float(row[4])
            node.Q = float(row[5])
            self.nodes.append(node)
        print("Nodes processed and stored.")
        for index, row in lines_data.iterrows():
            line = Line()
            line.ip = int(row[0])
            line.iq = int(row[1])
            line.R = float(row[2])
            line.X = float(row[3])
            line.G = float(row[4])
            line.B = float(row[5])
            line.KTR = float(row[6])
            self.lines.append(line)
        print("Lines processed and stored.")

    def assemble(self):
        for line in self.lines:
            for n in self.nodes:
                if n.ny == line.ip:
                    line._ip = n
                    n._inc.append(line)

                if n.ny == line.iq:
                    line._iq = n
                    n._inc.append(line)

    """
    Сопоставляет линии с соответствующими узлами.

    Данный метод проходит по всем линиям и сопоставляет каждой линии узлы начала и конца (ip и iq) на основе
    идентификаторов узлов (ny).

    Сопоставленные узлы сохраняются как атрибуты линии (_ip для начального и _iq для конечного узла).
    Каждой линии также добавляются в список входящих линий соответствующих узлов, обозначенный как _inc.
    """
    def display(self):
        if self.nodes:
            nodes_data = [
                [node.ny, node.name, node.type, node.unom, node.P, node.Q]
                for node in self.nodes]
            print("Узлы:")
            print(tabulate(
                nodes_data, headers=["NY", "Name", "Type", "Unom", "P", "Q"],
                tablefmt='pretty'))
        else:
            print("Узлы: список пуст")

        if self.lines:
            lines_data = [
                [line.ip, line.iq, line.R, line.X, line.G, line.B, line.KTR]
                for line in self.lines]
            print("\nЛинии:")
            print(tabulate(
                lines_data, headers=["IP", "IQ", "R", "X", "G", "B", "KTR"],
                tablefmt='pretty'))
        else:
            print("Линии: список пуст")

    """
    Отображает результат расчета режима сети
    """
    def display_results(self):
        headers = ["NY", "Name", "Type", "Unom", "P", "Q", "V", "Delta"]
        table_data = []

        for node in self.nodes:
            if isinstance(node.V, complex):
                magnitude = abs(node.V)
                phase_radians = cmath.phase(node.V)
                phase_degrees = (phase_radians * 180) / (cmath.pi)
            else:
                magnitude = 'N/A'
                phase_degrees = 'N/A'

            row = [
                node.ny,
                node.name,
                'Нагрузка' if node.type is True else 'База',
                node.unom,
                format(node.P, '.2f'),
                format(node.Q, '.2f'),
                format(magnitude, '.2f'),
                format(phase_degrees, '.2f')
            ]
            table_data.append(row)

        print("Узлы:")
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    """Метод сохранения результатов в файл."""
    def save_scheme(self):
        if input("Сохранить результат? (y/n): ") == "y":
            filename = input("Введите имя файла(либо нажмите Enter чтобы присвоить имя по умолчанию): ")
            if filename == "":
                filename = "scheme.csv"
            else:
                filename += ".csv"

            node_fields = ['NY', 'name', 'type', 'unom', 'P', 'Q', 'V', "Delta"]
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(node_fields)
                for node in self.nodes:
                    writer.writerow([
                        node.ny, node.name, node.type, node.unom,
                        node.P, node.Q, abs(node.V), (cmath.phase(node.V) * 180) / (cmath.pi)
                    ])
        else:
            print("Программа завершена")

    def diagramm(self):
        line_numbers = sorted(self.load_losses.keys())
        real_parts = [self.load_losses[line].real for line in line_numbers]
        imaginary_parts = [self.load_losses[line].imag for line in line_numbers]
        width = 0.35

        # Индексы для столбцов
        ind = np.arange(len(line_numbers))

        # Создание графика
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(ind - width/2, real_parts, width, label='Действительна часть', color='r')
        rects2 = ax.bar(ind + width/2, imaginary_parts, width, label='Мнимая часть', color='b')

        # Настройка подписей осей и заголовка
        ax.set_xlabel('Номер линии', fontsize=14)
        ax.set_ylabel('Величина потерь', fontsize=14)
        ax.set_title('Нагрузочные потери', fontsize=16)
        ax.set_xticks(ind)
        ax.set_xticklabels(line_numbers)

        # Включение легенды
        ax.legend()

        # Функция для добавления значений над столбцами
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # Смещение надписи на 3 точки вверх
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Добавляем значения
        autolabel(rects1)
        autolabel(rects2)

        # Визуализация графика
        plt.tight_layout()
        plt.grid(True)
        plt.show()


class Node:
    def __init__(self, ny: int = 0, name: str = "",
                 node_type: int = 0, unom: float = 0.0,
                 P: float = 0.0, Q: float = 0.0, V: complex = 0+0j) -> None:
        self.ny: int = ny
        self.name: str = name
        self.type: int = node_type
        self.unom: float = unom
        self.P: float = P
        self.Q: float = Q
        self.V: complex = V
        self._inc: List['Line'] = []
        self._marker: bool = False
        self._S: complex = complex()


class Line:
    def __init__(self, ip: int = 0, iq: int = 0,
                 R: float = 0.0, X: float = 0.0,
                 G: float = 0.0, B: float = 0.0, KTR: float = 0) -> None:
        self.ip: int = ip
        self.iq: int = iq
        self.R: float = R
        self.X: float = X
        self.G: float = G
        self.B: float = B
        self.KTR: float = KTR
        self._ip: Optional[Node] = None
        self._iq: Optional[Node] = None
        self._marker: bool = False
        self._S: complex = complex()


# Создаем экземпляр класса схемы
s = Scheme()
# Подгружаем данные узлов и ветвей
s.load_scheme("nodes.csv", "lines.csv")
# Выстраиваем граф
s.assemble()
# Отображаем схему
s.display()
# Выполняем расчет
s.two_steps(20, 0.00001)
# Отображаем результаты
s.display_results()
# Отображаем диаграмму нагрузочных потерь
s.diagramm()
# Сохраняем схему в файл
s.save_scheme()
# == Profit!
# by EgorSemenov
