import itertools
import random
import time

import tsplib95
import numpy as np


def held_karp(dists):
    """
    Implementacja algorytmu Helda-Karpa

    Parametry:
        dists: macierz krawędzi

    Return:
        (koszt, ścieżka).
    """
    n = len(dists)

    # Mapuje każdy podzbiór węzłów na koszt osiągnięcia tego podzbioru oraz na
    # numer węzła, który został odwiedzony przed osiągnięciem tego podzbioru.
    # Podzbiory węzłów są reprezentowane jako bity zbioru.
    C = {}

    # Ustawienie kosztu przejścia ze stanu początkowego
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)
        # print(C)

    # Iteracja po podzbiorach rosnącej wielkości i przechowywanie wyników pośrednich
    for subset_size in range(2, n):
        # Dla każdej kombinacji podzbioru o danym rozmiarze
        for subset in itertools.combinations(range(1, n), subset_size):

            # Ustala bity dla wszystkich węzłów w tym podzbiorze
            # print(f"podzbiór = {subset}")
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            # print(f"bity = {bits}")

            # Znajduje najniższy koszt dotarcia do tego podzbioru
            for k in subset:
                prev = bits & ~(1 << k)
                # print(f"poprzedni = {prev}")
                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                # print(f"wynik = {res}")
                C[(bits, k)] = min(res)
                # print(f"C = {C}")

    # Interesują nas wszystkie bity oprócz najmniej znaczącego (stan początkowy)
    bits = (2 ** n - 1) - 1
    # print(f"bits = {bits}")

    # Oblicza optymalny koszt powrotu do stanu początkowego
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Wspina się, aby znaleźć pełną ścieżkę
    path = [0]
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Dodaje stan początkowy na koniec
    path.append(0)

    return opt, list(reversed(path))


def generate_distances(n):
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dists[i][j] = dists[j][i] = random.randint(1, 99)

    return dists


def read_distances_txt(filename):
    dists = []
    with open(f"problems/{filename}") as f:
        size = int(f.readline())
        for line in f.readlines():
            dists.append(line.split())

    dists = [[int(x) for x in line] for line in dists]

    return dists


def read_distances_tsp(filename):
    problem = tsplib95.load(f"problems/{filename}")
    # if problem.edge_weight_type == "EUC_2D":
    #     num_nodes = problem.dimension
    #     coordinates = problem.node_coords
    #     dists = np.zeros((num_nodes, num_nodes))
    #     print(f"num_nodes = {num_nodes}")
    #     print(f"coordinates = {coordinates}")
    #     print(f"dists = {dists}")
    #     for i in range(num_nodes):
    #         x1, y1 = coordinates[i + 1]
    #         for j in range(num_nodes):
    #             if i != j:
    #                 x2, y2 = coordinates[j + 1]
    #                 dists[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    if problem.edge_weight_type == "EXPLICIT":
        num_nodes = problem.dimension
        dists = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                dists[i][j] = problem.get_weight(i, j)
    return dists


def test():
    with open("program.INI", "r") as config_file:
        config_data = config_file.read().splitlines()

    with open(config_data[-1], "w") as result_file:
        for line in config_data:
            if line[0] == ';':
                continue
            line = line.split(" ", 3)
            if len(line) > 1:
                result_file.write(f"{line[0]};{line[1]};{line[2]};{line[3]}\n")
                print(f"{line[0]};{line[1]};{line[2]};{line[3]}\n")
                if line[0][-3:] == "txt":
                    dists = read_distances_txt(line[0])
                elif line[0][-3:] == "tsp":
                    dists = read_distances_tsp(line[0])
                repeat = int(line[1])
                for i in range(repeat):
                    start_time = time.perf_counter()
                    min_cost, shortest_path = held_karp(dists)
                    end_time = time.perf_counter()
                    path = " ".join(str(x) for x in shortest_path)
                    result_file.write(f"{(end_time - start_time)};{min_cost};[{path}]\n")
                    print(f"{(end_time - start_time)};{min_cost};[{path}]\n")


if __name__ == '__main__':
    test()
