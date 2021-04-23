def parse_fraction(num_string):
    parts = num_string.split('/')
    if len(parts) == 1:
        return float(parts[0])
    return float(parts[0]) / float(parts[1])


if __name__ == '__main__':
    filepath = "tableaus/rk4_tableau.txt"
    f = open(filepath, "r")
    tableau = []
    for line in f:
        numbers = [parse_fraction(x) for x in line.split()]
        tableau.append(numbers)
    f.close()
    print(tableau)