import multiprocessing
from timeit import default_timer as timer

def WriteToCSV(process,time):
    file=open("report.csv",'a')
    file.write(str(process))
    file.write(';')
    file.write(str(time))
    file.write(';\n')
    file.close()

def math_function(x):
    return (1 / x) - (1 / x ** 2) - (1 / x ** 3)


def calc_area_part(a, h, start, stop):
    return sum([math_function(a + i*h) for i in range(start, stop)])


def calc_area(a, b, n, process_count):
    h = (b - a) / n
    step = n // process_count

    segment_range = [(step * i, step * (i + 1)) for i in range(process_count)]
    print(segment_range)
    print(len(segment_range))
    segment_data = [(a + h / 2, h, segment[0], segment[1])
                    for segment in segment_range]

    def parrallel_calculation():
        pool = multiprocessing.Pool(process_count)
        area_part_sum = sum(pool.starmap(calc_area_part, segment_data))
        pool.close()

        return area_part_sum

    start = timer()

    area_part_sum = parrallel_calculation()

    end = timer()

    print(f'{process_count} process time {end - start}')
    WriteToCSV(process_count,end-start)
    area = h * area_part_sum

    return area


if __name__ == '__main__':
    print("BE STARTED")

    file = open("report.csv", 'w')
    file.close()

    for t in range(7,3,-1):
        print("t=",t)

        file = open("report.csv", 'a')

        file.write(str(t))
        file.write( " степень n ")
        file.write(';\n')
        file.close()


        for n in range(1,8):
            #area = calc_area(-10, 10, int(1e8), 8)
            area = calc_area(-10, 10, int(10**t), n)
            print(area)
