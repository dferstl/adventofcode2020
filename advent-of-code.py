
# %% Day 1 - part1 two-sum

with open("input/day1-1-input.txt", "r") as fp:
    numbers = fp.readlines()

numbers = [int(number.strip()) for number in numbers]

target_number = 2020

numbers_minus = [target_number - number for number in numbers]

num_dict_minus = {target_number - number: number for number in numbers}

number_combinations = []
for number in numbers:
    if number in num_dict_minus:
        number_combinations.append((number, num_dict_minus[number]))

print("Twosum result O(N):")
for number_combination in number_combinations:
    print("{} + {} = {} -> {}".format(number_combination[0], number_combination[1], number_combination[0] + number_combination[1], number_combination[0] * number_combination[1]))

# %% Day1 - part2 three-sum

number_combinations = []
for n2 in numbers:
    for n3 in numbers:
        number = n2 + n3
        if number in num_dict_minus:
            number_combinations.append((n2, n3, num_dict_minus[number]))

print("Threesum result O(N^2):")
for ncs in number_combinations:
    print("{} + {} + {} = {} -> {}".format(ncs[0], ncs[1], ncs[2], ncs[0] + ncs[1] + ncs[2], ncs[0] * ncs[1] * ncs[2]))


# %% Day2 

with open("input/day2-1-input.txt", "r") as fp:
    passwords = fp.readlines()

passwords = [password.strip().split() for password in passwords]
passwords = [[pw[1][0], int(pw[0].split('-')[0]), int(pw[0].split('-')[1]), pw[2]] for pw in passwords]


def cout_occurences(str, char):
    num_occ = 0
    for c in str:
        if c == char:
            num_occ += 1

    return num_occ

num_valid = 0
for pw in passwords:
    num_occ = cout_occurences(pw[3], pw[0])
    if num_occ >= pw[1] and num_occ <= pw[2]:
        num_valid += 1

print("{} out of {} passwords are valid".format(num_valid, len(passwords)))

num_valid = 0
for letter, pos1, pos2, pw in passwords:
    pos_val = int(pw[pos1 - 1] == letter) + int(pw[pos2 - 1] == letter)
    if pos_val == 1:
        num_valid += 1

print("{} out of {} passwords are valid".format(num_valid, len(passwords)))


# %% Day3

with open("input/day3-1-input.txt", "r") as fp:
    forest = fp.readlines()

forest = [line.strip() for line in forest]

num_trees_list = []
tile_length = len(forest[0])
for r,d in [(1,1), (3,1), (5,1), (7,1), (1,2)]:
    num_trees = 0
    x = 0
    for l in forest[::d]:
        x %= tile_length
        if l[x] == '#':
            num_trees += 1
        x += r
    num_trees_list.append(num_trees)

    print("Traversing the forest with ({},{}) moves encountered {} trees".format(r, d, num_trees))

mul_results = 1
for nt in num_trees_list:
    mul_results *= nt

print("Multiplied result is {}".format(mul_results))


# %% Day4

with open("input/day4-1-input.txt", "r") as fp:
    passport_lines = fp.readlines()

reqired_fields = ["byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"] # "cid"
# byr (Birth Year) - four digits; at least 1920 and at most 2002.
# iyr (Issue Year) - four digits; at least 2010 and at most 2020.
# eyr (Expiration Year) - four digits; at least 2020 and at most 2030.
# hgt (Height) - a number followed by either cm or in:
# If cm, the number must be at least 150 and at most 193.
# If in, the number must be at least 59 and at most 76.
# hcl (Hair Color) - a # followed by exactly six characters 0-9 or a-f.
# ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
# pid (Passport ID) - a nine-digit number, including leading zeroes.
# cid (Country ID) - ignored, missing or not.



passports = []
passport = {}
for passportline in passport_lines:
    passportline = passportline.strip()
    if passportline == "" and passport:
        passports.append(passport)
        passport = {}
    else:
        passportline_tuples = [ple.split(":") for ple in passportline.split(" ")]
        passport.update( {value[0]: value[1] for value in passportline_tuples} )

def check_year(str, miny, maxy):
    if len(str) == 4 and str.isdigit():
        if int(str) >= miny and int(str) <= maxy:
            return True

    return False
    
def check_height(str):
    if len(str) > 2 and str[:-2].isdigit():
        str_int = int(str[:-2])
        if (str.endswith("cm") and str_int >= 150 and str_int <= 193) or (str.endswith("in") and str_int >= 59 and str_int <= 76):
            return True
    return False

def check_hcl(str):
    if len(str) == 7 and str[0] == "#" and re.match("^[a-f0-9]+$", str[1:]):
        return True
    return False

def check_ecl(str):
    # ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
    if str in ["amb", "blu", "brn", "gry", "grn", "hzl", "oth"]:
        return True
    return False

def check_pid(pid):
    # pid (Passport ID) - a nine-digit number, including leading zeroes.
    if len(pid) == 9 and pid.isdigit():
        return True
    return False


num_valid_passports = 0
for passport in passports:
    if all(k in passport for k in reqired_fields):
        valid = True
        for k,v in passport.items():
            if k == "byr":
                if not check_year(v, 1920, 2002):
                    valid = False
                    break
            elif k == "iyr":
                if not check_year(v, 2010, 2020):
                    valid = False
                    break
            elif k == "eyr":
                if not check_year(v, 2020, 2030):
                    valid = False
                    break
            elif k == "hgt":
                if not check_height(v):
                    valid = False
                    break
            elif k == "hcl":
                if not check_hcl(v):
                    valid = False
                    break
            elif k == "ecl":
                if not check_ecl(v):
                    valid = False
                    break
            elif k == "pid":
                if not check_pid(v):
                    valid = False
                    break
        
        if valid:
            num_valid_passports += 1

print("{} of {} passports valid".format(num_valid_passports, len(passports)))






# %%

with open("input/day5-1-input.txt", "r") as fp:
    seats = fp.readlines()

seats = [seat.strip() for seat in seats]

def get_row(str):
    r_max = 128
    row = 0
    for c in str:
        r_max //= 2
        if c == "B":
            row += r_max

    return row

def get_col(str):
    c_max = 8
    col = 0
    for c in str:
        c_max //= 2
        if c == "R":
            col += c_max

    return col

ids = []
for seat in seats:
    row = get_row(seat[:7])
    col = get_col(seat[-3:])
    ids.append(row*8 + col)

ids.sort()

print("highest id is {}".format(ids[-1]))

my_seat = -1
for i in range(ids[-1]):
    if i > 0 and i < len(ids)-1:
        if not i in ids and i-1 in ids and i+1 in ids:
            my_seat = i
            break

print("my seat is {}".format(my_seat))





# %% Day 6

with open("input/day6-1-input.txt", "r") as fp:
    answers = fp.readlines()
answers = [answer.strip() for answer in answers]

groups = []
group = []
for l in answers:
    if l == "":
        groups.append(group)
        group = []
    else:
        group.append(set(l))

if group:
    groups.append(group)


answers = []
total_sum = 0
for group in groups:
    answer = group[0]
    for i in range(1, len(group)):
        answer = answer.intersection(group[i])
    #answer = set().intersection(*group)
    total_sum += len(list(answer))
    answers.append(answer)

print("total sum of answers is {}".format(total_sum))
    




# %% Day 7
import collections

with open("input/day7-1-input.txt", "r") as fp:
    bag_rules = fp.readlines()
bag_rules = [bag_rule.strip() for bag_rule in bag_rules]

bag_colors = set()
for bag_rule in bag_rules:
    bag_color = bag_rule.split(" ")[0:2]
    bag_colors.add(" ".join(bag_color))

rule_graph = {}
inverse_graph = collections.defaultdict(list)

def parse_rule(bag_rule):
    rule_content = bag_rule.split(" ")
    bag_color = " ".join(rule_content[:2])
    content_str = rule_content[4:]
    num_content = len(content_str) // 4
    content = []
    for i in range(num_content): 
        content.append((" ".join(content_str[4*i+1: 4*i+3]), int(content_str[4*i])))
    return bag_color, content

inverse_rule_graph = collections.defaultdict(list)
rule_graph = {}
for bag_rule in bag_rules:
    bag_color, content = parse_rule(bag_rule)
    rule_graph[bag_color] = content
    for k in content:
        inverse_rule_graph[k[0]].append(bag_color)

# %%
my_bag = "shiny gold"
how_many = 0

queue = [inverse_rule_graph[my_bag]]
visited = set()
# visited.add(my_bag)

while queue:
    s = queue.pop(0)
    for color in s:
        if color not in visited:
            visited.add(color)
            queue.append(inverse_rule_graph[color])

how_many = len(visited)
print("{} bags contain at least one {} bag".format(how_many, my_bag))


#%%

def collect_numbers(rule_graph, queue):
    if not queue:
        return 0

    num_bags = 0
    for bag in queue:
        number = collect_numbers(rule_graph, rule_graph[bag[0]])
        num_bags += bag[1] + bag[1] * number

    return num_bags

how_many = collect_numbers(rule_graph, rule_graph[my_bag])

# How many individual bags are required inside your single shiny gold bag?
print("{} bag contains {} individual bags".format(my_bag, how_many))

# %% Day 8

import copy

with open("input/day8-1-input.txt", "r") as fp:
    commands = fp.readlines()
commands = [command.strip().split(' ') for command in commands]
# print(commands)
visited = [False] * len(commands)

idx = 0
accumulator = 0
jmp_nop_positions = []
while visited[idx] == False:
    visited[idx] = True
    if commands[idx][0] == "nop":
        jmp_nop_positions.append(idx)
        idx += 1
    elif commands[idx][0] == "acc":
        accumulator += int(commands[idx][1])
        idx += 1
    elif commands[idx][0] == "jmp":
        jmp_nop_positions.append(idx)
        idx += int(commands[idx][1])

print("The value of the accumulator at the loop is {}".format(accumulator))
exchange_at_pos = 0
while idx < len(commands) - 1:
    commands_new = copy.deepcopy(commands)
    exchange_idx = jmp_nop_positions.pop(0)
    if commands_new[exchange_idx][0] == "nop":
        print("exchange {} to {} at {}".format(commands_new[exchange_idx][0], "jmp", exchange_idx))
        commands_new[exchange_idx][0] = "jmp"
    elif commands_new[exchange_idx][0] == "jmp":
        print("exchange {} to {} at {}".format(commands_new[exchange_idx][0], "nop", exchange_idx))
        commands_new[exchange_idx][0] = "nop"
    else:
        print("ERROR")

    idx = 0
    accumulator = 0
    visited = [False] * len(commands)
    while visited[idx] == False:
        visited[idx] = True
        if commands_new[idx][0] == "nop":
            idx += 1
        elif commands_new[idx][0] == "acc":
            accumulator += int(commands_new[idx][1])
            # print(commands[idx])
            idx += 1
        elif commands_new[idx][0] == "jmp":
            idx += int(commands_new[idx][1])
        if idx > len(commands) - 1:
            print("End reached")
            break

print("The value of the accumulator at the loop is {}".format(accumulator))


# %% Day9

with open("input/day9-1-input-test.txt", "r") as fp:
    numbers = fp.readlines()
numbers = [int(command.strip()) for command in numbers]

pream_length = 5
last_nums = []
for i in range(pream_length, len(numbers)):
    curr_sum = numbers[i]
    # last_sums = [ curr_sum - num for num in numbers[i-pream_length:i] ]

    found = False
    for j in range(pream_length):
        for k in range(pream_length):
            if k != j and numbers[i-pream_length+j] + numbers[i-pream_length+k] == curr_sum:
                found = True
                break

    if found == False:
        break
        
print("Broken number is {}".format(curr_sum))

target = curr_sum
for i in range(0, len(numbers)):
    sum_ = numbers[i]
    for j in range(i+1, len(numbers)):
        sum_ += numbers[j]
        if sum_ > target:
            break
        if sum_ == target:
            print("Sum found between {} and {} which sum to {}".format(
                i,j,
                sum(numbers[i:j+1])))
            print("Min {}, Max {}, Min+Max = {}".format(
                min(numbers[i:j+1]),
                max(numbers[i:j+1]),
                min(numbers[i:j+1]) + max(numbers[i:j+1])
            ))

# linear hash solution
cumsum = [numbers[0]]
for i in range(1, len(numbers)):
    cumsum.append(cumsum[-1] + numbers[i])

cumsum_target = [cs - target for cs in cumsum]
cumsum_dict = {num: idx for idx,num in enumerate(cumsum)}

for j, cst in enumerate(cumsum_target):
    if cst in cumsum_dict:
        i = cumsum_dict[cst]
        print("Sum found between {} and {} which sum to {}".format(
            i,j,
            sum(numbers[i:j+1])))
        print("Min {}, Max {}, Min+Max = {}".format(
            min(numbers[i:j+1]),
            max(numbers[i:j+1]),
            min(numbers[i:j+1]) + max(numbers[i:j+1])
        ))
        break

# Two pointer solution
i = 0
j = 1
cumsum = [numbers[0]]
for i in range(1, len(numbers)):
    cumsum.append(cumsum[-1] + numbers[i])
while j < len(cumsum):
    if cumsum[j] - cumsum[i] < target:
        j += 1
    elif cumsum[j] - cumsum[i] > target:
        i += 1
    elif cumsum[j] - cumsum[i] == target:
        print("Sum found between {} and {} which sum to {}".format(
            i,j,
            sum(numbers[i:j+1])))
        print("Min {}, Max {}, Min+Max = {}".format(
            min(numbers[i:j+1]),
            max(numbers[i:j+1]),
            min(numbers[i:j+1]) + max(numbers[i:j+1])
        ))
        break


# %% Day10

with open("input/day10-1-input.txt", "r") as fp:
    jolts = fp.readlines()
jolts = [int(command.strip()) for command in jolts]

jolts.sort()

diffs = {1:0, 2:0, 3:0}
prev = 0
for j in jolts:
    diffs[j-prev] += 1
    prev = j

diffs[3] += 1

print("PART1:")
print(diffs)
print("{} 1jolt diffs and {} 3jolt diffs multiplied is {}".format(
    diffs[1], diffs[3], diffs[1]*diffs[3]
))


jolts = [0] + jolts + [jolts[-1] + 3]
num_solutions_to_end = [0] * len(jolts)
for idx in range(len(jolts)-1, -1, -1):
    if idx == len(jolts)-1:  # num solutions from last element = 1
        num_solutions_to_end[idx] = 1
    else: # for every element sum up solutions to the last element
        sum_solutions = 0
        i = idx + 1
        while i < len(jolts) and jolts[i] - jolts[idx] <= 3:
            sum_solutions += num_solutions_to_end[i]
            i += 1
        num_solutions_to_end[idx] = sum_solutions
        
num_solutions = num_solutions_to_end[0]

print("\nPART2:")
print("{} of distinct ways exist to connect the charging outlet to your device".format(num_solutions) )


#%% Day11

import copy

with open("input/day11-1-input.txt", "r") as fp:
    seats = fp.readlines()
seats = [command.strip() for command in seats]
seats = [[seat for seat in row] for row in seats]


directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

def num_seats_occupied(seats, x, y):
    W = len(seats[0])
    H = len(seats)
    num_nbh_seats_occupied = 0
    for d in directions:
        xn = x + d[0]
        yn = y + d[1]
        while xn < W and xn >= 0 and yn < H and yn >= 0:
            if seats[yn][xn] == "#":
                num_nbh_seats_occupied += 1
                break
            elif seats[yn][xn] == "L":
                break
            xn += d[0]
            yn += d[1]
    
    return num_nbh_seats_occupied

def get_tot_num_seats_occupied(seats):
    W = len(seats[0])
    H = len(seats)
    num_seats_occupied = 0
    for x in range(W):
        for y in range(H):
            if seats[y][x] == "#":
                num_seats_occupied += 1

    return num_seats_occupied

def fill_seats(seats):
    new_seats = copy.deepcopy(seats)
    W = len(seats[0])
    H = len(seats)
    for x in range(W):
        for y in range(H):
            if seats[y][x] == ".":
                continue
            if seats[y][x] == "L" and num_seats_occupied(seats, x, y) == 0:
                new_seats[y][x] = "#"
            elif seats[y][x] == "#" and num_seats_occupied(seats, x, y) >= 5:
                new_seats[y][x] = "L"
    return new_seats

def print_seats(seats):
    print("")
    for y in range(len(seats)):
        print("".join(seats[y]))

change = True
num_runs = 0
while change == True:
    new_seats = fill_seats(seats)
    if new_seats == seats:
        print("we don't have any changes")
        change = False
    seats = new_seats
    num_runs += 1

print("Number of occupied seats is {}".format(get_tot_num_seats_occupied(seats)))


# %% Day 12

with open("input/day12-1-input.txt", "r") as fp:
    moves = fp.readlines()
moves = [command.strip() for command in moves]
moves = [(s[0], int(s[1:])) for s in moves]


#
#            (90)
#             N
#             |
# (180) W  -  0  -  E (0)
#             |
#             S
#            (270)
#
#           E, N
position = [0, 0]
dir_deg = 0

for move in moves:
    if move[0] == "F":
        if dir_deg == 0:
            position[0] += move[1]
        elif dir_deg == 90:
            position[1] += move[1]
        elif dir_deg == 180:
            position[0] -= move[1]
        elif dir_deg == 270:
            position[1] -= move[1]
    elif move[0] == "N":
        position[1] += move[1]
    elif move[0] == "S":
        position[1] -= move[1] 
    elif move[0] == "E":
        position[0] += move[1] 
    elif move[0] == "W":
        position[0] -= move[1]
    elif move[0] == "L":
        dir_deg = (dir_deg + move[1])%360
    elif move[0] == "R":
        dir_deg = (dir_deg - move[1])%360

manhattan_dist = abs(position[0]) + abs(position[1])

print("Part1: The ships manhattan distance is {}".format(manhattan_dist))
#           E   N
waypoint = [10, 1]
position = [0,  0]
dir_deg = 0

for move in moves:
    if move[0] == "F":
        position[0] += waypoint[0] * move[1]
        position[1] += waypoint[1] * move[1]
    elif move[0] == "N":
        waypoint[1] += move[1]
    elif move[0] == "S":
        waypoint[1] -= move[1]
    elif move[0] == "E":
        waypoint[0] += move[1]
    elif move[0] == "W":
        waypoint[0] -= move[1]
    elif move[0] == "L":
        if move[1] == 270:
            waypoint[0], waypoint[1] = waypoint[1], -waypoint[0]
        elif move[1] == 180:
            waypoint[0], waypoint[1] = -waypoint[0], -waypoint[1]
        elif move[1] == 90:
            waypoint[0], waypoint[1] = -waypoint[1], waypoint[0]

    elif move[0] == "R":
        if move[1] == 90:
            waypoint[0], waypoint[1] = waypoint[1], -waypoint[0]
        elif move[1] == 180:
            waypoint[0], waypoint[1] = -waypoint[0], -waypoint[1]
        elif move[1] == 270:
            waypoint[0], waypoint[1] = -waypoint[1], waypoint[0]        

    # print("Command {}/{}".format(move[0], move[1]))
    # print("Position E{} / N{}".format(position[0], position[1]))
    # print("Waypoint E{} / N{}".format(waypoint[0], waypoint[1]))


manhattan_dist = abs(position[0]) + abs(position[1])

print("Part2: The ships manhattan distance is {}".format(manhattan_dist))

# %% Day 13

import math

with open("input/day13-1-input-test.txt", "r") as fp:
    lines = fp.readlines()
lines = [command.strip() for command in lines]

timestamp_start = int(lines[0])
bus_ids = [int(s) for s in lines[1].split(',') if s.isdigit()]

time_diff = []
for bus_id in bus_ids:
    div = float(timestamp_start) / float(bus_id)
    time_diff.append(int( bus_id * math.ceil(div) )  - timestamp_start )

bus_id = time_diff.index(min(time_diff))

print("Part1:")
print("BusID {} - waiting time {}".format(bus_ids[bus_id], time_diff[bus_id]))
print("Result is {}".format(bus_ids[bus_id]*time_diff[bus_id]))


from toolz import reduce, last

busses = [(i, int(s)) for i,s in enumerate(lines[1].split(',')) if s.isdigit()]
N = reduce(lambda a, b: a * b, map(last, busses))  # Product of all IDs (or moduli)
common_divisor = sum((m - r) * N // m * pow(N // m, -1, m) for r, m in busses) % N

print("Part 2:", common_divisor)
# %% Day 14

# input either
# * update the bitmask
# * write a value to memory
# values and memory are both 36bit
# bitmask is given in 36 bit string from most to least significant bit
# 2^36 ... 2^0
# 

from timer import Timer

with open("input/day14-input-test.txt", "r") as fp:
    lines = fp.readlines()
lines = [command.strip() for command in lines]

def mask_value_p1(value, mask, bitlength=36):
    binary_value = [c for c in "{0:b}".format(value)]
    binary_value = ['0' for _ in range(bitlength - len(binary_value))] + binary_value
    for i in range(bitlength):
        if mask[-i] == "X":
            continue
        elif mask[-i] == "0":
            binary_value[-i] = "0"
        elif mask[-i] == "1":
            binary_value[-i] = "1"

    return int("".join(binary_value), 2)

with Timer():
    memory = {}
    for line in lines:
        if line.startswith("mask"):
            mask = line.split(" = ")[1]
        elif line.startswith("mem"):
            value = int(line.split(" = ")[1])
            position = int(line.split("[")[1].split("]")[0])
            value_masked = mask_value_p1(value, mask)
            memory[position] = value_masked

print("Part1: Sum of memory is {}".format(sum(memory.values())))

def mask_value_p2(value, mask, bitlength=36):
    binary_value = [c for c in "{0:b}".format(value)]
    binary_value = ['0' for _ in range(bitlength - len(binary_value))] + binary_value
    for i in range(bitlength):
        if mask[-i] == "0":
            continue
        elif mask[-i] == "X":
            binary_value[-i] = "0"
        elif mask[-i] == "1":
            binary_value[-i] = "1"

    starting_value = int("".join(binary_value), 2)

    x_bits = [pow(2,i) for i,c in enumerate(mask[::-1]) if c == "X"]
    values = [0]
    for x_bit in x_bits:
        values.extend([v + x_bit for v in values ])
    values = [v + starting_value for v in values]

    return values

with Timer():
    memory = {}
    for line in lines:
        if line.startswith("mask"):
            mask = line.split(" = ")[1]
        elif line.startswith("mem"):
            value = int(line.split(" = ")[1])
            position = int(line.split("[")[1].split("]")[0])
            positions_masked = mask_value_p2(position, mask)
            for position_masked in positions_masked:
                memory[position_masked] = value

print("Part2: Sum of memory is {}".format(sum(memory.values())))



# %% Day 15

from timer import Timer

with open("input/day15-input-test.txt", "r") as fp:
    lines = fp.readlines()
lines = [command.strip() for command in lines]

max_turn = 30000000

for line in lines:
    with Timer():
        starting_values = line.split(',')
        spoken_dict = {int(starting_value): i for i, starting_value in enumerate(starting_values[:-1])}

        last_number = int(starting_values[-1])

        idx = len(starting_values) - 1

        while idx < max_turn - 1:
            if last_number not in spoken_dict:
                spoken_dict[last_number] = idx
                new_number = 0
            else:
                new_number = idx - spoken_dict[last_number]
                spoken_dict[last_number] = idx

            last_number = new_number
            idx += 1

        print("For sequence {}".format(line))
        print("Number spoken at turn {} is {}".format(max_turn, new_number))

# print("Part2:")



# %% Day16

from timer import Timer

def parse_day16(filename):
    with open(filename, "r") as fp:
        lines = fp.readlines()
    lines = [command.strip() for command in lines]

    rules = {}
    my_ticket = []
    nearby_tickets = []

    def parse_rule_str(rule_str):
        rule_str = rule_str.replace(' ', '')
        cv = rule_str.split(':')
        ranges_str = cv[1].split('or')
        ranges = []
        for rs in ranges_str:
            ranges.append([int(s) for s in rs.split('-')])
        rule = {cv[0] : ranges}
        return rule

    line = lines.pop(0)
    while line:
        rules.update(parse_rule_str(line))
        line = lines.pop(0)

    while lines:
        line = lines.pop(0)
        if line == "your ticket:":
            my_ticket = [int(l) for l in lines.pop(0).split(',')]
        if line == "nearby tickets:":
            while lines:
                nearby_tickets.append([int(l) for l in lines.pop(0).split(',')])
    
    return rules, my_ticket, nearby_tickets

def isInRule(number, rule):
    return any([number >= rule_range[0] and number <= rule_range[1] for rule_range in rule]) > 0

with Timer():
    rules, my_ticket, nearby_tickets = parse_day16("day16-input.txt")

    not_applied_numbers = []
    all_numbers = [item for sublist in [my_ticket] + nearby_tickets for item in sublist]
    for number in all_numbers:
        if not any([isInRule(number, rule) for rule in rules.values()]):
            not_applied_numbers.append(number)

    print("Part1: scanning error rate is {}".format(sum(not_applied_numbers))) # 23009

with Timer():
    rules, my_ticket, nearby_tickets = parse_day16("day16-input.txt")

    valid_tickets = []
    for ticket in nearby_tickets:
        if all([any([isInRule(number, rule) for rule in rules.values()]) for number in ticket]):
            valid_tickets.append(ticket)

    num_numbers = len(nearby_tickets[0])

    valid_tickets.append(my_ticket)
    rule_position_map = [[False] * num_numbers for _ in range(len(rules))]

    rule_names = sorted(list(rules.keys()))

    for position in range(num_numbers):
        for i, rule_key in enumerate(rule_names):
            if all([isInRule(valid_tickets[i][position], rules[rule_key]) for i in range(len(valid_tickets)) ]):
                rule_position_map[i][position] = True

    # [( number of matches, rule_idx )]
    sorted_rules = sorted([(sum(pos), i) for i, pos in enumerate(rule_position_map)])

    rule_position = [-1] * num_numbers

    rule_position_str = {}

    rules_to_match = 20

    while rules_to_match:
        for rule_idx_sort in sorted_rules:
            rule_idx = rule_idx_sort[1]
            positions = [i for i, v in enumerate(rule_position_map[rule_idx]) if v == True]

            if len(positions) == 1:
                # print("matched rule {}, idx {}".format(rule_names[rule_idx], rule_idx))
                rules_to_match -= 1
            else:
                # print("Rules left {}".format(rules_to_match))
                break

            position_idx = positions[0]

            rule_position_str[rule_names[rule_idx]] = position_idx
            
            for p in range(num_numbers):
                rule_position_map[rule_idx][p] = False
            for r in range(num_numbers):
                rule_position_map[r][position_idx] = False

    result = 1
    for rule_name in rule_names:
        if "departure" in rule_name:
            result *= my_ticket[rule_position_str[rule_name]]

    print("Part2: result is {}".format(result)) # 10458887314153


# %% Day 17


# %%
