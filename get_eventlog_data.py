import os
import json
import csv
import itertools
import datetime
import random


def preprocess():
    if not os.path.exists('data'):
        os.mkdir('data')
        print('made directory ./data/')

    file_path = os.path.join('data', 'bpi2017w_converted_sample.csv')
    raw_data = []
    with open(file_path) as csvfile:
        # ignore header: https://stackoverflow.com/questions/11349333/when-processing-csv-data-how-do-i-ignore-the-first-line-of-data
        has_header = csv.Sniffer().has_header(csvfile.read(1024))
        csvfile.seek(0)  # Rewind.
        reader = csv.reader(csvfile)
        if has_header:
            next(reader)  # Skip header row.
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            toadd = row[:3]
            toadd.append(int(float(row[-1])))
            raw_data.append(toadd)
            #print(toadd)

    def sortIdx(elem):
        return elem[0]

    # sorting: https://stackoverflow.com/questions/20183069/how-to-sort-multidimensional-array-by-column/20183121
    # 01.02.2012 08:19:00 - date still index 2
    # https://stackabuse.com/converting-strings-to-datetime-in-python/
    new_list = [
        [instance[1:] for instance in sorted(group, key=lambda x: datetime.datetime.strptime(x[2], '%d.%m.%Y %H:%M'))]
        for group in
        [list(g) for k, g in itertools.groupby(sorted(raw_data, key=sortIdx), sortIdx)]]
    random.shuffle(new_list)
    # print("_____________________")
    # print(new_list[1])
    # print("_____________________")

    ones = [part for part in new_list if int(max([int(tt[-1]) for tt in part]) - 1) == 1]
    zeroes = [part for part in new_list if int(max([int(tt[-1]) for tt in part]) - 1) == 0]

    eq_amount = min(len(ones), len(zeroes))
    eq_distr_data = []
    for i in range(eq_amount):
        eq_distr_data.append(ones[i])
        eq_distr_data.append(zeroes[i])
    random.shuffle(eq_distr_data)


    train = eq_distr_data[:int((len(eq_distr_data) / 10) * 9)]
    valid = eq_distr_data[int((len(eq_distr_data) / 10) * 9):]
    raw_data = [train, valid]
    print("new_list: ", len(new_list))
    print("eq_distr_data: ", len(eq_distr_data))
    print(len(train) + len(valid))
    return raw_data


# with start node
# all nodes are Hydrogen -> no node featrues
# only one edge type
def experiment1(raw_data):
    def makeEdges(group):
        edges = []
        prev = -1
        node_ids = [0]
        for nbr, instance in enumerate(group):
            if prev != int(instance[0]):
                if not (edges.__contains__((prev + 1, 1, int(instance[0]) + 1)) |
                        edges.__contains__((int(instance[0]) + 1, 1, prev + 1))):
                    if not node_ids.__contains__(int(instance[0]) + 1):
                        node_ids.append(int(instance[0]) + 1)
                    edges.append((prev + 1, 1, int(instance[0]) + 1))
                    prev = int(instance[0])
        # for i in range(max(node_ids)):
        #   if not node_ids.__contains__(i):
        #      edges.append((i, 1, i))
        return edges, max(node_ids)  # len(edges)

    processed_data = {'train': [], 'valid': []}
    for j, section in enumerate(['train', 'valid']):
        for i, group in enumerate(raw_data[j]):
            nodes = []
            target = int(max([int(tt[-1]) for tt in group]) - 1)
            # if group[-1][0] == '2':
            #    target = 1
            #    group = group[:-1]
            edges, ret = makeEdges(group)
            if len(edges) < 2:
                continue
            for k in range(ret + 1):
                nodes.append([1, 0, 0, 0, 0])
            if target < 0:
                continue
            processed_data[section].append({
                'targets': [[target]],
                'graph': edges,
                'node_features': nodes
            })
        with open('process_ex1_%s.json' % section, 'w') as f:
            json.dump(processed_data[section], f)


# start edge -> type 1
# end edge -> type 2
# directed edge -> type 3
# undirected edge -> type 4
# recursive edge -> type 5
def experiment2(raw_data):
    def makeEdges(group):
        edges = []
        prev: int = -1
        node_ids = [0]
        for nbr, instance in enumerate(group):
            #start edge
            if nbr == 0:
                edges.append((prev + 1, 1, int(instance[0]) + 1))
            #recursive edge
            elif prev == int(instance[0]) and not edges.__contains__((int(instance[0]) + 1, 5, int(instance[0]) + 1)):
                edges.append((int(instance[0]) + 1, 5, int(instance[0]) + 1))
                if edges.__contains__((int(instance[0]) + 1, 3, int(instance[0]) + 1)):
                    edges.remove((int(instance[0]) + 1, 3, int(instance[0]) + 1))
            elif edges.__contains__((int(instance[0]) + 1, 3, prev + 1)) \
                    and not edges.__contains__((int(instance[0]) + 1, 4, prev + 1)) \
                    and not edges.__contains__((prev + 1, 4, int(instance[0]) + 1)) \
                    and not edges.__contains__((prev + 1, 3, int(instance[0]) + 1)):
                edges.append((prev + 1, 4, int(instance[0]) + 1))
                edges.append((int(instance[0]) + 1, 4, prev + 1))
                if edges.__contains__((prev + 1, 3, int(instance[0]) + 1)):
                    edges.remove((prev + 1, 3, int(instance[0]) + 1))
                if edges.__contains__((int(instance[0]) + 1, 3, prev + 1)):
                    edges.remove((int(instance[0]) + 1, 3, prev + 1))
            elif not edges.__contains__((prev + 1, 3, int(instance[0]) + 1)) and not edges.__contains__((prev + 1, 5, int(instance[0]) + 1)) and not edges.__contains__((prev + 1, 4, int(instance[0]) + 1)):
                edges.append((prev + 1, 3, int(instance[0]) + 1))

            prev = int(instance[0])
            node_ids.append(int(instance[0]) + 1)

        edges.append(((prev + 1, 2, 0)))

        # giving the nodes their label as feature -> Label information gets lost, only topology is regarded. here the Event ID is preserved.
        features = []
        for k in range(max(node_ids) + 1):
            feature = []
            for k in range(9):
                feature.append(int(0))
            features.append(feature)
        for node in node_ids:
            feature = []
            for k in range(9):
                feature.append(int(0))
            feature[node] = 1
            features[node] = feature

        return edges, features  # len(edges)

    processed_data = {'train': [], 'valid': []}
    for j, section in enumerate(['train', 'valid']):
        for i, group in enumerate(raw_data[j]):
            # nodes = []
            target = int(max([int(tt[-1]) for tt in group]) - 1)
            # if group[-1][0] == '2':
            #    target = 1
            #    group = group[:-1]
            edges, features = makeEdges(group)
            if len(edges) < 2:
                continue
            # for k in range(ret):
            #     nodes.append([1, 0, 0, 0, 0])
            if target < 0:
                continue
            processed_data[section].append({
                'targets': [[target]],
                'graph': edges,
                'node_features': features
            })
        with open('process_ex5_%s.json' % section, 'w') as f:
            json.dump(processed_data[section], f)


# def getProcessModel2(raw_data):
#     def makeEdges(group):
#         edges = []
#         prev: int = -1
#         node_ids = [0]
#         for nbr, instance in enumerate(group):
#             if not (edges.__contains__((prev + 1, int(instance[0]) + 1))):
#                 if not node_ids.__contains__(int(instance[0]) + 1):
#                     node_ids.append(int(instance[0]) + 1)
#                 edges.append((prev + 1, int(instance[0]) + 1))
#                 prev = int(instance[0])
#
#         edges.append(((prev + 1, 99)))
#         return edges
#
#     data = raw_data[0] + raw_data[1]
#
#     edges = []
#     for i, group in enumerate(data):
#         tmp_edges = makeEdges(group)
#         for i in tmp_edges:
#             if not edges.__contains__(i):
#                 edges.append(i)
#
#     processed_data = {'process': []}
#     processed_data['process'].append(edges)
#     with open('process5.json', 'w') as f:
#         json.dump(processed_data, f)
#
#     print(len(data))


raw_data = preprocess()

# getProcessModel2(raw_data)

experiment1(raw_data)
experiment2(raw_data)
