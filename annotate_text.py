import argparse
import csv
import json
import os
import sys

from models import model_chassis
from utils.utils import AnnotateArgsExp


def annotate_stream(model, threshold, input_iterator, output_writer):
    for i, (key, text) in enumerate(input_iterator):
        sys.stdout.write(
            "\rProgress:: %.2f%%" % (100.0 * i // len(input_iterator)))
        sys.stdout.flush()
        ants = model.annotate_text(text, threshold=threshold)
        output_writer.write(key, ants, model)
    sys.stdout.write("\n")


class DirOutputStream:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def write(self, key, ants, model):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.output_dir + '/' + key, 'w') as fp:
            for ant in ants:
                fp.write(
                    '\t'.join(map(str, ant)) + '\t' + model.ont.names[ant[2]][
                        0] + '\n')


class CSVOutputStream:

    def __init__(self, output_csv_file):
        self.output_csv_file = output_csv_file
        open(self.output_csv_file, 'w').close()

    def write(self, key, ants):
        with open(self.output_csv_file, 'a') as fw:
            csv_writer = csv.writer(fw, delimiter=';')
            csv_writer.writerow([key] + [str(x) for x in ants])


class DirInputStream:

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.filelist = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.filelist)

    def __iter__(self):
        self.filelist_iter = iter(self.filelist)
        return self

    def __next__(self):
        filename = next(self.filelist_iter)
        return filename, open(self.input_dir + '/' + filename,
                              encoding='utf-8').read()


class JsonInputStream:

    def __init__(self, input_json_file):
        with open(input_json_file, 'r') as fp:
            self.notes = json.load(fp)

    def __len__(self):
        return len(self.notes)

    def __iter__(self):
        self.notes_iter = iter(self.notes)
        return self

    def __next__(self):
        key = self.notes_iter.next()
        return key, self.notes[key]


class CSVInputStream:
    def __init__(self, input_csv_file, max_rows=-1):
        self.input_csv_file = input_csv_file
        if max_rows == -1:
            with open(self.input_csv_file) as fp:
                self.length = sum(1 for row in fp)
        else:
            self.length = max_rows

    def __len__(self):
        return self.length

    def __iter__(self):
        self.csvfile = open(self.input_csv_file, 'r')
        self.reader = csv.reader(self.csvfile, delimiter=',')
        self.csv_iter = iter(self.reader)

        try:
            next(self.csv_iter)
        except StopIteration:
            self.csvfile.close()
            raise StopIteration

        self.ct = 0
        return self

    def __next__(self):
        self.ct += 1
        if self.ct > self.length:
            self.csvfile.close()
            raise StopIteration

        try:
            row = next(self.csv_iter)
        except StopIteration:
            self.csvfile.close()
            raise StopIteration

        key = row[0]
        text = row[-1]
        return key, text


def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--params', help="Path to json containing parameters.")
    args = AnnotateArgsExp(parser.parse_args().params)
    model = model_chassis.ExplicitChassis.loadfromfile(args.model_dir)
    print("model loaded")

    if args.output_dir.endswith('.csv'):
        output_stream = CSVOutputStream(args.output_dir)
    else:
        output_stream = DirOutputStream(args.output_dir)

    if args.input_dir.endswith('.csv'):
        input_stream = CSVInputStream(args.input_dir, args.max_rows)
    elif args.input_dir.endswith('.json'):
        input_stream = JsonInputStream(args.output_dir)
    else:
        input_stream = DirInputStream(args.input_dir)

    print("streams ready")
    annotate_stream(model, args.threshold, input_stream, output_stream)

if __name__ == "__main__":
    main()
