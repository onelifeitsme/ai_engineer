import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks = []


def get_chunks(path):
    chunks = []
    with open(path, mode='r') as f:
        current_chunk = ''
        for line in f.readlines():
            if line == '\n':
                continue
            if all([line[0].isdigit(), line[1] == '.']):
                if current_chunk and all([current_chunk[0].isdigit(), current_chunk[1] == '.']):
                    chunks.append(current_chunk.strip('\n'))
                current_chunk = line
            else:
                current_chunk += line
    return chunks
