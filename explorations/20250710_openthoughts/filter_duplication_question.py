import json
from tqdm import tqdm


for shard in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', 
              '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', ]:
    # in_file = '/fsx-comem/diwu0162/OpenThoughts3/data/train_full_1.2m.jsonl'
    in_file = f'/fsx-comem/diwu0162/OpenThoughts3/data/train_full_1.2m.jsonl.{shard}.part'
    out_file = in_file + '.qdedup.jsonl'


    in_data_list = [json.loads(line) for line in open(in_file).readlines()]
    in_data_dict = {}
    for entry in tqdm(in_data_list, desc='Reading data'):
        if entry['conversations'][0]['value'] not in in_data_dict:
            in_data_dict[entry['conversations'][0]['value']]  = []
        in_data_dict[entry['conversations'][0]['value']].append(entry)


    out_data = []
    dup_count = 0
    incomplete_count = 0
    for key, entries in tqdm(in_data_dict.items(), desc='Filtering data'):
        dup_count += len(entries) - 1
        complete_thought_found = False
        for entry in entries:
            if '</think>' in entry['conversations'][1]['value']:
                complete_thought_found = True
                out_data.append(entry)
                break
        if not complete_thought_found:
            incomplete_count += 1
            out_data.append(entries[0])

    print(f'{dup_count} duplicated question entries dropped.')
    print(f'{incomplete_count} incomplete thoughts included in the data.')


    with open(out_file, 'w') as f:
        for entry in out_data:
            print(json.dumps(entry), file=f)
        print('Saved to', out_file)