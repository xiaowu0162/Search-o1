import sys
import json


def segment_thoughts_v1(x):
    return x.strip().split('\n\n')

def segment_thoughts_v2(x):
    # note: excluding things like "so" "therefore", "but", "let me" 
    reasoning_word_list = [
        'okay', 'hmm', 'wait', 'but wait', 'oh wait', 'no wait', 'no, wait', 'but let me', 'but actually', 'alternatively', 
        'now', 'the question', 'ah', 'oh', 'next', 'another angle', 'another approach', 'also', 'hold on', 'looking it up', 
        'another point', 'I don\'t think', 'perhaps I', 'putting this together', 'Putting it all together', 'i\'m', 'but i\'m',   
        'let me think again', 'I don\'t see', 'maybe I', 'alternative', "I wonder if", "another way", 'an alternative', 
    ]
    prefix_len = max([len(x) for x in reasoning_word_list])
    newline_segmented_thoughts = segment_thoughts_v1(x)
    final_thoughts = []
    for t in newline_segmented_thoughts:
        t_lower = t.lower()
        is_segment_start = False
        for r_w in reasoning_word_list:
            if t_lower.startswith(r_w.lower()):
                is_segment_start = True
                break
        if is_segment_start or not final_thoughts:
            final_thoughts.append(t)
        else:
            final_thoughts[-1] += '\n\n' + t
    return final_thoughts

if __name__ == '__main__':
    
    model_alias = sys.argv[1] #  'ds-llama-8b'
    task = sys.argv[2] #  'aime'
    idx = int(sys.argv[3]) #  20
    
    in_file = f'/fsx-comem/diwu0162/Search-o1/explorations/20250710_hint_eval_dataset/diff_direct_pred_and_hint/{model_alias}/{task}.json'
    in_data = json.load(open(in_file))
    entry = in_data[idx]
    print('Question:')
    print(entry['question'].split('Question:')[-1].strip())
    print('Answer:', entry['answer'])
    print('Hint:')
    print(entry['hint_pred']['prompt'].split('<think>')[-1].strip())
    thinking = entry['hint_pred']['output'].split('</think>')[0].strip('Okay,').strip()
    thinking_segmented = segment_thoughts_v2(thinking)
    # print(json.dumps({f'Step {i+1}': x for i, x in enumerate(thinking_segmented)}, indent=4))
    
    for i, t in enumerate(thinking_segmented):
        print(f'\n\n================================== Step {i+1} ==================================\n\n')
        print(t)
    # print(*segment_thoughts_v2(thinking), sep=f'\n\n================================== Step {i} ==================================\n\n')