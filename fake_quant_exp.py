import warnings
warnings.filterwarnings('ignore')
import torch, random, argparse, re
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset

random.seed(0)
torch.manual_seed(0)

MODEL_PATH = '/mnt/data/gloomyteam/kivi_clone/models/Llama-2-7b-hf'
GROUP_SIZE = 32

CONFIGS = {
    'fp16_baseline': {'k': None,         'v': None,         'bits': 2},
    '4bit_K-T_V-T':  {'k': 'pertoken',   'v': 'pertoken',   'bits': 4},
    '2bit_K-T_V-T':  {'k': 'pertoken',   'v': 'pertoken',   'bits': 2},
    '2bit_K-C_V-C':  {'k': 'perchannel', 'v': 'perchannel', 'bits': 2},
    '2bit_K-T_V-C':  {'k': 'pertoken',   'v': 'perchannel', 'bits': 2},
    '2bit_K-C_V-T':  {'k': 'perchannel', 'v': 'pertoken',   'bits': 2},
}

def fake_quant_perchannel(x, group_size, bits):
    B, nh, T, D = x.shape
    if T < group_size:
        return x
    T_trunc = (T // group_size) * group_size
    x_main = x[:, :, :T_trunc, :]
    x_tail = x[:, :, T_trunc:, :]
    max_int = 2 ** bits - 1
    xr = x_main.reshape(B, nh, T_trunc // group_size, group_size, D)
    mn = xr.min(dim=-2, keepdim=True)[0]
    mx = xr.max(dim=-2, keepdim=True)[0]
    scale = ((mx - mn) / max_int).clamp(min=1e-6)
    xq = ((xr - mn) / scale).clamp(0, max_int).round_()
    out = (xq * scale + mn).reshape(B, nh, T_trunc, D)
    return torch.cat([out, x_tail], dim=2) if x_tail.shape[2] > 0 else out

def fake_quant_pertoken(x, group_size, bits):
    B, nh, T, D = x.shape
    if D < group_size:
        return x
    max_int = 2 ** bits - 1
    xr = x.reshape(B, nh, T, D // group_size, group_size)
    mn = xr.min(dim=-1, keepdim=True)[0]
    mx = xr.max(dim=-1, keepdim=True)[0]
    scale = ((mx - mn) / max_int).clamp(min=1e-6)
    xq = ((xr - mn) / scale).clamp(0, max_int).round_()
    return (xq * scale + mn).reshape(B, nh, T, D)

def apply_hooks(model, k_dir, v_dir, bits):
    hooks = []
    def make_hook(kd, vd, b, layer_idx):
        def fn(module, input, output):
            if not isinstance(output, tuple) or len(output) < 3:
                return output
            past_kv = output[2]
            if past_kv is None:
                return output
            from transformers.cache_utils import DynamicCache
            if isinstance(past_kv, DynamicCache):
                if layer_idx < len(past_kv.key_cache):
                    k = past_kv.key_cache[layer_idx]
                    v = past_kv.value_cache[layer_idx]
                    if kd == 'pertoken':
                        k = fake_quant_pertoken(k, GROUP_SIZE, b)
                    elif kd == 'perchannel':
                        k = fake_quant_perchannel(k, GROUP_SIZE, b)
                    if vd == 'pertoken':
                        v = fake_quant_pertoken(v, GROUP_SIZE, b)
                    elif vd == 'perchannel':
                        v = fake_quant_perchannel(v, GROUP_SIZE, b)
                    past_kv.key_cache[layer_idx] = k
                    past_kv.value_cache[layer_idx] = v
                return output
            elif isinstance(past_kv, tuple):
                k, v = past_kv[0], past_kv[1]
                if kd == 'pertoken':
                    k = fake_quant_pertoken(k, GROUP_SIZE, b)
                elif kd == 'perchannel':
                    k = fake_quant_perchannel(k, GROUP_SIZE, b)
                if vd == 'pertoken':
                    v = fake_quant_pertoken(v, GROUP_SIZE, b)
                elif vd == 'perchannel':
                    v = fake_quant_perchannel(v, GROUP_SIZE, b)
                return output[:2] + ((k, v),) + output[3:]
            return output
        return fn
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(k_dir, v_dir, bits, layer_idx)))
    return hooks

def extract_gold_gsm8k(text):
    m = re.search(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    return m.group(1).replace(',', '').strip() if m else None

def extract_pred_gsm8k(text):
    m = re.search(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    matches = re.findall(r'[-+]?\d*\.?\d+', text.replace(',', ''))
    return matches[-1].strip() if matches else None

class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.newline_id = tokenizer.encode(chr(10), add_special_tokens=False)[-1]
        self.hash_ids   = tokenizer.encode('####', add_special_tokens=False)
        self.prompt_len = prompt_len
        self.seen_hash  = False
    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.prompt_len:].tolist()
        if self.hash_ids[0] in gen:
            self.seen_hash = True
        return self.seen_hash and input_ids[0, -1].item() == self.newline_id

def eval_gsm8k(model, tokenizer, num_samples):
    dataset = load_dataset('gsm8k', 'main')
    fewshot = ''
    for i in range(5):
        fewshot += 'Question: ' + dataset['train'][i]['question'] + '\n'
        fewshot += 'Answer: '   + dataset['train'][i]['answer']   + '\n'
    correct = 0
    for i in range(num_samples):
        prompt = fewshot + 'Question: ' + dataset['test'][i]['question'] + '\nAnswer:'
        inputs = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        stopper = StopOnNewline(tokenizer, inputs.shape[1])
        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=256, do_sample=False,
                                    stopping_criteria=StoppingCriteriaList([stopper]))
        pred_text = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        ok = (extract_pred_gsm8k(pred_text) is not None and
              extract_gold_gsm8k(dataset['test'][i]['answer']) is not None and
              extract_pred_gsm8k(pred_text) == extract_gold_gsm8k(dataset['test'][i]['answer']))
        correct += int(ok)
        if (i+1) % 20 == 0 or (i+1) == num_samples:
            print('  [GSM8K ' + str(i+1) + '/' + str(num_samples) + '] ' + str(round(100.0*correct/(i+1), 2)) + '%')
    return round(100.0 * correct / num_samples, 2)

def eval_coqa(model, tokenizer, num_samples):
    dataset = load_dataset('coqa', trust_remote_code=True)
    correct = 0
    total = min(num_samples, len(dataset['validation']))
    for i in range(total):
        story   = dataset['validation'][i]['story']
        question = dataset['validation'][i]['questions'][0]
        answer   = dataset['validation'][i]['answers']['input_text'][0].lower().strip()
        prompt = 'Read the following story and answer the question.\nStory: ' + story + '\nQuestion: ' + question + '\nAnswer:'
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).input_ids.cuda()
        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=32, do_sample=False)
        pred = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).lower().strip()
        ok = answer in pred or pred in answer
        correct += int(ok)
        if (i+1) % 20 == 0 or (i+1) == total:
            print('  [CoQA ' + str(i+1) + '/' + str(total) + '] ' + str(round(100.0*correct/(i+1), 2)) + '%')
    return round(100.0 * correct / total, 2)

def eval_truthfulqa(model, tokenizer, num_samples):
    dataset = load_dataset('truthful_qa', 'generation', trust_remote_code=True)
    scores = []
    total = min(num_samples, len(dataset['validation']))
    for i in range(total):
        question = dataset['validation'][i]['question']
        best_ans = dataset['validation'][i]['best_answer'].lower().strip()
        prompt = 'Q: ' + question + '\nA:'
        inputs = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=64, do_sample=False)
        pred = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).lower().strip()
        words_pred = set(pred.split())
        words_gold = set(best_ans.split())
        overlap = len(words_pred & words_gold)
        if len(words_pred) + len(words_gold) > 0:
            f1 = 2 * overlap / (len(words_pred) + len(words_gold))
        else:
            f1 = 0.0
        scores.append(f1)
        if (i+1) % 20 == 0 or (i+1) == total:
            print('  [TruthfulQA ' + str(i+1) + '/' + str(total) + '] avg F1: ' + str(round(sum(scores)/len(scores), 4)))
    return round(sum(scores) / len(scores) * 100, 2)

def run_eval(model, tokenizer, num_samples, k_dir, v_dir, bits, name, tasks):
    print('=' * 55)
    print('[실험] ' + name)
    print('  K=' + str(k_dir or 'fp16') + ' | V=' + str(v_dir or 'fp16') + ' | bits=' + str(bits))
    print('=' * 55)
    hooks = apply_hooks(model, k_dir, v_dir, bits) if (k_dir or v_dir) else []
    task_results = {}
    if 'gsm8k' in tasks:
        print('  >> GSM8K 평가 중...')
        task_results['gsm8k'] = eval_gsm8k(model, tokenizer, num_samples)
    if 'coqa' in tasks:
        print('  >> CoQA 평가 중...')
        task_results['coqa'] = eval_coqa(model, tokenizer, num_samples)
    if 'truthfulqa' in tasks:
        print('  >> TruthfulQA 평가 중...')
        task_results['truthfulqa'] = eval_truthfulqa(model, tokenizer, num_samples)
    for h in hooks:
        h.remove()
    for t, v in task_results.items():
        print('  [' + t + '] ' + str(v) + '%')
    return task_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()))
    parser.add_argument('--tasks', nargs='+', default=['gsm8k', 'coqa', 'truthfulqa'])
    args = parser.parse_args()
    print('=' * 55)
    print('KIVI 논문 Table 1: Fake Quantization 실험')
    print('G=' + str(GROUP_SIZE) + ' | 샘플=' + str(args.num_samples) + ' | 태스크=' + str(args.tasks))
    print('=' * 55)
    print('모델 로딩 중...')
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    print('완료!')
    all_results = {}
    for name in args.configs:
        if name not in CONFIGS:
            print('[경고] 알 수 없는 설정: ' + name)
            continue
        cfg = CONFIGS[name]
        all_results[name] = run_eval(model, tokenizer, args.num_samples,
                                     cfg['k'], cfg['v'], cfg['bits'], name, args.tasks)
    print(chr(10) + '=' * 55)
    print('최종 결과 요약')
    print('=' * 55)
    header = 'Config'.ljust(20)
    for t in args.tasks:
        header += t.rjust(14)
    print(header)
    print('-' * 55)
    for name, res in all_results.items():
        row = name.ljust(20)
        for t in args.tasks:
            row += (str(res.get(t, '-')) + '%').rjust(14)
        print(row)
    with open('fake_quant_results.txt', 'w') as f:
        f.write('KIVI Fake Quantization 실험 결과' + chr(10))
        f.write('G=' + str(GROUP_SIZE) + ' | 샘플=' + str(args.num_samples) + chr(10) + chr(10))
        f.write(header + chr(10) + '-'*55 + chr(10))
        for name, res in all_results.items():
            row = name.ljust(20)
            for t in args.tasks:
                row += (str(res.get(t, '-')) + '%').rjust(14)
            f.write(row + chr(10))
    print('결과 저장 완료: fake_quant_results.txt')

if __name__ == '__main__':
    main()
