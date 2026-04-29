from huggingface_hub import snapshot_download
import os

# 임시 폴더 경로를 용량 큰 /mnt/data 쪽으로 변경 (핵심 수정)
os.environ["HF_HOME"] = "/mnt/data/gloomyteam/hf_cache"
os.environ["TMPDIR"] = "/mnt/data/gloomyteam/tmp"

# 임시 폴더 미리 생성
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

token = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Llama-2-7b-hf"
save_dir = "/mnt/data/gloomyteam/kivi_clone/models/Llama-2-7b-hf"
os.makedirs(save_dir, exist_ok=True)

print(f"다운로드 시작: {model_id}")
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    token=HF_TOKEN,
)
print(f"완료! 저장 위치: {save_dir}")
