"""Rebuild the self-contained notebook. No private-repo token at runtime."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def cell(kind,source):
    result={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result

intro='''# Muhtemel Aşk: indir → çevir → altyazıyı MP4'e göm

RunPod veya açık PC gerekmez. Colab T4 GPU kullanır. Kaynak video, ASR parçaları,
çeviri paketi ve Endonezce altyazısı gömülü MP4 Drive'da saklanır.

1. GPU oturumunda `MODE = "prepare"`: bölüm indirilir. Resmî Türkçe altyazı hazırsa
   aynı video kaynağıyla kullanılır; yoksa **beklenmeden large-v3 ASR başlar**.
2. Çeviri mevcut ChatGPT ile, paketteki özgün talimat ve sözlükle yapılır.
3. `MODE = "finish"`: çeviri doğrulanır ve eski projenin NVIDIA destekli koduyla
   altyazı **MP4'e gömülür**. SRT tek başına son teslim değildir.

Cuma 25 Eylül 06:00 **Singapur** için ChatGPT kontrol görevi kuruldu. Bu notebook
kendi başına zamanlayıcı değildir. Colab GPU tahsisi ve Drive izni geçerli olmalıdır;
ücretsiz Colab gözetimsiz başlatma/çalışma garantisi vermez. Görev gerçek erişim
engeli olursa bildirir; olmayan altyazıyı sessizce beklemez.

Yalnız kaynak ASR kelime zamanları kullanılır; düzeltilmiş metne tekrar forced
alignment yoktur. Belirsiz senkron yerleri dinleme ekranından düzeltilir. İncelemesi
bitmemiş video `.draft.mp4` olarak üretilir; kalite kontrolü yapılmış gibi sunulmaz.
'''
settings='''EPISODE = 15
MODE = "prepare"  # "prepare" veya "finish"
SOURCE_URL = ""  # Boşsa Show TV sayfasındaki gerçek MP4 bulunur; ayrıca YouTube URL kabul edilir.
FORCE_ASR = False  # True: yayıncı altyazısı olsa da ses üzerinden çalışır.
SOURCE_FILE = ""  # Alternatif: Drive içindeki kaynak videonun tam yolu.
DRIVE_ROOT = "/content/drive/MyDrive/Muhtemel_Ask_Subtitles/Colab_v1"
'''
install='''import os, subprocess, sys, shutil
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
packages = ["PyYAML==6.0.2", "ipywidgets==8.1.7"]
if MODE == "prepare":
    packages += ['faster-whisper==1.2.1', 'ctranslate2==4.8.1', 'yt-dlp[default]==2026.8.19', 'numpy==2.2.6', 'onnxruntime==1.30.0', 'huggingface-hub==1.32.0', 'tokenizers==0.23.2', 'av==18.1.0', 'nvidia-cublas-cu12==12.8.4.1', 'nvidia-cudnn-cu12==9.10.2.21']
subprocess.run([sys.executable,"-m","pip","install","-q",*packages],check=True,timeout=900)
if MODE == "prepare":
    import nvidia.cublas.lib, nvidia.cudnn.lib
    from pathlib import Path
    libs=[str(Path(nvidia.cublas.lib.__path__[0])),str(Path(nvidia.cudnn.lib.__path__[0]))]
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs+[os.environ.get("LD_LIBRARY_PATH","")])
    if SOURCE_URL and any(host in SOURCE_URL for host in ["youtube.com", "youtu.be"]):
        import hashlib, urllib.request, zipfile, io
        tool_dir=Path('/content/ma-sub-tools');tool_dir.mkdir(exist_ok=True)
        executable=tool_dir/'deno'
        if not executable.exists():
            url='https://github.com/denoland/deno/releases/download/v2.9.5/deno-x86_64-unknown-linux-gnu.zip'
            with urllib.request.urlopen(url,timeout=60) as response: archive=response.read(100*1024*1024)
            if hashlib.sha256(archive).hexdigest()!='8b010a3b1a4a0188a67cdb8a7a27348b2a501af78aec7fc74f2ace167368d530':
                raise RuntimeError('Deno archive checksum mismatch')
            with zipfile.ZipFile(io.BytesIO(archive)) as z: executable.write_bytes(z.read('deno'))
            executable.chmod(0o755)
        os.environ['PATH']=str(tool_dir)+':'+os.environ['PATH']
if not shutil.which("ffmpeg"):
    raise RuntimeError("FFmpeg is missing from this Colab runtime")
'''
# Deno version/checksum are inherited from the existing repository bootstrap.
config_files={p.name:p.read_text(encoding='utf-8') for p in sorted((ROOT/'config/production').iterdir())
              if p.name in ['TRANSLATION_INSTRUCTIONS.md','names.yaml','religious_terms.yaml']}
support_names=['__init__.py','official_subtitles.py','video_flow.py','progress.py','reliability.py',
               'engine/__init__.py','engine/burned_mp4.py','engine/download.py','engine/srt.py']
support_files={name:(ROOT/'src/mas'/name).read_text() for name in support_names}
setup='''from google.colab import drive
from pathlib import Path
import importlib, sys, shutil
drive.mount("/content/drive")
ROOT = Path(DRIVE_ROOT) / f"Muhtemel Ask {EPISODE}.Bolum"
CONFIG = Path("/content/ma_sub_config")
CONFIG.mkdir(exist_ok=True)
'''+ 'support_files = '+repr(support_files)+'\n'+'''for name,content in support_files.items():
    target=Path('/content/mas')/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(content,encoding='utf-8')
shutil.copyfile('/content/ma_sub_colab.py','/content/mas/colab_flow.py')
'''+ 'config_files = '+repr(config_files)+'\n'+'''for name,content in config_files.items():
    (CONFIG/name).write_text(content,encoding="utf-8")
if "/content" not in sys.path: sys.path.insert(0,"/content")
from mas import colab_flow as flow, video_flow as video
flow = importlib.reload(flow);video = importlib.reload(video)
RETURN = ROOT / "handoff" / f"Muhtemel Ask {EPISODE}.Bolum_TRANSLATED.zip"
print("Bölüm klasörü:",ROOT)
'''
run='''if MODE == "prepare":
    PACK = video.prepare(ROOT, EPISODE, CONFIG, source_file=SOURCE_FILE, source_url=SOURCE_URL, force_asr=FORCE_ASR)
    print("ChatGPT'ye verilecek dosya:",PACK)
    print("Çeviri beklerken GPU oturumunu kapat. Drive'daki dosyalar korunur.")
elif MODE == "finish":
    report = video.finish(ROOT, RETURN)
    print("Altyazısı gömülü MP4:", report["path"])
else:
    raise ValueError("MODE prepare veya finish olmalı")
'''
review='''# Yalnız çeviri döndükten sonra çalıştır.
if flow.read_json(ROOT/"video_workflow.json")["route"] == "asr":
    flow.review_ui(ROOT, RETURN)
else:
    print("Yayıncı zamanları kullanıldı; kaynakla senkron dinleme kontrolü gerekir.")
'''
preview='''# Son çıktının durumu ve tam Drive yolu.
if (ROOT / "video_output.json").exists():
    result=flow.read_json(ROOT / "video_output.json")
    print(result["status"],result["path"])
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'colab':{'name':'Muhtemel_Ask.ipynb','provenance':[]}},
    'cells':[cell('markdown',intro),cell('code',settings),cell('code',install),
             cell('markdown','## İşlem kodu\nBu hücre repodaki `src/mas/colab_flow.py` dosyasının birebir kopyasıdır.\n'),
             cell('code','%%writefile /content/ma_sub_colab.py\n'+(ROOT/'src/mas/colab_flow.py').read_text()),
             cell('code',setup),cell('code',run),
             cell('markdown','## Dinleme ve düzeltme\n`latest_output.json` sorunlu satırları listeler. Başlangıç/bitiş alanları bölümün mutlak milisaniyesidir. Klibin başlangıcı ayrıca gösterilir. Zamanları sırf okuma hızını düşürmek için uzatma. Eksik konuşmada birden çok satır gerekiyorsa JSON alanında ayrı zamanları kullan.\n'),
             cell('code',review),cell('code',preview)]}
for i,c in enumerate(nb['cells']):c['id']=f'cell-{i:02d}'
(ROOT/'colab/Muhtemel_Ask.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
if __name__=='__main__':print('Built self-contained notebook')
