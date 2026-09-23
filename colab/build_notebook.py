"""Rebuild the self-contained notebook. No private-repo token at runtime."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def cell(kind,source):
    result={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result

intro='''# Muhtemel Aşk: Türkçe ses -> doğal Endonezce altyazı

PC ve RunPod gerekmez. Hazırlama aşaması Colab GPU kullanır; çeviri burada ChatGPT'de
mevcut sözlük ve çeviri kurallarıyla yapılır. Son aşama CPU oturumunda çalışabilir.

1. Telefonda Colab'ı aç. Runtime > Change runtime type > GPU seç.
2. Bölüm numarasını ve varsa video bağlantısını gir. Hazırlama bölümündeki kod hücrelerini sırayla çalıştır.
3. Drive'da gösterilen `TRANSLATION_PACK.zip` dosyasını ChatGPT'ye ver.
4. Dönen `TRANSLATED.zip` dosyasını aynı `handoff` klasörüne koy.
5. `MODE = "finish"` yapıp hazırlama bölümündeki kod hücrelerini tekrar çalıştır. GPU gerekmez.
6. Taslak SRT ve kontrol listesini incele. Dinleme aracıyla işaretli yerleri düzelt.

**06:00'da kendiliğinden başlama kurulmuş değildir.** Colab oturumu ve Drive izni
telefondan açılır. GPU/kota/oturum süresi garanti değildir. Tamamlanan ASR parçaları
Drive'da kalır. GPU'yu çeviri beklerken Runtime > Disconnect and delete runtime ile
serbest bırak; devam etmek için notebook'u yeniden aç.

Bu bir üretim adayıdır: gerçek Colab GPU ve tam bölüm dinleme testi henüz yapılmadı.
Whisper kelime zamanları tahmindir. Yazı düzeltilince yeniden forced alignment yoktur;
yalnız dinlenerek yapılan, kaydı tutulan zaman düzeltmeleri uygulanır.
YouTube indirme engellenirse erişebildiğin kaynak videoyu Drive'a koyup SOURCE_FILE kullan.
'''
settings='''EPISODE = 15
MODE = "prepare"  # "prepare" veya "finish"
SOURCE_URL = ""  # Boşsa resmî kanalda tam bölüm başlığını arar.
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
    if not SOURCE_FILE:
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
setup='''from google.colab import drive
from pathlib import Path
import importlib, sys
drive.mount("/content/drive")
ROOT = Path(DRIVE_ROOT) / f"Muhtemel Ask {EPISODE}.Bolum"
CONFIG = Path("/content/ma_sub_config")
CONFIG.mkdir(exist_ok=True)
'''+ 'config_files = '+repr(config_files)+'\n'+'''for name,content in config_files.items():
    (CONFIG/name).write_text(content,encoding="utf-8")
if "/content" not in sys.path: sys.path.insert(0,"/content")
import ma_sub_colab as flow
flow = importlib.reload(flow)
RETURN = ROOT / "handoff" / f"Muhtemel Ask {EPISODE}.Bolum_TRANSLATED.zip"
print("Bölüm klasörü:",ROOT)
'''
run='''if MODE == "prepare":
    PACK = flow.prepare(ROOT, EPISODE, CONFIG, source_file=SOURCE_FILE, source_url=SOURCE_URL)
    print("ChatGPT'ye verilecek dosya:",PACK)
    print("Çeviri beklerken GPU oturumunu kapat. Drive'daki dosyalar korunur.")
elif MODE == "finish":
    report = flow.finalize(ROOT, RETURN)
    print("Kontrol listesi:",ROOT/"latest_output.json")
else:
    raise ValueError("MODE prepare veya finish olmalı")
'''
review='''# Yalnız çeviri döndükten sonra çalıştır.
flow.review_ui(ROOT, RETURN)
'''
preview='''# SRT'yi videoyla VLC gibi bir oynatıcıda da açabilirsin.
# Aşağıdaki isteğe bağlı hücre hızlı MKV üretir; videoyu yeniden kodlamaz.
MAKE_PREVIEW = False
BURN_IN_MP4 = False  # True: altyazıyı videoya gömer; CPU'da uzun sürebilir.
if MAKE_PREVIEW or BURN_IN_MP4:
    flow.mux_preview(ROOT, burn=BURN_IN_MP4)
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
