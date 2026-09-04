import argparse,shutil,subprocess,sys
from .pipeline import run,status
from .config import episode_dir
def doctor():
 print('mas doctor')
 for x in ('python','ffmpeg','ffprobe','git'): print(f"{x}: {'OK' if shutil.which(x) else 'MISSING'}")
 try:
  import torch; print(f'torch: {torch.__version__}; cuda={torch.cuda.is_available()}')
 except Exception: print('torch: optional/not installed in CPU dev environment')
 return 0
def test(): return subprocess.call([sys.executable,'-m','pytest','-q'])
def clean(ep,destroy=False):
 d=episode_dir(ep)
 for p in [d/'work']: print(('DELETE ' if destroy else 'DRY-RUN ')+str(p)); shutil.rmtree(p,ignore_errors=True) if destroy else None
 return 0
def main(argv=None):
 p=argparse.ArgumentParser(prog='mas'); sub=p.add_subparsers(dest='cmd',required=True); r=sub.add_parser('run'); r.add_argument('episode',type=int); r.add_argument('--source-url'); r.add_argument('--fixture',action='store_true'); s=sub.add_parser('status'); s.add_argument('episode',type=int); sub.add_parser('doctor'); sub.add_parser('test'); c=sub.add_parser('clean'); c.add_argument('episode',type=int); c.add_argument('--dry-run',action='store_true',default=True); c.add_argument('--destroy',action='store_true'); a=p.parse_args(argv)
 try: return {'run':lambda:run(a.episode,a.source_url,a.fixture),'status':lambda:status(a.episode),'doctor':doctor,'test':test,'clean':lambda:clean(a.episode,a.destroy)}[a.cmd]()
 except Exception as e:
  print(f'FAILED STAGE: {a.cmd.upper()}\nCAUSE: {e}\nCHECKPOINT PRESERVED: yes',file=sys.stderr)
  if getattr(a,'episode',None): print(f'SAFE RETRY:\n./mas run {a.episode}',file=sys.stderr)
  return 1
if __name__=='__main__': raise SystemExit(main())
