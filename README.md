# ma-sub

## Weekly workflow

1. Start RunPod and clone/update this repository.
2. `./runpod/bootstrap.sh`
3. `./mas run 13 --source-url "VIDEO_URL"`
4. Stop the GPU when the CLI says `GPU WORK COMPLETE`.
5. Give the generated TR pack and prompt to ChatGPT Pro, then place the returned ZIP in `translation_output/`.
6. `./mas run 13` resumes automatically.
7. Repeat the ID handoff.
8. Final TR/ID SRTs are written under `final/subtitles/`; QC is under `reports/`.

Commands: `./mas status 13`, `./mas doctor`, `./mas test`, `./mas clean 13 --dry-run`.

v0.1.0 is an offline/checkpoint/pack-validation baseline. It does not claim a real GPU full-episode run yet. The first RunPod run is the GPU integration test.
