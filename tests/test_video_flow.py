import pytest

from mas import colab_flow as f, video_flow as video


def ready(root, monkeypatch, sizes, reported_size=None):
    (root/'source.mp4').write_bytes(b'immutable source')
    (root/'episode_ID.srt').write_text('1\n00:00:00,000 --> 00:00:01,000\nHalo.\n')
    f.write_json(root/'source.json', dict(episode=15))
    f.write_json(root/'video_workflow.json', dict(route='asr', work_directory='.',
                 source_sha256=f.file_hash(root/'source.mp4')))
    monkeypatch.setattr(f, 'finalize', lambda *a: dict(issues=['audio review pending'],
        output_files=[dict(path='episode_ID.srt', sha256=f.file_hash(root/'episode_ID.srt'))]))
    calls=[]
    def burn(source, subtitle, output, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        size=sizes[len(calls)]
        output.write_bytes(b'x'*size)
        calls.append((output, kwargs))
        return dict(output_bytes=size if reported_size is None else reported_size)
    monkeypatch.setattr(video, 'burn_indonesian_mp4', burn)
    return calls


def test_size_equal_to_limit_retries_once_and_preserves_first_output(tmp_path, monkeypatch):
    calls=ready(tmp_path,monkeypatch,[100,95])
    result=video.finish(tmp_path,'return.zip',target_size_gb=90/1e9,max_output_bytes=100)
    assert result['size_verified'] and result['receipt']['output_bytes']==95
    assert result['status']=='BURNED_DRAFT'
    assert len(calls)==2 and calls[0][0]!=calls[1][0]
    assert calls[0][0].stat().st_size==100
    assert calls[1][1]['target_size_gb']<calls[0][1]['target_size_gb']
    assert calls[1][1]['timeout_seconds']<=calls[0][1]['timeout_seconds']


def test_oversized_second_result_is_not_delivered_or_retried_forever(tmp_path, monkeypatch):
    calls=ready(tmp_path,monkeypatch,[105,101])
    with pytest.raises(f.ContractError,match='not deliverable'):
        video.finish(tmp_path,'return.zip',target_size_gb=90/1e9,max_output_bytes=100)
    assert len(calls)==2 and not (tmp_path/'video_output.json').exists()
    assert len(f.read_json(tmp_path/'video_size_failure.json')['attempts'])==2


def test_actual_output_bytes_must_match_receipt(tmp_path, monkeypatch):
    ready(tmp_path,monkeypatch,[105],reported_size=95)
    with pytest.raises(f.ContractError,match='size changed'):
        video.finish(tmp_path,'return.zip',target_size_gb=90/1e9,max_output_bytes=100)


def test_normal_output_does_not_spend_gpu_on_second_encode(tmp_path, monkeypatch):
    calls=ready(tmp_path,monkeypatch,[95])
    result=video.finish(tmp_path,'return.zip',target_size_gb=90/1e9,max_output_bytes=100)
    assert len(calls)==1 and result['size_verified']
