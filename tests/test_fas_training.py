from dataclasses import asdict
import csv
import json
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torchvision")
pytest.importorskip("sklearn")

from lift_classify import LiFTClassifier, LiFTConfig
from lift_classify.fas_data import FASTransform, FASDataset, discover_records
from lift_classify.fas_runtime import (classification_metrics, evaluate, is_better, load_model,
                                      read_checkpoint, resolve_device)
from lift_classify import fas_training as training
from test_fas_data import fas_root


@pytest.fixture
def run_config():
    return training.validate_config(dict(
        model=asdict(LiFTConfig(image_size=16,patch_size=4,hidden_dim=16,num_heads=2,
                                prelude_depth=1,core_depth=1,num_loops=2,num_classes=2,dropout=.1)),
        training=dict(epochs=2,warmup_epochs=0,batch_size=3,num_workers=0)))


def test_metric_direction_threshold_and_ties():
    result=classification_metrics([0,0,1,1],[.1,.5,.2,.8])
    assert result["apcer"] == .5 and result["bpcer"] == .5 and result["acer"] == .5
    assert result["accuracy"] == .5 and result["roc_auc"] == .75
    assert is_better(dict(acer=.1,loss=.4),dict(acer=.1,loss=.5))
    assert not is_better(dict(acer=.1,loss=.4),dict(acer=.1,loss=.4))
    assert not is_better(dict(acer=.2,loss=.1),dict(acer=.1,loss=.5))
    assert classification_metrics([1,1],[.5,.8])["acer"] is None
    with pytest.raises(ValueError):classification_metrics([0,1],[float("nan"),.5])


def test_device_priority_and_explicit_failures(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.backends.mps,"is_available",lambda:True)
    assert str(resolve_device()) == "cuda"
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert str(resolve_device()) == "mps"
    with pytest.raises(ValueError,match="CUDA"):resolve_device("cuda")
    monkeypatch.setattr(torch.backends.mps,"is_available",lambda:False)
    assert str(resolve_device()) == "cpu"
    with pytest.raises(ValueError,match="MPS"):resolve_device("mps")


def test_class_weights_optimizer_and_schedule(run_config):
    weights=training.class_weights([dict(label=0)]*4+[dict(label=1)]*2)
    torch.testing.assert_close(weights,torch.tensor([.75,1.5]))
    logits=torch.tensor([[1.,0.],[1.,0.]])
    loss=F.cross_entropy(logits,torch.tensor([0,1]),weight=weights)
    expected=-(logits.log_softmax(-1)[0,0]*.75+logits.log_softmax(-1)[1,1]*1.5)/2.25
    torch.testing.assert_close(loss,expected)
    model=LiFTClassifier(LiFTConfig(**run_config["model"]))
    optimizer=training.make_optimizer(model,run_config["training"])
    decay_ids={id(p) for p in optimizer.param_groups[0]["params"]}
    assert id(model.base_condition) not in decay_ids
    assert id(model.core.blocks[0].attn.q_norm.weight) not in decay_ids
    assert id(model.patch_embedding.weight) in decay_ids
    assert id(model.patch_embedding.bias) not in decay_ids
    factors=[training.lr_factor(i,training.TRAIN_DEFAULTS) for i in range(100)]
    assert factors[0] == .2 and factors[4] == 1 and factors[5] == 1
    assert factors[-1] == pytest.approx(.01)
    assert all(a>=b for a,b in zip(factors[5:],factors[6:]))


def test_worker_seeds_reproduce_epoch_augmentations(fas_root,run_config):
    from lift_classify.fas_data import preprocessing_spec
    dataset=FASDataset(fas_root,discover_records(fas_root,"train"),
                       FASTransform(preprocessing_spec(16),training=True))
    settings=run_config["training"] | {"num_workers":2}
    first=list(training.make_loader(dataset,settings,torch.device("cpu"),training=True,epoch=1))
    second=list(training.make_loader(dataset,settings,torch.device("cpu"),training=True,epoch=1))
    for a,b in zip(first,second):
        torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
        assert a[2] == b[2]


def test_train_resume_matches_uninterrupted_and_cli_roundtrip(fas_root,run_config,tmp_path,monkeypatch):
    full, resumed=tmp_path/"full",tmp_path/"resumed"
    training.run_training(run_config,fas_root,full,"cpu")
    original_save=training.save_checkpoint
    def interrupt(path,payload):
        original_save(path,payload)
        if path.name == "last.pt" and payload["epoch"] == 1:
            raise KeyboardInterrupt("simulated epoch boundary")
    monkeypatch.setattr(training,"save_checkpoint",interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.run_training(run_config,fas_root,resumed,"cpu")
    assert (resumed/"last.pt").exists() and not (resumed/"best.pt").exists()
    monkeypatch.setattr(training,"save_checkpoint",original_save)
    report=training.run_training(run_config,fas_root,resumed,"cpu",resumed/"last.pt")
    a,b=read_checkpoint(full/"last.pt"),read_checkpoint(resumed/"last.pt")
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key],b["model"][key],atol=0,rtol=0)
    assert a["scheduler"] == b["scheduler"]
    assert a["best_metrics"] == b["best_metrics"]
    assert report["default_num_loops"] == 2
    assert set(report["metrics_by_num_loops"]) == {"1","2","4","8"}
    assert b["preprocessing"]["image_size"] == 16 and b["class_to_idx"] == {"fake":0,"live":1}
    assert b["environment"]["training_precision"] == "fp32"
    model,checkpoint=load_model(resumed/"best.pt",torch.device("cpu"))
    dataset=FASDataset(fas_root,discover_records(fas_root,"test"),FASTransform(checkpoint["preprocessing"]))
    metrics,_=evaluate(model,training.make_loader(dataset,run_config["training"],torch.device("cpu")),torch.device("cpu"))
    assert metrics == report["metrics_by_num_loops"]["2"]
    events = []
    progress_metrics, _ = evaluate(
        model, training.make_loader(dataset, run_config["training"], torch.device("cpu")),
        torch.device("cpu"), batch_progress=lambda **event: events.append(event))
    assert progress_metrics == metrics
    assert [event["step"] for event in events] == [1, 2]
    assert events[0]["acer"] is None  # First batch contains only fake images.
    assert events[-1]["examples"] == len(dataset)
    for key in ("loss", "accuracy", "acer"):
        assert events[-1][key] == pytest.approx(metrics[key])
    prediction=tmp_path/"predictions.csv"
    subprocess.run([sys.executable,"-m","tools.predict","--checkpoint",str(resumed/"best.pt"),
                    "--input",str(fas_root/"test/live/0.png"),"--output",str(prediction),"--device","cpu"],check=True)
    with prediction.open(newline="") as stream:
        single_rows = list(csv.DictReader(stream))
    assert single_rows[0]["path"] == "0.png"
    assert "p_live" in single_rows[0]
    directory_prediction = tmp_path / "directory_predictions.csv"
    subprocess.run([sys.executable, "-m", "tools.predict", "--checkpoint", str(resumed/"best.pt"),
                    "--input", str(fas_root/"test"), "--output", str(directory_prediction),
                    "--device", "cpu"], check=True)
    with directory_prediction.open(newline="") as stream:
        directory_rows = list(csv.DictReader(stream))
    expected_paths = {r["relative_path"].removeprefix("test/")
                      for r in discover_records(fas_root, "test")}
    assert {r["path"] for r in directory_rows} == expected_paths
    assert {"fake/0.png", "live/0.png"} <= expected_paths
    assert str(fas_root) not in directory_prediction.read_text()
    evaluation_dir=tmp_path/"evaluation"
    subprocess.run([sys.executable,"-m","tools.evaluate","--checkpoint",str(resumed/"best.pt"),
                    "--data-root",str(fas_root),"--output-dir",str(evaluation_dir),"--device","cpu",
                    "--batch-size","3"],check=True)
    evaluated=json.loads((evaluation_dir/"validation_report.json").read_text())
    assert evaluated["metrics_by_num_loops"]["2"]["acer"] == metrics["acer"]
    assert evaluated["checkpoint"] == "best.pt"
    assert "data_root" not in evaluated
    assert str(tmp_path) not in (evaluation_dir/"validation_report.json").read_text()
    subprocess.run([sys.executable,"tools.train.py","--resume",str(resumed/"last.pt"),
                    "--data-root",str(fas_root),"--output-dir",str(resumed),"--device","cpu"],check=True)
    with pytest.raises(ValueError,match="not empty"):
        training.run_training(run_config,fas_root,resumed,"cpu")
    changed=json.loads(json.dumps(run_config));changed["training"]["epochs"]=3
    with pytest.raises(ValueError,match="same"):
        training.run_training(changed,fas_root,resumed,"cpu",resumed/"last.pt")
    from PIL import Image
    Image.new("RGB",(16,16),(123,45,67)).save(fas_root/"train/live/0.png")
    with pytest.raises(ValueError,match="Dataset contents changed"):
        training.run_training(run_config,fas_root,resumed,"cpu",resumed/"last.pt")


@pytest.mark.parametrize("failure", ["before_last", "before_best"])
def test_checkpoint_commit_survives_interrupted_improving_epoch(fas_root,run_config,tmp_path,monkeypatch,failure):
    output=tmp_path/"run"
    original_save=training.save_checkpoint
    monkeypatch.setattr(training,"is_better",lambda metrics,best:True)
    def interrupt(path,payload):
        if payload["epoch"]==2 and path.name == ("last.pt" if failure=="before_last" else "best.pt"):
            raise KeyboardInterrupt("simulated checkpoint write failure")
        original_save(path,payload)
    monkeypatch.setattr(training,"save_checkpoint",interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.run_training(run_config,fas_root,output,"cpu")
    committed=read_checkpoint(output/"last.pt")
    assert committed["epoch"] == (1 if failure=="before_last" else 2)
    monkeypatch.setattr(training,"save_checkpoint",original_save)
    training.run_training(run_config,fas_root,output,"cpu",output/"last.pt")
    final=read_checkpoint(output/"last.pt")
    exported=read_checkpoint(output/"best.pt")
    assert exported["epoch"] == final["best_epoch"] == 2
    for key in final["best_model"]:
        torch.testing.assert_close(exported["model"][key],final["best_model"][key],atol=0,rtol=0)


@pytest.mark.parametrize("field,value",[("batch_size",0),("warmup_epochs",100),("precision","fp16"),
                                      ("learning_rate",float("nan")),("num_workers",-1)])
def test_config_validation(run_config,field,value):
    run_config["training"][field]=value
    with pytest.raises(ValueError):training.validate_config(run_config)


@pytest.mark.parametrize("device_name",["cuda","mps"])
def test_available_accelerator_inference(device_name):
    available=torch.cuda.is_available() if device_name=="cuda" else torch.backends.mps.is_available()
    if not available:pytest.skip(f"{device_name} unavailable")
    config=LiFTConfig(image_size=16,patch_size=4,hidden_dim=16,num_heads=2,num_classes=2)
    model=LiFTClassifier(config).eval()
    from conftest import activate_modulation
    activate_modulation(model)
    images=torch.randn(2,3,16,16)
    with torch.no_grad():
        cpu=model(images)
        actual=model.to(device_name)(images.to(device_name)).cpu()
    torch.testing.assert_close(actual,cpu,atol=1e-4,rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bfloat16_training_step(run_config):
    if not torch.cuda.is_bf16_supported():pytest.skip("CUDA BF16 unavailable")
    model=LiFTClassifier(LiFTConfig(**run_config["model"])).cuda()
    from conftest import activate_modulation
    activate_modulation(model)
    optimizer=training.make_optimizer(model,run_config["training"])
    before=model.patch_embedding.weight.detach().clone()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits=model(torch.randn(2,3,16,16,device="cuda"))
    loss=F.cross_entropy(logits.float(),torch.tensor([0,1],device="cuda"))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before,model.patch_embedding.weight)
