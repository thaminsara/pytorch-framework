import os
import sys
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import threading
import time
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import Engine
from src.model_builder import ModelBuilder, AutoModelBuilder
from src.layers import LayerRegistry
from src.losses import get_loss
from src.metrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AverageMeter
from src.visualization import Visualizer
from src.datasets import DatasetManager
from src.optimizers import get_optimizer, get_scheduler
from src.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import ensure_dir, get_device, count_parameters

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "pytorch-framework-secret"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ensure_dir(app.config["UPLOAD_FOLDER"])
ensure_dir(os.path.join(os.path.dirname(__file__), "static", "plots"))

training_status = {}
training_results = {}


@app.route("/")
def index():
    layers = LayerRegistry.list_layers()
    optimizers = ["sgd", "adam", "adamw", "rmsprop", "adagrad", "adadelta"]
    schedulers = ["none", "step", "multistep", "exponential", "cosine", "reduce_on_plateau", "cyclic", "onecycle"]
    losses = ["cross_entropy", "bce", "mse", "mae", "focal", "label_smoothing", "dice", "dice_bce"]
    metrics = ["accuracy", "precision", "recall", "f1", "iou", "confusion_matrix"]
    return render_template("index.html", layers=layers, optimizers=optimizers,
                           schedulers=schedulers, losses=losses, metrics=metrics)


@app.route("/models")
def models():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    saved_models = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pt") or f.endswith(".pth"):
                filepath = os.path.join(models_dir, f)
                saved_models.append({
                    "name": f,
                    "size": os.path.getsize(filepath),
                    "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M"),
                })
    return render_template("models.html", saved_models=saved_models)


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "GET":
        layers = LayerRegistry.list_layers()
        return render_template("train.html", layers=layers)
    data = request.get_json()
    experiment_id = str(uuid.uuid4())
    training_status[experiment_id] = {"status": "running", "epoch": 0, "total_epochs": int(data.get("epochs", 10))}
    thread = threading.Thread(target=_run_training, args=(experiment_id, data))
    thread.start()
    return jsonify({"experiment_id": experiment_id, "status": "started"})


def _run_training(experiment_id, data):
    try:
        batch_size = int(data.get("batch_size", 32))
        epochs = int(data.get("epochs", 10))
        lr = float(data.get("learning_rate", 0.001))
        optimizer_name = data.get("optimizer", "adam")
        scheduler_name = data.get("scheduler", "none")
        loss_name = data.get("loss", "cross_entropy")
        device = data.get("device", "auto")
        dataset_name = data.get("dataset", "cifar10")
        model_config = json.loads(data.get("model_config", "[]"))
        use_pretrained = data.get("use_pretrained", False)
        pretrained_model = data.get("pretrained_model", "")
        dm = DatasetManager()
        dataloaders = dm.get_common_dataset(dataset_name, batch_size=batch_size)
        num_classes = dm.num_classes
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        if use_pretrained and pretrained_model:
            import torchvision.models as models
            model_fn = getattr(models, pretrained_model, None)
            if model_fn:
                model = model_fn(weights="IMAGENET1K_V1" if "pretrained" in data else None)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            else:
                model = AutoModelBuilder.build_cnn(input_channels=3, num_classes=num_classes, img_size=32)
        else:
            builder = ModelBuilder(input_shape=[3, 32, 32])
            for layer_conf in model_config:
                builder.add_layer(layer_conf["type"], **layer_conf.get("params", {}))
            model = builder.build()
        criterion = get_loss(loss_name)
        if loss_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        model.to(get_device(device))
        optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr)
        scheduler = get_scheduler(scheduler_name, optimizer) if scheduler_name != "none" else None
        engine = Engine(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        device=device, config={"paths": {"plots_dir": os.path.join(os.path.dirname(__file__), "static", "plots")}})
        cb_es = EarlyStopping(monitor="val_loss", patience=max(5, epochs // 4), verbose=False)
        engine.add_callback(cb_es)
        history = engine.fit(epochs)
        eval_results = engine.evaluate()
        plots = engine.visualizer.plot_all_metrics(
            history, model=model, input_shape=[3, 32, 32],
            class_names=dm.classes
        )
        model_path = os.path.join(os.path.dirname(__file__), "..", "models",
                                  f"model_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        engine.save_model(model_path)
        training_results[experiment_id] = {
            "status": "completed",
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "evaluation": eval_results,
            "plots": {k: os.path.basename(v) for k, v in plots.items()},
            "model_path": os.path.basename(model_path),
            "model_summary": engine.get_model_summary(),
            "num_classes": num_classes,
            "class_names": dm.classes,
        }
        training_status[experiment_id] = {"status": "completed", "epoch": epochs, "total_epochs": epochs}
    except Exception as e:
        training_status[experiment_id] = {"status": "error", "error": str(e)}


@app.route("/train/status/<experiment_id>")
def train_status(experiment_id):
    status = training_status.get(experiment_id, {"status": "unknown"})
    result = training_results.get(experiment_id)
    return jsonify({"status": status, "result": result})


@app.route("/visualize")
def visualize():
    experiment_id = request.args.get("experiment_id")
    result = training_results.get(experiment_id) if experiment_id else None
    all_results = list(training_results.keys())
    return render_template("visualize.html", result=result, experiment_id=experiment_id,
                           all_results=all_results)


@app.route("/results")
def results():
    experiment_id = request.args.get("experiment_id")
    if experiment_id and experiment_id in training_results:
        result = training_results[experiment_id]
    else:
        result = None
    return render_template("results.html", result=result, experiment_id=experiment_id)


@app.route("/static/plots/<filename>")
def serve_plot(filename):
    plots_dir = os.path.join(os.path.dirname(__file__), "static", "plots")
    return send_file(os.path.join(plots_dir, filename), mimetype="image/png")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model_path = data.get("model_path")
    image_data = data.get("image_data")
    if not model_path or not image_data:
        return jsonify({"error": "Missing model_path or image_data"}), 400
    model_file = os.path.join(os.path.dirname(__file__), "..", "models", os.path.basename(model_path))
    if not os.path.exists(model_file):
        return jsonify({"error": "Model file not found"}), 404
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    from src.model_builder import ModelBuilder
    input_shape = checkpoint.get("config", {}).get("model", {}).get("input_shape", [3, 32, 32])
    builder = ModelBuilder(input_shape=input_shape)
    model_config = checkpoint.get("config", {}).get("model_config", [])
    for layer_conf in model_config:
        builder.add_layer(layer_conf["type"], **layer_conf.get("params", {}))
    model = builder.build()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    import base64
    import io
    from PIL import Image
    img_data = base64.b64decode(image_data.split(",")[1])
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_shape[1], input_shape[2])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    return jsonify({"prediction": pred_class, "confidence": float(confidence)})


@app.route("/layers")
def list_layers():
    return jsonify({"layers": LayerRegistry.list_layers()})


@app.route("/layer_info/<layer_name>")
def layer_info(layer_name):
    layer_class = LayerRegistry.get(layer_name)
    import inspect
    sig = inspect.signature(layer_class.__init__)
    params = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params.append({
            "name": name,
            "default": str(param.default) if param.default is not inspect.Parameter.empty else "required",
            "type": str(param.annotation) if param.annotation is not inspect.Parameter.empty else "any",
        })
    return jsonify({"layer": layer_name, "params": params})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
