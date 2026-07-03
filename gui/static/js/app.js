function addLayer() {
    const template = document.getElementById('layer-row-template');
    const container = document.getElementById('layers-container');
    const clone = template.cloneNode(true);
    clone.removeAttribute('id');
    clone.querySelector('.layer-type-select').addEventListener('change', function() {
        updateLayerParams(clone, this.value);
    });
    container.appendChild(clone);
    updateLayerParams(clone, clone.querySelector('.layer-type-select').value);
}

function removeLayer(btn) {
    btn.closest('.layer-row').remove();
}

function updateLayerParams(row, layerType) {
    const existing = row.querySelector('.layer-params');
    if (existing) existing.remove();
    const paramDiv = document.createElement('div');
    paramDiv.className = 'layer-params';
    const layerParams = {
        'linear': 'in_features,out_features,dropout',
        'conv2d': 'in_channels,out_channels,kernel_size,padding,dropout',
        'maxpool2d': 'kernel_size,stride',
        'relu': '',
        'dropout': 'p',
        'batchnorm2d': 'num_features',
        'flatten': '',
        'lstm': 'input_size,hidden_size,num_layers,dropout',
        'linear': 'in_features,out_features,dropout',
        'softmax': 'dim',
    };
    const params = layerParams[layerType] || '';
    if (params) {
        paramDiv.innerHTML = '<label>Parameters (JSON):</label>' +
            '<input type="text" class="layer-params-input" placeholder="{\\"out_features\\": 256}" style="width:100%;padding:0.5rem;border:1px solid #e2e8f0;border-radius:6px;font-size:0.9rem;">';
    }
    row.appendChild(paramDiv);
}

function buildModelPreview() {
    const layers = getLayerConfig();
    fetch('/layers', {method: 'GET'}).then(r => r.json()).then(data => {
        console.log('Available layers:', data.layers);
    });
    alert('Model preview: ' + layers.length + ' layers configured.\\n\\n' + JSON.stringify(layers, null, 2));
}

function getLayerConfig() {
    const rows = document.querySelectorAll('#layers-container .layer-row');
    const config = [];
    rows.forEach(row => {
        const type = row.querySelector('.layer-type-select').value;
        const paramsInput = row.querySelector('.layer-params-input');
        let params = {};
        if (paramsInput && paramsInput.value) {
            try { params = JSON.parse(paramsInput.value); }
            catch(e) { params = {}; }
        }
        config.push({type: type, params: params});
    });
    return config;
}

async function startTraining() {
    const form = document.getElementById('train-form');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    data.model_config = JSON.stringify(getLayerConfig());
    data.use_pretrained = document.getElementById('use_pretrained').checked;
    if (data.use_pretrained) {
        data.pretrained_model = document.getElementById('pretrained_model').value;
    }
    document.getElementById('training-status').style.display = 'block';
    const statusText = document.getElementById('status-text');
    const progressFill = document.getElementById('progress-fill');
    const logsContainer = document.getElementById('logs-container');
    statusText.textContent = 'Starting training...';
    progressFill.style.width = '0%';
    logsContainer.innerHTML = '';
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.experiment_id) {
            statusText.textContent = 'Training started! Monitoring progress...';
            pollTrainingStatus(result.experiment_id);
        }
    } catch (error) {
        statusText.textContent = 'Error starting training: ' + error.message;
    }
}

async function pollTrainingStatus(experimentId) {
    const statusText = document.getElementById('status-text');
    const progressFill = document.getElementById('progress-fill');
    const logsContainer = document.getElementById('logs-container');
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/train/status/${experimentId}`);
            const data = await response.json();
            const status = data.status;
            if (status.status === 'running') {
                const progress = (status.epoch / status.total_epochs) * 100;
                progressFill.style.width = progress + '%';
                statusText.textContent = `Training... Epoch ${status.epoch}/${status.total_epochs}`;
            } else if (status.status === 'completed') {
                clearInterval(pollInterval);
                progressFill.style.width = '100%';
                statusText.textContent = 'Training completed!';
                if (data.result) {
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry';
                    logEntry.textContent = `Final validation accuracy: ${(data.result.evaluation.accuracy * 100).toFixed(2)}%`;
                    logsContainer.appendChild(logEntry);
                }
                setTimeout(() => {
                    window.location.href = `/results?experiment_id=${experimentId}`;
                }, 2000);
            } else if (status.status === 'error') {
                clearInterval(pollInterval);
                statusText.textContent = 'Training failed: ' + status.error;
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                logEntry.style.color = '#f87171';
                logEntry.textContent = 'Error: ' + status.error;
                logsContainer.appendChild(logEntry);
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

document.addEventListener('DOMContentLoaded', function() {
    addLayer();
});
