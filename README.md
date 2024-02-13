# Model description

* base: Standard Transformer
* denseformer: DenseFormer
* connect_to_last: Only a single DWA after the last layer
* base_w_gains: Adding learned gains (multiplication factor) to each skip connection.

## Quickstart 

Install dependencies: 

```
pip install -r requirements.txt
```

Running training is done by executing `./main.py`. The config arguments can be seen in `config/base.py`.
