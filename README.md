# adorNER

This is a multi-ontology concept recognition model. It takes OBO ontologies as inputs, and trains a text classifier for identifying specific ontology terms in text. 

----------

### Requirements

- python==3.9.12

- tensorflow==2.9.1

- tensorflow-addons==0.18.0

- regex

- numpy

- A copy of the pre-trained [elmo model](https://tfhub.dev/google/elmo/3)

----------

### Training example

Example training parameter json:
```{json}
{
    "model_type": "sae",
    "obo_params" : "path/to/obo/parameter/file.json",
    "hub_dir": "directory/containing/elmo/model",
    "output": "directory/for/trained/model",
    "num_filters": 1024,
    "concept_dim": 1024,
    "compression_ratio": 8,
    "epochs": 10000,
    "mean": true
}
```

Example OBO parameter json:
```{json}
{
    "ontologies": [
        {   "name": "Human Phenotype Ontology",
            "obo_path": "ontologies/hpo.obo",
            "root_id": "HP:0000001",
            "main": 1
        },
        {   "name": "Mammalian Phenotype Ontology",
            "obo_path": "ontologies/mp.obo",
            "root_id": "MP:0000001",
            "main": 0
        }
    ]
}
```

To train model:
```{bash}
python adorNER/train.py --params path/to/training/parameter/file.json
```

-----

### Use trained model

Example annotation parameter json:
```
{
    "hub_dir": "elmo",
    "model_dir": "directory/of/trained/model",
    "input_dir": "directory/containing/txt/files",
    "output_dir": "directory/to/output/identified/ontology/terms",
    "threshold": 0.55,
    "max_rows": -1
}
```

To use model:
```
python adorNER/annotate_text.py --params path/to/annotation/parameter/file.json
```

