# Joint Syntaco-Discourse Parsing and Treebank

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](label.png)

This repository contains the implementation of the joint syntaco-discourse parser and the syntaco-discourse treebank. For more details, please refer to the paper [Joint Syntacto-Discourse Parsing and the Syntacto-Discourse Treebank](http://aclweb.org/anthology/D/D17/D17-1224.pdf).

### Syntaco-Discourse Treebank

Due to copyright restriction, we can not provide the joint treebank in the form that can be directly used to train a parser. Instead, we provide a patch tool kit to generate the Syntaco-Discourse Treebank giving the [RST Discourse Treebank](https://catalog.ldc.upenn.edu/ldc2002t07) and the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42).

#### Required Python Dependencies

1. ```python-gflags``` for parsing script arguments.

2. ```nltk``` for tokenization.

#### Procedures to Generate Treebank

Please follow the steps below to generate the treebank:

1. Place the RST Discourse Treebank in folder ```dataset/rst```. Put the discourse trees (```wsj_xxxx.out.dis``` files) in the RST Discourse Treebank to ```dataset/rst/train``` and ```dataset/rst/test``` respectively. Here each ```wsj_xxxx.out.dis``` file corresponds to one WSJ article, where ```xxxx``` is the article number.

2. Place the Penn Treebank trees in folder ```dataset/ptb```. These constituency trees are in parentheses format. They are grouped as one treebank file (with name ```wsj_xxxx.cleangold```) for a WSJ article.

3. Apply patches to the RST Discourse Treebank files and Penn Treebank files. This step is necessary because there are some small mismatches between the RST Discourse tree texts and the Penn tree texts.
	```bash
	cd dataset/rst/train
	patch -p0 < ../../../patches/rst-ptb.train.patch
	cd ../test
	patch -p0 < ../../../patches/rst-ptb.test.patch
	cd ../../ptb
	patch -p0 < ../../patches/ptb-rst.patch
	cd ...
	```

4. Run tokenization.
	```bash
	python src/tokenize_rst.py --rst_path dataset/rst/train
	python src/tokenize_rst.py --rst_path dataset/rst/test
	```

3. Generate the training set and testing set for the joint treebank separately:

   ```bash
   mkdir dataset/joint
   python src/aligner.py --rst_path dataset/rst/train --const_path dataset/ptb > dataset/joint/train.txt
   python src/aligner.py --rst_path dataset/rst/test --const_path dataset/ptb > dataset/joint/test.txt
   ```


To test the scripts above, you can play with the sample data:

   ```bash
   python src/tokenize_rst.py --rst_path sampledata/rst
   python src/aligner.py --rst_path sampledata/rst --const_path sampledata/ptb > sampledata/joint.txt
   ```

### Syntaco-Dsicourse Parser

#### Required Python Dependencies

Since the joint parser is based on the [Span-based Constituency Parser](https://github.com/jhcross/span-parser), please have the following required dependencies installed:

1. [```DyNet```](http://dynet.readthedocs.io/en/latest/python.html) for the underlying neural model.

2. ```numpy``` for interacting with DyNet.

#### Training

To train on the provided sample data, you can simply run:

```bash
mkdir exps
python src/trainer.py --train sampledata/joint.txt --dev sampledata/joint.txt --epoch 200 --save exps/sampledata.model
```

You can find the training parameters and their descriptions in ```src/trainer.py```.

#### Evaluating

To evaluate the trained model on the sample data, you can run:

```bash
python src/parser.py --train sampledata/joint.txt --test sampledata/joint.txt --model exps/sampledata.model --verbose
```