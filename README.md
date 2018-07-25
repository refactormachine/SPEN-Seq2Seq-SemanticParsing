# SPEN-Seq2Seq-SemanticParsing

## First time configuration:  
``` bash
git clone https://github.com/Verose/SPEN-Seq2Seq-SemanticParsing.git
cd SPEN-Seq2Seq-SemanticParsing
apt install libpq-dev
pip install -r requirements.txt
cd data
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove.6B
wget https://nlp.stanford.edu/projects/scone/scone.zip
unzip scone.zip
cd ..
sh run_tangrams.sh
```
