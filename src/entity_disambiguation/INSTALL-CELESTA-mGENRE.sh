#source ~/anaconda3/etc/profile.d/conda.sh
if [ -d "GENRE" ] 
then
    # Copying  
    cp -R CELESTA-mGENRE/ GENRE/
else

	# installing GenerationEval from WebNLG2020
	git clone https://github.com/facebookresearch/GENRE.git

	cd GENRE

	# installing dependecies
	pip install .
	# Install other dependencies (excluding kilt)
    pip install -r requirements.txt || echo "Skip kilt installation from requirements.txt"

    # Install kilt manually from GitHub
    git clone https://github.com/facebookresearch/KILT.git
    cd KILT
    pip install .
    
    cd ..
    mkdir models
    cd models
    wget -c https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz
    tar -xvf fairseq_multilingual_entity_disambiguation.tar.gz
    cd ..
    mkdir data
    cd data
    wget -c https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl
    wget -c http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl
    cd ../..
    cp -R CELESTA-mGENRE/ GENRE/
    cd ..
fi

# back to root folder 
cd ..