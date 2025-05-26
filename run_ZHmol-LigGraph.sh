#!/bin/bash

if [ ! -d "data/after_dock" ]; then
    mkdir -p data/after_dock
fi

folder_count=$(find data/native -mindepth 1 -maxdepth 1 -type d | wc -l)

if [ "$folder_count" -gt 1 ]; then
    while IFS= read -r line; do
        complex="${line:0:4}"

        if [ ! -f "data/native/${complex}/${complex}.lig.mol2" ]; then
            echo "Error: ${complex}.lig.mol2 not found in data/native/${complex}/"
            continue
        fi

        if [ ! -f "data/native/${complex}/${complex}.rec.pdb" ]; then
            echo "Error: ${complex}.rec.pdb not found in data/native/${complex}/"
            continue
        fi

        if [ ! -f "data/complex_before/${complex}.pdb" ]; then
            echo "Error: ${complex}.pdb not found in data/complex_before/"
            continue
        fi

        python ZHmol-LigGraph.py --ligand_file=data/native/${complex}/${complex}.lig.mol2 \
            --nucleic_file=data/native/${complex}/${complex}.rec.pdb \
            --pose_file=data/complex_before/${complex}.pdb \
            --output_file=ZHmol-LigGraph_tmp/${complex}.mol2 \
            --prediction_model=models/prediction_model.pt \
            --selection_model=models/selection_model.pt \
            --tmp_dir=ZHmol-LigGraph_tmp/

        mkdir -p data/after_dock/${complex}
        cp ZHmol-LigGraph_tmp/${complex}* data/after_dock/${complex}/
        rm -rf ZHmol-LigGraph_tmp
        echo "${complex} is done"
    done < data/pdb_list_test

else
    complex=$(basename $(find data/native -mindepth 1 -maxdepth 1 -type d))
    
    if [ ! -f "data/native/${complex}/${complex}.lig.mol2" ]; then
        echo "Error: ${complex}.lig.mol2 not found in data/native/${complex}/"
        exit 1
    fi

    if [ ! -f "data/native/${complex}/${complex}.rec.pdb" ]; then
        echo "Error: ${complex}.rec.pdb not found in data/native/${complex}/"
        exit 1
    fi

    if [ ! -f "data/complex_before/${complex}.pdb" ]; then
        echo "Error: ${complex}.pdb not found in data/complex_before/"
        exit 1
    fi

    python ZHmol-LigGraph.py --ligand_file=data/native/${complex}/${complex}.lig.mol2 \
        --nucleic_file=data/native/${complex}/${complex}.rec.pdb \
        --pose_file=data/complex_before/${complex}.pdb \
        --output_file=ZHmol-LigGraph_tmp/${complex}.mol2 \
        --prediction_model=models/prediction_model.pt \
        --selection_model=models/selection_model.pt \
        --tmp_dir=ZHmol-LigGraph_tmp/
    
    mkdir -p data/after_dock/${complex}
    cp ZHmol-LigGraph_tmp/${complex}* data/after_dock/${complex}/
    rm -rf ZHmol-LigGraph_tmp
    echo "${complex} is done"
fi

