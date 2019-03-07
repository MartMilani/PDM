#! /bin/bash

mkdir -p PointDistFiles
cd PointDistFiles

## Lebedev grids
mkdir -p lebedev
cd lebedev
wget -r -np -nd -A txt -e robots=off https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html

cd ..

## Spherical designs
mkdir -p sphdesigns
cd sphdesigns
## Get spherical designs from Hardin & Sloane
mkdir -p HardinSloane
cd HardinSloane

wget -r -np -nd -A txt http://neilsloane.com/sphdesigns/dim3/

for file in des*.txt; do
    i=${file%*.txt}
    i=${i:6}
    NODES=$(printf "%05d" ${i%.*})
    ORDER=$(printf "hs%03d" ${i#*.})
    mv ${file} $(printf $ORDER.$NODES)
done

cd ..

python ../../reshape_coord.py

## Get spherical designs from Womersley (large file sizes!)
wget -r -nd -A zip http://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/index.html --no-check-certificate

unzip -n \*.zip
rm *.zip

mkdir WomersleySym
mkdir WomersleyNonSym

mv SS* ./WomersleySym
mv SF* ./WomersleyNonSym

cd ../..
