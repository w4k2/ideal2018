all:
	python -W ignore process.py
	cp results/results.csv ../paper/
	cp -r plots ../paper/

clean:
	rm plots/*
	rm results/*

readme:
	python readme.py > README.md
