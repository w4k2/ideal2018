all:
	python -W ignore process.py
	cp results/results.csv ../paper/

clean:
	rm plots/*
	rm results/*

readme:
	python readme.py > README.md
