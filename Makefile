all:
	python -W ignore process.py

clean:
	rm plots/*
	rm results/*

readme:
	python readme.py > README.md
