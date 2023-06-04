#!/bin/bash
#SBATCH --time=23:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000

python mt5_ft.py -lang en_XX zh_CN de_DE es_XX it_IT fa_IR ru_RU -form hyperbole idiom metaphor \
-prompt 'Which figure of speech does this text contain? (A) Literal. (B) {}. | Text: '