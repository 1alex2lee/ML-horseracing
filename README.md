# ML-horseracing
Scripts to scrape data and train models to predict winner in horse races.

## Sequence of Code
1. hkjc-scrape will use Selenium WebDriver with Google Chrome to scrape data off the HKJC website.
2. hkjc-datacleanup will clean, format, and normalise the scraped data to be ready for feature selection and training.
3. hkjc-train, which was run on a remote device with CUDA, will train models with random architectures and hyper parameters in search for the best one. It calls model.py and train.py modules..
4. hkjc-evaluate-model will then find the performance of the model if bets were made using its predictions on the test set.
5. The best performing models are saved in hkjc-model-architectures.
6. hkjc-raceday-scrape will then scrape the race, horse, and jockey details when they release on the day of the races.
7. hkjc-raceday-update will scrape and update the odds of each horse as they update closer to the time of the races.
8. hkjc-use-model will then predict the order of the race result in horse number using the best performing model.

## Other Files
The other files contain my attempt in making the models predict a listwise ranking - considering the horses in a race co-dependently instead of independently, reflecting real life.

## Note
- Data is not in this repository, but the entire dataset spans across 7 years and has around 12,000 data points (a horse in a race).
- This repository is a combination of hkjc5, hkjc6, and hkjc7, with the numbers reflecting the iteration of the code. There are 8 iterations in total. 
