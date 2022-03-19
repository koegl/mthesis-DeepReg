# loading bar
from time import sleep
from tqdm import tqdm

for i in tqdm(range(10), desc="Description", ):
    sleep(2)
